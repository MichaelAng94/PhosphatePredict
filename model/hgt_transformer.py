# model_hgt_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData


class SiteMotifEncoder(nn.Module):
    """位点序列窗口(motif)编码器。

    输入整数编码的氨基酸窗口 [num_site, motif_len]（0=填充/未知，1..20=标准氨基酸，
    见 data_prep/add_motif_features.py），经 token 嵌入 + 位置嵌入和一层 Transformer
    后做中心残基条件化读出（中心残基表示 + attention 池化上下文），
    输出 [num_site, hidden_dim]，并入位点节点表示。
    """
    def __init__(self, motif_len: int, hidden_dim: int, vocab_size: int = 21,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, motif_len, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden_dim)
        # P3 中心残基条件化读出：[中心残基表示 ‖ attention 池化上下文] 线性融合
        self.center = motif_len // 2
        self.attn_w = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.readout = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        pad_mask = window == 0                      # [N, L]，True 表示填充
        # 全填充行放开首个位置，避免该行注意力全部被掩码而产生 NaN
        fully_padded = pad_mask.all(dim=1)
        if fully_padded.any():
            pad_mask[fully_padded, 0] = False
        h = self.token_embed(window) + self.pos_embed[:, :window.size(1)]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        # P3: 中心残基（磷酸化残基本身）单独保留，上下文用 attention 池化聚合，
        # 避免均值池化稀释决定特异性的中心 ±1~3 位信号
        c = min(self.center, window.size(1) - 1)
        h_center = h[:, c]                                   # [N, H]
        scores = (h @ self.attn_w).masked_fill(pad_mask, float('-inf'))
        attn = torch.softmax(scores, dim=1)                  # [N, L]（全填充行已放开首位置）
        h_ctx = (h * attn.unsqueeze(-1)).sum(dim=1)          # [N, H]
        return self.readout(torch.cat([h_center, h_ctx], dim=-1))


class HGTTransformer(nn.Module):
    """
    联合 HGT + Transformer 模型：
    - 输入：HeteroData 对象，包含：
        * kinase: esm_x, n2v_x 两种特征
        * site: x 特征；可选 window（整数编码的序列窗口，由 SiteMotifEncoder 编码后相加融合）
        * 边：('kinase','ppis','kinase'), ('kinase','phosphorylates','site'), ('site','rev_phosphorylates','kinase')
    - 输出：激酶节点和底物节点的最终嵌入（可用于链接预测或节点分类）
    """
    def __init__(
        self,
        esm_dim: int,
        n2v_dim: int,
        site_dim: int,
        num_families: int,               # 激酶家族类别数
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_hgt_layers: int = 2,
        num_transformer_layers: int = 1,
        dropout: float = 0.2,
        motif_len: int = 0,              # 序列窗口长度；0 表示不使用 motif 特征
        motif_vocab: int = 21,           # 0=填充/未知, 1..20=标准氨基酸
    ):
        super().__init__()
        self.motif_len = motif_len

        # ---- 输入投影：将不同特征映射到 hidden_dim ----
        self.kinase_proj = Linear(esm_dim + n2v_dim, hidden_dim)
        self.site_proj = Linear(site_dim, hidden_dim)
        if motif_len > 0:
            self.motif_encoder = SiteMotifEncoder(
                motif_len, hidden_dim,
                vocab_size=motif_vocab, num_heads=num_heads, dropout=dropout,
            )

        # ---- HGT 层（异构消息传递） ----
        # 定义元数据：节点类型和边类型
        self.metadata = (
            ['kinase', 'site'],
            [
                ('kinase', 'ppis', 'kinase'),
                ('kinase', 'phosphorylates', 'site'),
                ('site', 'rev_phosphorylates', 'kinase'),
            ]
        )
        # 构建多个 HGTConv 层
        self.hgt_layers = nn.ModuleList()
        for _ in range(num_hgt_layers):
            self.hgt_layers.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=self.metadata,
                    heads=num_heads
                )
            )

        # ---- Transformer Encoder 层（全局节点自注意力） ----
        # 对所有节点（kinase + site）统一进行 Transformer 编码
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        # 激酶家族分类器（MLP）
        self.family_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_families)
        )

        # 关系打分器：MLP over [h_k; h_s; h_k*h_s; |h_k-h_s|]，
        # 比可对角加权点积(DistMut式)表达力更强，能建模非对称的激酶-位点交互
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # A1 家族原型先验分支：family_prototypes 第 f 行 = 家族 f 的 motif 偏好向量。
        # 参数全家族共享（C×H，无逐激酶自由度），结构上无法记忆单个激酶；
        # prior_weight 为先验融合权重，初始 0 → 训练起点与原模型完全等价，信任程度由梯度自学
        self.family_prototypes = nn.Parameter(torch.randn(num_families, hidden_dim) * 0.02)
        self.prior_weight = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def predict_family(self, kinase_emb: torch.Tensor) -> torch.Tensor:
        """
        参数: kinase_emb [num_kinase, hidden_dim] (从 forward 返回的 'kinase')
        返回: logits [num_kinase, num_families]
        """
        return self.family_classifier(kinase_emb)

    def interaction_logits(self, kinase_emb: torch.Tensor, site_emb: torch.Tensor,
                           k_idx, s_idx, fam_probs=None) -> torch.Tensor:
        """
        返回激酶-位点对的原始打分 [B]（供 BCEWithLogitsLoss 使用）。
        总分 = 交互 MLP 主分支 + prior_weight · 家族原型先验（A1）：
        pref = 家族概率 @ 原型矩阵，再与位点嵌入做缩放点积。
        fam_probs 可传入训练时已算好的 softmax 家族概率 [K, C]（避免重复计算、
        保证正负样本用同一组概率）；缺省时在内部计算（评估路径）。
        """
        k_emb = kinase_emb[k_idx]   # [B, H]
        s_emb = site_emb[s_idx]     # [B, H]
        feats = torch.cat([k_emb, s_emb, k_emb * s_emb, (k_emb - s_emb).abs()], dim=-1)
        main_logits = self.interaction_mlp(feats).squeeze(-1)   # [B]

        # A1 先验分支：家族概率软混合原型偏好，与位点嵌入做缩放点积（/√d 控制量级）
        if fam_probs is None:
            fam_probs = torch.softmax(self.predict_family(kinase_emb), dim=-1)  # [K, C]
        pref = fam_probs @ self.family_prototypes                                   # [K, H]
        prior_logits = (pref[k_idx] * s_emb).sum(dim=-1) / (s_emb.size(-1) ** 0.5)  # [B]
        return main_logits + self.prior_weight * prior_logits

    def predict_interaction(self, kinase_emb: torch.Tensor, site_emb: torch.Tensor,
                            k_idx, s_idx) -> torch.Tensor:
        """
        参数: kinase_emb [num_kinase, hidden_dim], site_emb [num_site, hidden_dim]
        返回: 概率 [B] (每个激酶-位点对的磷酸化概率)
        """
        return torch.sigmoid(self.interaction_logits(kinase_emb, site_emb, k_idx, s_idx))

    def forward(self, data: HeteroData) -> dict:
        """
        参数:
            data: HeteroData 对象，包含：
                - data['kinase'].esm_x, data['kinase'].n2v_x
                - data['site'].x（可选 data['site'].window，需与 __init__ 的 motif_len>0 匹配）
                - data['kinase', 'ppis', 'kinase'].edge_index
                - data['kinase', 'phosphorylates', 'site'].edge_index
                - data['site', 'rev_phosphorylates', 'kinase'].edge_index (可选，但 HGT 需要)
        返回:
            dict: {'kinase': kinase_embeds, 'site': site_embeds}
        """
        # 1. 准备输入特征
        kinase_esm = data['kinase'].esm_x
        kinase_n2v = data['kinase'].n2v_x
        kinase_feat = torch.cat([kinase_esm, kinase_n2v], dim=-1)
        kinase_x = self.kinase_proj(kinase_feat)

        site_x = self.site_proj(data['site'].x)
        if self.motif_len > 0:
            window = getattr(data['site'], 'window', None)
            if window is None:
                raise ValueError(
                    "模型启用了 motif 编码(motif_len>0)，但图中缺少 data['site'].window；"
                    "请先运行 data_prep/add_motif_features.py 生成带 window 的图"
                )
            site_x = site_x + self.motif_encoder(window.to(site_x.device))

        # 2. HGT 消息传递
        x_dict = {'kinase': kinase_x, 'site': site_x}
        # HGT 后用 GELU（平滑、小负值不归零，缓解 3 层无残差结构下的 dead-ReLU）
        for hgt_layer in self.hgt_layers:
            x_dict = hgt_layer(x_dict, data.edge_index_dict)
            x_dict = {k: F.gelu(v) for k, v in x_dict.items()}
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        # 3. 融合所有节点表示（kinase + site）进行 Transformer 全局建模
        kinase_embeds = x_dict['kinase']     # [num_kinase, hidden_dim]
        site_embeds = x_dict['site']         # [num_site, hidden_dim]
        all_embeds = torch.cat([kinase_embeds, site_embeds], dim=0)  # [N_total, hidden_dim]

        # Transformer 输入需要添加 seq_len 维度 batch_first=True 时形状 (batch, seq, hidden)
        all_embeds = all_embeds.unsqueeze(0)   # [1, N_total, hidden_dim]
        all_embeds = self.transformer_encoder(all_embeds)
        all_embeds = all_embeds.squeeze(0)     # [N_total, hidden_dim]

        # 切回激酶和底物
        num_kinase = kinase_embeds.size(0)
        updated_kinase = all_embeds[:num_kinase]
        updated_site = all_embeds[num_kinase:]

        # 可选：层归一化
        updated_kinase = self.norm(updated_kinase)
        updated_site = self.norm(updated_site)

        return {'kinase': updated_kinase, 'site': updated_site}

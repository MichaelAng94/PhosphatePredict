# model_hgt_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData

class HGTTransformer(nn.Module):
    """
    联合 HGT + Transformer 模型：
    - 输入：HeteroData 对象，包含：
        * kinase: esm_x, n2v_x 两种特征
        * site: x 特征
        * 边：('kinase','ppis','kinase'), ('kinase','phosphorylates','site'), ('site','rev_phosphorylates','kinase')
    - 输出：激酶节点和底物节点的最终嵌入（可用于链接预测或节点分类）
    """
    def __init__(
        self,
        esm_dim: int,
        n2v_dim: int,
        site_dim: int,
        num_families: int,               # 新增：激酶家族类别数
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_hgt_layers: int = 2,
        num_transformer_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # ---- 输入投影：将不同特征映射到 hidden_dim ----
        self.kinase_proj = Linear(esm_dim + n2v_dim, hidden_dim)
        self.site_proj = Linear(site_dim, hidden_dim)
        
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
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        # 新增：激酶家族分类器（MLP）
        self.family_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_families)
        )

        # 新增：关系预测的可学习对角权重（等价于带权重点积）
        self.interaction_weight = nn.Parameter(torch.ones(hidden_dim) * 0.1)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def predict_family(self, kinase_emb: torch.Tensor) -> torch.Tensor:
        """
        参数: kinase_emb [num_kinase, hidden_dim] (从 forward 返回的 'kinase')
        返回: logits [num_kinase, num_families]
        """
        return self.family_classifier(kinase_emb)

    def predict_interaction(self, kinase_emb: torch.Tensor, site_emb: torch.Tensor, k_idx, s_idx) -> torch.Tensor:
        """
        参数: kinase_emb [num_kinase, hidden_dim], site_emb [num_site, hidden_dim]
        返回: 概率矩阵 [num_kinase, num_site] (每个激酶-位点对的磷酸化概率)
        """
        k_emb = kinase_emb[k_idx]   # [B, H]
        s_emb = site_emb[s_idx]     # [B, H]
        w = self.interaction_weight # [H]
        scores = (w * k_emb * s_emb).sum(dim=-1)  # [B]
        return torch.sigmoid(scores)
        # 计算带权重点积: sum(w * h_k * h_s)
        # weighted = w * kinase_emb.unsqueeze(1) * site_emb.unsqueeze(0)  # [K, S, D]
        # scores = weighted.sum(dim=-1)                  # [K, S]
    
    def forward(self, data: HeteroData) -> dict:
        """
        参数:
            data: HeteroData 对象，包含：
                - data['kinase'].esm_x, data['kinase'].n2v_x
                - data['site'].x
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
        
        # 2. HGT 消息传递
        x_dict = {'kinase': kinase_x, 'site': site_x}
        for hgt_layer in self.hgt_layers:
            x_dict = hgt_layer(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        
        # 3. 融合所有节点表示（kinase + site）进行 Transformer 全局建模
        #   注意：Transformer 要求输入形状 (seq_len, batch, hidden_dim)，这里 batch=1
        #   我们将两种节点拼接成一个序列
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
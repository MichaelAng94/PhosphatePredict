#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双任务联合训练：激酶家族分类 + 激酶-底物关系预测
共享 GNN 编码器 (HGT + Transformer)
数据集划分：训练集 70%，验证集 20%，测试集 10%
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, auc, roc_curve, f1_score, precision_score, recall_score, average_precision_score
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入模型
from model.hgt_transformer import HGTTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 11

# ==================== 配置参数 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = "origin_data/joint_hetero_graph.pt"
KINASE_FAMILY_FILE = "origin_data/kinase_family.xls"   # 列名: Name, Family

LINK_PRED_BATCH_SIZE = 2000      # 每次采样多少正负样本对
NUM_NEG_SAMPLES = 1               # 每个正样本对应的负样本数
# LAMBDA_FAMILY = 1.0               # 分类损失权重
# LAMBDA_LINK = 1.0                 # 链接预测损失权重
LEARNING_RATE = 1e-4
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 10

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== 加载数据 ====================
def load_kinase_family_labels(kinase_nodes, family_file):
    """
    加载激酶家族标签。
    参数:
        kinase_nodes: list, 所有激酶节点名称（按字母排序，与图中顺序一致）
        family_file: 文件路径（包含 Name, Family 两列）
    返回:
        labels: torch.LongTensor, 长度为 len(kinase_nodes)，没有标签的为 -1（忽略）
        num_classes: int
    """
    df = pd.read_excel(family_file)
    # 归一化激酶名称（与构建图时一致：大写，不删数字）
    df['Kinase_norm'] = df['Name'].astype(str).str.strip().str.upper()
    # 构建映射
    label_map = {}
    for _, row in df.iterrows():
        label_map[row['Kinase_norm']] = row['Family']
    # 编码家族名称
    families = sorted(set(label_map.values()))
    le = LabelEncoder()
    le.fit(families)
    labels = []
    for k in kinase_nodes:
        if k in label_map:
            labels.append(le.transform([label_map[k]])[0])
        else:
            labels.append(-1)  # 缺失标签
    labels = torch.tensor(labels, dtype=torch.long)
    num_classes = len(families)
    logging.info(f"激酶家族类别数: {num_classes}, 有标签的激酶数: {(labels != -1).sum().item()}")
    return labels, num_classes

def sample_link_prediction_pairs(edge_index_ks, num_kinase, num_site, num_neg=1, batch_size=10000):
    """
    随机采样正负样本对，用于链接预测训练。
    返回的索引均在指定设备上。
    """
    device = edge_index_ks.device
    pos_kinase = edge_index_ks[0].cpu().numpy()
    pos_site = edge_index_ks[1].cpu().numpy()
    total_pos = len(pos_kinase)
    
    if batch_size > total_pos:
        indices = np.arange(total_pos)
    else:
        indices = np.random.choice(total_pos, batch_size, replace=False)
    pos_k = torch.tensor(pos_kinase[indices], dtype=torch.long, device=device)
    pos_s = torch.tensor(pos_site[indices], dtype=torch.long, device=device)
    
    # 构建已存在的边集合（仅用于当前采样 batch 的负样本生成）
    pos_set = set(zip(pos_kinase[indices], pos_site[indices]))
    neg_k_list = []
    neg_s_list = []
    for _ in range(num_neg):
        k_cand = np.random.randint(0, num_kinase, size=len(indices))
        s_cand = np.random.randint(0, num_site, size=len(indices))
        for i in range(len(indices)):
            while (k_cand[i], s_cand[i]) in pos_set:
                k_cand[i] = np.random.randint(0, num_kinase)
                s_cand[i] = np.random.randint(0, num_site)
        neg_k_list.append(torch.tensor(k_cand, dtype=torch.long, device=device))
        neg_s_list.append(torch.tensor(s_cand, dtype=torch.long, device=device))
    neg_k = torch.cat(neg_k_list, dim=0)
    neg_s = torch.cat(neg_s_list, dim=0)
    return pos_k, pos_s, neg_k, neg_s

# ==================== 训练和评估函数（支持掩码）====================
def train_epoch(model, data, kinase_labels, train_kinase_mask, train_edge_index,
                optimizer, criterion_class, criterion_link):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播得到嵌入
    emb_dict = model(data)   # {'kinase': [K, H], 'site': [S, H]}
    # logging.info("计算嵌入完成")
    kinase_emb = emb_dict['kinase']
    site_emb = emb_dict['site']
    
    # 1. 激酶家族分类损失（仅对训练集中的有标签激酶）
    # family_logits = model.predict_family(kinase_emb)   # [K, num_families]
    # mask = (kinase_labels != -1) & train_kinase_mask
    # if mask.sum() > 0:
    #     loss_family = criterion_class(family_logits[mask], kinase_labels[mask])
    # else:
    #     loss_family = torch.tensor(0.0, device=DEVICE)
    
    # 2. 链接预测损失（使用训练边集）
    num_kinase = kinase_emb.size(0)
    num_site = site_emb.size(0)
    pos_k, pos_s, neg_k, neg_s = sample_link_prediction_pairs(
        train_edge_index, num_kinase, num_site,
        num_neg=NUM_NEG_SAMPLES, batch_size=LINK_PRED_BATCH_SIZE
    )
    pos_pred = model.predict_interaction(kinase_emb, site_emb, pos_k, pos_s)  # [B]
    neg_pred = model.predict_interaction(kinase_emb, site_emb, neg_k, neg_s)  # [B * num_neg]
    # logging.info(f"链接预测样本数: 正样本 {pos_pred.size(0)}, 负样本 {neg_pred.size(0)}")
    # pos_pred = pos_score[pos_k, pos_s]
    # neg_pred = pos_score[neg_k, neg_s]
    pos_label = torch.ones_like(pos_pred)
    neg_label = torch.zeros_like(neg_pred)
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    label = torch.cat([pos_label, neg_label], dim=0)
    loss_link = criterion_link(pred, label)
    
    # loss = lambda_family * loss_family + lambda_link * loss_link
    loss_link.backward()
    optimizer.step()
    return loss_link.item()

def evaluate(model, data, kinase_labels, mask_dict, edge_index_dict, num_neg_eval=100):
    """
    评估分类和链接预测性能。
    mask_dict: {'train': train_kinase_mask, 'val': val_kinase_mask, 'test': test_kinase_mask}
    edge_index_dict: {'train': train_edge_index, 'val': val_edge_index, 'test': test_edge_index}
    """
    model.eval()
    with torch.no_grad():
        emb_dict = model(data)
        kinase_emb = emb_dict['kinase']
        site_emb = emb_dict['site']
        
        # 分类性能（对每个子集评估）
        family_logits = model.predict_family(kinase_emb)
        results = {}
        for split, mask in mask_dict.items():
            valid_mask = (kinase_labels != -1) & mask
            if valid_mask.sum() > 0:
                pred_class = family_logits[valid_mask].argmax(dim=1)
                acc = (pred_class == kinase_labels[valid_mask]).float().mean().item()
            else:
                acc = 0.0
            results[f'{split}_acc'] = acc
        
        # 链接预测AUC（对每个边集采样评估）
        num_kinase = kinase_emb.size(0)
        num_site = site_emb.size(0)
        for split, eidx in edge_index_dict.items():
            if eidx.size(1) == 0:
                results[f'{split}_auc'] = 0.0
                results[f'{split}_f1'] = 0.0
                results[f'{split}_acc_link'] = 0.0
                results[f'{split}_precision'] = 0.0
                results[f'{split}_recall'] = 0.0
                continue
            pos_k, pos_s, neg_k, neg_s = sample_link_prediction_pairs(
                eidx, num_kinase, num_site,
                num_neg=num_neg_eval, batch_size=min(1000, eidx.size(1))
            )
            pos_pred = model.predict_interaction(kinase_emb, site_emb, pos_k, pos_s)
            neg_pred = model.predict_interaction(kinase_emb, site_emb, neg_k, neg_s)
            pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
            label = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
            auc = roc_auc_score(label, pred)
            pr_auc = average_precision_score(label, pred)
            pred_labels = (pred >= 0.5).astype(int)
            
            results[f'{split}_auc'] = auc
            results[f'{split}_pr_auc'] = pr_auc
            results[f'{split}_f1'] = f1_score(label, pred_labels, zero_division=0)
            results[f'{split}_acc_link'] = accuracy_score(label, pred_labels)
            results[f'{split}_precision'] = precision_score(label, pred_labels, zero_division=0)
            results[f'{split}_recall'] = recall_score(label, pred_labels, zero_division=0)
    return results

def get_link_pred_predictions(model, data, edge_index, num_neg=10):
    """
    在给定的边集上进行链接预测，返回所有正负样本的预测概率和真实标签。
    参数:
        model: 训练好的模型
        data: 异构图数据
        edge_index: 待评估的边索引 (2, E)
        num_neg: 每个正样本对应的负样本数 (建议设为 5~10 以获得稳定ROC)
    返回:
        preds: numpy array, 预测概率 (长度 E + E*num_neg)
        labels: numpy array, 真实标签 (1为正，0为负)
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        emb_dict = model(data)
        kinase_emb = emb_dict['kinase']
        site_emb = emb_dict['site']
        num_kinase, num_site = kinase_emb.size(0), site_emb.size(0)

        pos_k = edge_index[0].cpu().numpy()
        pos_s = edge_index[1].cpu().numpy()
        total_pos = len(pos_k)

        # 正样本预测
        pos_k_t = torch.tensor(pos_k, device=device, dtype=torch.long)
        pos_s_t = torch.tensor(pos_s, device=device, dtype=torch.long)
        pos_pred = model.predict_interaction(kinase_emb, site_emb, pos_k_t, pos_s_t).cpu().numpy()

        # 负样本采样（随机生成不存在的边）
        pos_set = set(zip(pos_k, pos_s))
        neg_k_list, neg_s_list = [], []
        for _ in range(num_neg):
            k_cand = np.random.randint(0, num_kinase, size=total_pos)
            s_cand = np.random.randint(0, num_site, size=total_pos)
            # 去重（确保负样本不在正样本集中）
            for i in range(total_pos):
                while (k_cand[i], s_cand[i]) in pos_set:
                    k_cand[i] = np.random.randint(0, num_kinase)
                    s_cand[i] = np.random.randint(0, num_site)
            neg_k_list.append(k_cand)
            neg_s_list.append(s_cand)
        neg_k = np.concatenate(neg_k_list)
        neg_s = np.concatenate(neg_s_list)

        neg_k_t = torch.tensor(neg_k, device=device, dtype=torch.long)
        neg_s_t = torch.tensor(neg_s, device=device, dtype=torch.long)
        neg_pred = model.predict_interaction(kinase_emb, site_emb, neg_k_t, neg_s_t).cpu().numpy()

        preds = np.concatenate([pos_pred, neg_pred])
        labels = np.concatenate([np.ones_like(pos_pred), np.zeros_like(neg_pred)])
    return preds, labels
    
# ==================== 主程序 ====================
def main():
    # 1. 加载图数据
    logging.info("加载异构图...")
    data = torch.load(GRAPH_PATH, map_location=DEVICE, weights_only=False)
    
    kinase_names = data['kinase'].node_names
    num_kinase = len(kinase_names)
    num_site = data['site'].x.size(0)
    logging.info(f"激酶节点数: {num_kinase}, 底物节点数: {num_site}")
    
    # 2. 加载激酶家族标签
    kinase_labels, num_families = load_kinase_family_labels(kinase_names, KINASE_FAMILY_FILE)
    kinase_labels = kinase_labels.to(DEVICE)
    
    # 3. 获取模型输入维度
    esm_dim = data['kinase'].esm_x.size(1)
    n2v_dim = data['kinase'].n2v_x.size(1)
    site_dim = data['site'].x.size(1)
    
    # 4. 划分数据集（7:2:1）
    # ---- 激酶节点划分（分类任务）----
    # 只对有标签的节点进行随机划分（不采用分层，避免稀有类别报错）
    labeled_indices = torch.where(kinase_labels != -1)[0].cpu().numpy()
    train_idx, temp_idx = train_test_split(
        labeled_indices, test_size=0.3, random_state=RANDOM_SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1/3, random_state=RANDOM_SEED
    )
    # 构建掩码（全节点长度）
    train_kinase_mask = torch.zeros(num_kinase, dtype=torch.bool)
    val_kinase_mask = torch.zeros(num_kinase, dtype=torch.bool)
    test_kinase_mask = torch.zeros(num_kinase, dtype=torch.bool)
    train_kinase_mask[train_idx] = True
    val_kinase_mask[val_idx] = True
    test_kinase_mask[test_idx] = True
    logging.info(f"激酶分类 - 训练: {train_idx.size}, 验证: {val_idx.size}, 测试: {test_idx.size}")
    
    # ---- 边划分（链接预测任务）----
    full_edge_index = data['kinase', 'phosphorylates', 'site'].edge_index
    num_edges = full_edge_index.size(1)
    edge_indices = np.arange(num_edges)
    train_edge_idx, temp_edge_idx = train_test_split(
        edge_indices, test_size=0.3, random_state=RANDOM_SEED
    )
    val_edge_idx, test_edge_idx = train_test_split(
        temp_edge_idx, test_size=1/3, random_state=RANDOM_SEED
    )
    train_edge_index = full_edge_index[:, train_edge_idx]
    val_edge_index = full_edge_index[:, val_edge_idx]
    test_edge_index = full_edge_index[:, test_edge_idx]
    logging.info(f"链接预测 - 训练边: {train_edge_index.size(1)}, 验证边: {val_edge_index.size(1)}, 测试边: {test_edge_index.size(1)}")
    
    train_data = data.clone()
    train_data['kinase', 'phosphorylates', 'site'].edge_index = train_edge_index
    train_data['site', 'rev_phosphorylates', 'kinase'].edge_index = train_edge_index[[1,0]]

    # 将划分掩码和边索引移到设备
    train_kinase_mask = train_kinase_mask.to(DEVICE)
    val_kinase_mask = val_kinase_mask.to(DEVICE)
    test_kinase_mask = test_kinase_mask.to(DEVICE)
    train_edge_index = train_edge_index.to(DEVICE)
    val_edge_index = val_edge_index.to(DEVICE)
    test_edge_index = test_edge_index.to(DEVICE)
    
    # 5. 模型初始化
    model = HGTTransformer(
        esm_dim=esm_dim,
        n2v_dim=n2v_dim,
        site_dim=site_dim,
        num_families=num_families,
        hidden_dim=256,
        num_heads=4,
        num_hgt_layers=2,
        num_transformer_layers=1,
        dropout=0.2
    ).to(DEVICE)
    
    # 6. 损失函数与优化器
    criterion_class = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_link = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # 7. 训练循环
    best_val_auc = 0.0
    patience_counter = 0
    
    # 准备掩码字典和边字典供评估使用
    mask_dict = {'train': train_kinase_mask, 'val': val_kinase_mask, 'test': test_kinase_mask}
    edge_dict = {'train': train_edge_index, 'val': val_edge_index, 'test': test_edge_index}
    
    # logging.info("start training...")
    for epoch in range(1, EPOCHS+1):
        loss = train_epoch(
            model, train_data, kinase_labels, train_kinase_mask, train_edge_index,
            optimizer, criterion_class, criterion_link
        )
        
        # 每 5 个 epoch 评估一次（使用验证集）
        if epoch % 5 == 0:
            results = evaluate(model, data, kinase_labels, mask_dict, edge_dict, num_neg_eval=1)
            val_auc = results['val_auc']
            logging.info(
                f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                f"Train Acc: {results['train_acc']:.4f}, Val Acc: {results['val_acc']:.4f} | "
                f"Train AUC: {results['train_auc']:.4f}, Val AUC: {val_auc:.4f}, Val PR-AUC: {results['val_pr_auc']:.4f} | "
                f"Val F1: {results['val_f1']:.4f}, Val Prec: {results['val_precision']:.4f}, Val Rec: {results['val_recall']:.4f}"
            )
            
            # 早停（基于验证集 AUC）
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), "model/best_joint_model.pt")
                logging.info(f"  -> 保存最佳模型 (Val AUC = {val_auc:.4f}, Val F1 = {results['val_f1']:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
        else:
            if epoch % 1 == 0:
                logging.info(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
    
    # 加载最佳模型并在测试集上评估
    logging.info(f"训练完成，最佳验证集 AUC: {best_val_auc:.4f}")
    model.load_state_dict(torch.load("model/best_joint_model.pt", map_location=DEVICE))
    final_results = evaluate(model, data, kinase_labels, mask_dict, edge_dict, num_neg_eval=1)
    logging.info(
        f"最终测试集性能: "
        f"Link AUC = {final_results['test_auc']:.4f}, "
        f"PR-AUC = {final_results['test_pr_auc']:.4f}, "
        f"Link F1 = {final_results['test_f1']:.4f}, "
        f"Link Acc = {final_results['test_acc_link']:.4f}, "
        f"Precision = {final_results['test_precision']:.4f}, "
        f"Recall = {final_results['test_recall']:.4f}"
    )

    # 获取训练、验证、测试集的预测结果
    train_preds, train_labels = get_link_pred_predictions(model, data, train_edge_index, num_neg=10)
    val_preds, val_labels = get_link_pred_predictions(model, data, val_edge_index, num_neg=10)
    test_preds, test_labels = get_link_pred_predictions(model, data, test_edge_index, num_neg=10)

    # 计算 ROC 曲线和 AUC
    fpr_train, tpr_train, _ = roc_curve(train_labels, train_preds)
    auc_train = auc(fpr_train, tpr_train)

    fpr_val, tpr_val, _ = roc_curve(val_labels, val_preds)
    auc_val = auc(fpr_val, tpr_val)

    fpr_test, tpr_test, _ = roc_curve(test_labels, test_preds)
    auc_test = auc(fpr_test, tpr_test)

    plt.figure(figsize=(5.5, 5))   # 适合双栏排版

    # Nature 推荐配色
    colors = {'train': '#F4898B', 'val': '#87BFD6', 'test': '#61DCB8'}  # 红、蓝、深蓝

    # 绘制三条曲线
    plt.plot(fpr_train, tpr_train, color=colors['train'], lw=1.2,
            label=f'Train (AUC = {auc_train:.3f})')
    plt.plot(fpr_val, tpr_val, color=colors['val'], lw=1.2,
            label=f'Val (AUC = {auc_val:.3f})')
    plt.plot(fpr_test, tpr_test, color=colors['test'], lw=1.2,
            label=f'Test (AUC = {auc_test:.3f})')

    # 对角线（随机分类器）
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Random')

    # 坐标轴设置
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # 图例（无边框）
    plt.legend(loc='lower right', frameon=False)

    # 去掉上、右脊线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # 保存高分辨率矢量图
    plt.tight_layout()
    plt.savefig('figures/roc_curves.png', dpi=500, bbox_inches='tight')  # 可选
    plt.show()

if __name__ == "__main__":
    main()
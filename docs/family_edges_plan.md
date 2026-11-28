# 回收被剔除的 KSR 注释：工作记录

- 日期：2026-11-28
- 状态：代码与数据表的改动已完成。**除别名表生成脚本外，本轮代码都没有运行过**，下一步在 WSL 上重建图、跑测试、训练。
- 起因：建图日志 `剔除 5229 行 KSR（127 个名字无法归属到具体激酶）`。按唯一边算是 2,479 条，占 KSR 全部唯一边的 11.3%。

## 一、方案

| 项 | 做法 |
|---|---|
| D1 泛称（`PKA_GROUP`、`CAMKII`…） | 每个家族建一个节点 `FAMILY_<家族>`（36 个），特征取成员特征的均值，泛称的边挂在这个节点上，**只进训练图**。不把边复制给成员 |
| D2 评估 | 家族层面注释过的位点，对家族成员既不算正例也不算负例：负采样、批内对比损失都避开它，全候选排序把它移出成员的候选（成员自己的特异正例不移出） |
| D3 自磷酸化（`AUTOCATALYSIS`） | 按底物蛋白的 UniProt 反推出激酶。反推出的边只进训练图；评估激酶的自磷酸化边两边都不进 |
| D4 非人物种名（`CDC2_DROME`） | 去掉后缀，按人源同源处理（51 条） |
| D5 不进图 | 非蛋白激酶、融合蛋白 / 注释文字、反推不出激酶的自磷酸化、没有成员定义的泛称，共 246 条（1.1%） |

生成脚本试跑的回收效果：进图的边 19,452 → 21,455（88.7% → 97.8%）。新增家族层面边 1,900 条、自磷酸化边 87 条。

## 二、今日改动

| 文件 | 改动 |
|---|---|
| `origin_data/kinase_alias_map.csv`、`kinase_family.csv`、`kinase_autophos_map.csv` | 由 `data_prep/build_kinase_alias_map.py` 正式生成（此前只写在 `/tmp`） |
| `data_prep/edge_levels.py`（共享模块） | 新增 `train_level`（训练边与层级标注同序）、`apply_train_edges`（替换边时同步替换 `edge_level`）、`masked_sites`（旧图不再越界）。负样本保护补成双向：成员的特异位点不再被抽成家族节点的负样本。补全逻辑只读补全前的位点集合，一个激酶属于两个家族时结果不再随处理顺序变化 |
| `train/train_hgt_no_transformer.py`、`train_hgt_trans.py`、`train_hgt_twin_tower.py` | 训练图快照改用 `apply_train_edges`；检索谱的来源改为"有特定激酶训练边的激酶"；全候选评估用 `masked_sites` |
| `train/eval_checkpoint.py` | 同上。另修正两处会让脚本无法执行的错误：协议打印引用了不存在的 `val_k` / `test_k`（上一轮改动引入）；`main()` 按 7 个值解包 5 个值、并给模型传了不存在的 `esmc_dim` 参数（已提交版本里就有） |
| `train/full_rank_eval.py` | 全候选评估用 `masked_sites` |
| `train/check_dup_kinases.py` | 划分改用共享模块；家族节点单列，不再被报成"跨划分的重复身份" |
| `graph/local_build_check.py` | T1 期望值按新别名表更新，增加家族节点与成员表一致性的检查。T3a 把 KSR 的行归属也切回旧行为，老图数字（567 / 453 / 11123 / 16216，划分 11601 / 2935 / 1680）改为硬性断言。划分预览只按特定激酶的注释。新增 T5：家族节点的成员与特征、边层级标注、按层级去重、与 KSR 逐行归属一致 |
| `data_prep/test_edge_levels.py`（新增） | 需要 torch。合成图上逐条断言各项约定；真实旧图上新划分与原划分代码逐边一致；真实新图上检查训练图的组成与负样本过滤 |

所有改动都没有暂存、没有提交。已有文件保持原有换行符。

## 三、家族节点进入训练流程后的副作用排查

逐项看了训练脚本里依赖激酶集合的位置。结论来自读代码，没有实测。

| 检查点 | 结论 | 处理 |
|---|---|---|
| 相似边（similar_to） | 家族节点会与成员相连（它是成员特征的均值，对成员是近亲）。它的表示只由训练边计算，冷启动激酶从它继承的是训练信息，没有泄漏 | 不改，这是设计里的收益路径 |
| 检索谱 | 原实现把家族节点当检索源。家族节点的底物全是家族层面位点，这些位点对成员既不计正也不计负（D2），作检索源没有收益；紧凑家族（PKA、CK2）的节点还会在 softmax(τ=0.05) 下分走近亲激酶的权重 | **改为不作检索源。** 对特定激酶，检索谱与改动前逐值相同 |
| 均衡采样权重 | 权重 ∝ 1/√deg，家族节点是枢纽、单条边的权重被压低。估算 36 个家族节点的采样份额约 14%，与它们占训练边的比例相当 | 不改 |
| 负采样 | 成员一侧已保护；家族节点一侧原来缺保护 | 已补成双向 |
| 分类损失 | 家族节点没有家族标签，不参与家族 / 大组分类损失，分类划分的掩码里也没有它们 | 无需处理 |
| 批内对比损失 | `(成员, 家族层面位点)`、`(家族节点, 成员位点)` 都在非负样本键集合里 | 无需处理 |

遗留一处未处理：家族节点的家族分类输出没有训练过，打分时它的家族原型先验一项是噪声。这只影响家族层面边自身的训练损失，要消除需要改模型。

## 四、待办（WSL）

```bash
# 1. 重建之前：留住旧图，并在旧图上验证划分逐边一致
cp origin_data/joint_hetero_graph.pt origin_data/joint_hetero_graph_prev.pt
python3 data_prep/test_edge_levels.py origin_data/joint_hetero_graph_prev.pt

# 2. 重建图、加 motif
python3 graph/build_joint_graph.py
python3 data_prep/add_motif_features.py \
    --graph origin_data/joint_hetero_graph.pt \
    --fasta origin_data/phosphosite_windows.fasta \
    --out origin_data/joint_hetero_graph_motif.pt

# 3. 重建之后：新图的不变量
python3 data_prep/test_edge_levels.py

# 4. 训练
python3 train/train_hgt_no_transformer.py
```

`graph/local_build_check.py` 只需要 numpy / pandas / scikit-learn，用来在建图之前核对名字与图结构。

## 五、未验证与已知限制

- **本轮代码没有运行过**，只做了逐文件通读。两个检查脚本自身也没运行过，第一次跑出错的可能性不低。
- **效果无法承诺。** 多出来的 1,900 条边集中在本来数据就多的家族（PKA 408、PKC 293、CK2 276），对稀疏激酶的帮助可能有限。
- **指标与之前的运行不可比。** 别名合并已经让划分变了；家族成员的候选位点因屏蔽而变少；训练图多了约 2,000 条边。
- **D4 的同源关系是凭知识写的，没有逐一查库核对**：`TOR`→MTOR、`CAKI`→CDK7、`MLCK`→MYLK、`LIT-1`→NLK、`SGG`→GSK3 家族节点、`HOP`→JAK、`SRC64B`→SRC、`UNC-43`→CAMK2、`CKI_ALPHA`→CK1。
- **家族成员的范围**：`TK`（90 个成员）、`CDK`（20 个）、`MAPK`（10 个）、`PKC`（9 个）这几个组很粗，家族节点的特征是整个组的均值。
- 旧 checkpoint 与新图不兼容。

已关闭的一项：自磷酸化反推出的 133 条不进图的边，今日查明原因，不是漏。底物不是蛋白激酶（如 `P42338` 是 PIK3CB），或是小鼠 / 大鼠的同源蛋白（PhosphoSitePlus 对啮齿类位点单独建号，HGNC 不登记），或 accession 不在 HGNC。按 D5 本来就不进图。

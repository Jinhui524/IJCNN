# Hawkes-DyG：服务器运行说明

代码在 `hawkes_dyg/`，三个数据集入口在 `scripts/`，旧 `IJCNN.py` 保留。只训练下一活动预测，参数离线学习。无需 DGL、LLM、文本嵌入或 DyG-RAG 服务。

## 1. 安装与数据位置

建议 Python 3.10 或以上，已在 Python 3.12 的 CPU 环境验证。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_hawkes_dyg.txt
python -m unittest discover -s tests -v
```

GPU 服务器需要安装适配其驱动/CUDA 的 PyTorch wheel；执行以下命令确认环境：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

本地已验证 CPU 执行；CUDA 执行需要在服务器验证。显式指定 `--device cuda:0` 而 CUDA 不可用时会报错，不静默切换到 CPU。

默认目录：

```text
项目根目录/
  hawkes_dyg/
  scripts/
  tests/
  dataset/
    BPIC_2012_W.csv
    BPI_Challenge_2013_C.csv
    Loan_Application_events.csv
    BPIC_2020_{D,I,Pe,Pr,R}.csv
    Receipt.csv
    Helpdesk_normalized.csv
    dataset_conversion_report.json
  requirements_hawkes_dyg.txt
```

代码包不包含原始数据。解压到服务器后，将三个 CSV 放入 `dataset/`，或者传入 `--data-dir /服务器上的数据目录`。脚本默认路径相对于项目根目录解析，不要求固定的工作目录。

## 2. 先做小规模检查

```bash
python scripts/run_bpic_2012_w.py --folds 3 --epochs 1 --max-cases 100 --max-prefixes 250 --batch-size 16 --device cpu --output-dir smoke_results
python scripts/run_bpi_2013_c.py --folds 3 --epochs 1 --max-cases 100 --max-prefixes 250 --batch-size 16 --device cpu --output-dir smoke_results
python scripts/run_loan.py --folds 3 --epochs 1 --max-cases 100 --max-prefixes 250 --batch-size 16 --device cpu --output-dir smoke_results
```

`--max-cases` 按固定随机种子抽取案例；`--max-prefixes` 限制**每折每个 split** 的样本数量，截取该 split 内最先生成的前缀，仅用于调试。两个选项不适合论文主实验，也不限制单条轨迹长度。读入 CSV 和预处理所选案例仍需要内存；前缀索引只占 O(案例数) 内存，批次按最长前缀 padding。

## 3. 正式主实验

```bash
python scripts/run_bpic_2012_w.py --device cuda:0
python scripts/run_bpi_2013_c.py --device cuda:0
python scripts/run_loan.py --device cuda:0
python scripts/run_bpic_2020_d.py --device cuda:0
python scripts/run_bpic_2020_i.py --device cuda:0
python scripts/run_bpic_2020_pe.py --device cuda:0
python scripts/run_bpic_2020_pr.py --device cuda:0
python scripts/run_bpic_2020_r.py --device cuda:0
python scripts/run_receipt.py --device cuda:0
python scripts/run_helpdesk.py --device cuda:0
```

或使用公共入口：

```bash
python -m hawkes_dyg.runner --dataset BPIC_2012_W --device cuda:0
```

默认：3 折、100 epochs 上限、patience 10、batch size 64、AdamW 学习率 0.001、embedding 32、hidden 64、2 层因果历史边 GNN、history window 32、dropout 0.2、20 个分位数桶。

支持 `--device`、`--data-dir`、`--output-dir`、`--folds`、`--epochs`、`--batch-size`、`--learning-rate`、`--patience`、`--seed`、`--split-seed`、`--selection-metric`、`--skip-test`、`--time-bins`、`--graph-layers`、`--history-window`、`--hidden-dim`、`--embedding-dim`、`--dropout`、`--weight-decay`、`--intensity-scale-init`、`--initial-alpha`、`--initial-beta`、`--initial-mu`、`--omega-scale-init`、`--omega-scale-max`、`--num-workers`、`--cpu-threads`。

默认 `--num-workers 0`、`--cpu-threads 4`。长前缀 GPU 内存不足时先降低 batch size。输出目标中已有结果时会报错，避免小实验覆盖主实验；重复实验使用新的 `--output-dir`，例如 `results_seed43`。

## 4. 实际实现的数学定义

事件位置 i 的节点输入包括活动 ID、所选结构化属性、`duration`、`start_duration` 的分箱嵌入、连续 `delta`/elapsed 投影，以及固定正弦位置编码；拼接后线性投影到 hidden_dim，接 ReLU 和 Dropout。

- `duration[i] = t_i - t_(i-1)`，首事件为 0。
- `start_duration[i] = t_i - t_1`。
- 两个时间属性以**天**为单位，不使用未来总时长。
- 时间/数值分箱器按每折训练集事件拟合；重复分位数边界合并，所以实际桶数可能小于 20。常量时间使用一个桶。
- 类别属性使用独立词表；0 是 PAD，1 是 UNK，缺失值在训练中出现时有独立类别；常量/全空属性按训练集判定并删除。
- 对相同时间戳保持 CSV 原始顺序，保留零间隔，不合并重复活动的事件实例；时间解析支持混合秒/小数秒格式。
- 固定位置编码使用零起始位置，不依赖前缀最终长度。

图是固定历史窗口内的有向因果图。对目标事件 i，始终保留 `i-1 -> i` 的 `seq` 边，并保留最多 32 个最近历史事件到 i 的 `hist` 边。边只允许从过去流向当前，不添加反向边；多跳关系由 `lag=i-j` 的嵌入表示，不重复添加二跳/三跳边。

```text
s_t = median(训练折内严格正的 duration)；全部为零时取 1 天
s_t 的数值下界为 1e-8 天
α_ab = softplus(θ_alpha[a, b])
β = softplus(θ_beta) + 1e-6
ω_(j,i) = α_(a_j, a_i) * exp(-β * (t_i - t_j) / s_t), j < i
```

所有活动对的初始 α 均为 0.5，共享 β 的 softplus 部分初始为 0.5，baseline μ 初始为 0.1。不使用转移计数初始化。首节点和 padding 位置没有历史边；目标 `a_(k+1)` 不参与当前前缀边权计算。

边表示活动对、类别属性变化、数值属性差值、连续时间差、lag 和 Hawkes-inspired 时间激发分数。直接 `seq` 边使用独立消息通道；`hist` 边在有效历史边上使用显式 masked softmax，避免入度随前缀长度增加而放大消息。

```text
m_i = Σ_j α_(j,i) (V h_j + U r_(j,i))
h_i' = LayerNorm(h_i + Dropout(ReLU(W_self h_i + m_i)))
```

`W_self` 和 residual 是节点自身通道，不把 `j=i` 放入 Hawkes 强度。每层后清零 padding。读出是最后一个真实节点、masked mean 和最后两个节点的差值拼接。候选活动强度为：

```text
lambda_b(t_k | H_k) = mu_b + Σ_(j<=k) alpha_(a_j,b) exp(-beta * (t_k-t_j)/s_t)
logits_b = graph_logits_b + eta * log(lambda_b + epsilon)
```

其中 eta 为有界的正可学习缩放因子，且 `log(lambda)` 在候选活动维度归一化。该强度头通过下一活动交叉熵学习，是 Hawkes-inspired intensity prior，不是完整点过程最大似然估计。

损失只有下一活动交叉熵，不加 L1 稀疏项、TPP 似然、熵或信息增益。其余模型参数使用 AdamW 的 1e-4 weight decay，α/β 参数组的 weight decay 明确为 0，因此不出现对未观测活动对的间接正则更新。可通过 `--weight-decay 0` 关闭其余参数的 weight decay。

**能力边界：**图消息使用最近 32 个历史事件，候选强度头使用完整已观测前缀。这里的“动态”表示前缀增加导致事件图增长，模型参数推理时固定，无持续学习记忆。

## 5. 属性与评估协议

| 数据集 | 案例列 | 节点的额外属性 |
|---|---|---|
| BPIC_2012_W | case | lifecycle:transition |
| BPI_Challenge_2013_C | case | org:group, resource country, organization country, organization involved, impact, lifecycle:transition |
| Loan_Application_events | case:concept:name | Action, org:resource |
| BPIC_2020_D/I/Pe/Pr/R | case | org:resource, org:role |
| Receipt | case | org:resource, org:group |
| Helpdesk | case | org:resource, product, service_level, workgroup |

案例型数值属性（例如 Loan 的 RequestedAmount）当前不作为节点或边特征。`remaining_time`、案例 ID、EventID、原始索引列、concept:instance 等未作为节点特征。BPIC 2020 的 `id`、Receipt 的 `concept:instance` 和常量生命周期字段也未作为节点特征。

案例级 `KFold(shuffle=True, random_state=split_seed)`；每折训练部分再以 `split_seed+fold` 划出 15% 验证案例。模型初始化使用独立的 `seed`。默认只用 validation NLL 早停，最佳模型只测试一次；调参时使用 `--skip-test`。前缀从长度 2 到案例长度减 1，不预测 EOC。短于 3 个事件的案例没有训练/评估样本；极小调试子集可能出现无样本 split，此时需要增加案例数。

活动类别全集按任务预先已知的闭集标签空间处理，包括仅在留出案例出现的活动；这是计划约定的标签 schema，不使用其计数初始化模型。其余词表、数值/时间桶和时间尺度均仅从每折训练数据拟合。metadata 会列出训练折未见活动。这不属于开放类别预测协议。

指标：Accuracy、Macro-Precision、Macro-Recall、Macro-F1、Macro-G-Mean、NLL。宏指标采用与原始 FHGSN/IJCNN.py 相同的标签集合规则：每个 split 使用真实标签和预测标签的并集，不额外加入该 split 中完全未出现的活动类别。另保存按案例平均的 `case_*` 指标。G-Mean 为 `sqrt(macro sensitivity * macro specificity)`，不是多类别召回率的几何平均。NLL 用每样本交叉熵汇总，不能按批次均值等权平均。

`summary.csv` 包括每折、Average 和 Std；Std 是折间总体标准差（ddof=0），不是独立重复运行的标准差。正式与 FHGSN/其他方法比较时，应使用保存的同一案例划分和同一可用属性；旧 IJCNN 使用测试集早停，因此新指标不可直接与旧脚本结果视为同一严格实验协议。

## 6. 结果与重载

```text
results/<dataset>/
  run_config.json       # 参数、依赖版本、命令、调试标记
  experiment.log
  summary.csv           # 每完成一折即更新
  fold_1/ (fold_2/, fold_3/)
    best_model.pt       # 仅模型 state_dict
    model_config.json
    preprocessor.json   # 活动映射、属性词表、分箱边界、时间尺度
    split_cases.json
    metadata.json
    history.csv
    loss_curve.svg
    metrics.json
    test_predictions.csv
    alpha.npy
    influence.json      # α、β、μ 和强度先验缩放范围
```

α 的行/列顺序均由 preprocessor.json 的 `activities` 给出。

重载示例（在项目根目录）：

```python
from hawkes_dyg.runner import load_trained_fold
from hawkes_dyg.data import load_event_log, PrefixDataset, collate_prefix_batch

model, encoder = load_trained_fold('results/BPIC_2012_W/fold_1', device='cpu')
# 重载后对实际已观察前缀调用 encoder.transform_frame(prefix_df)，再 collate。
# prefix_df 要先具备 duration/start_duration；不重新 fit 编码器。
```

脚本保存结果后可直接读取 summary.csv 作为本方法的主实验行。验证集调参入口为：

```bash
python scripts/tune_hawkes_dyg.py --dataset Helpdesk --device cuda:0 \
  --trials 28 --output-dir tuning/helpdesk
```

该入口固定 `split_seed`，只使用 validation NLL 排序，并在每个 trial 中跳过测试集。

## 7. 代码来源核查

IJCNN.py 的主要参考是训练框架、案例前缀、结构化属性、时间特征和分位数分箱。新实现采用事件实例节点和有向链，并修正测试集早停、padding 尾位读出和结果文件覆盖等问题。

DyG-RAG 核查版本：`ca37e449f0bfba188644bd5e66809473578c30bf`，仓库 <https://github.com/RingBDStack/DyG-RAG>。

- `_op.py` 的 `ExponentialDecayTimeWeight` 实现按时间距离指数衰减。
- 同文件的关系构造实际按“实体项与时间项线性加权”排序选择 Top-K，并与无向图存储结合；它与论文中 Jaccard × 时间指数项的公式并不完全一致。
- `_storage/vdb_timestamp.py` 的 Fourier 编码主要针对年/月/日时间锚点，通用解析后仍只使用年月日，不能原样表示流程日志的秒级间隔。
- 本实现独立编写 PyTorch 门控与有向消息传递，仅参考事件单位与时间权重思想，没有复制 DyG-RAG 的 RAG/NER/检索代码，也没有使用它的模型或 API。

不能把本版方法表述成 DyG-RAG 或完整 Hawkes 过程的直接复现。

## 数据集转换

如果从包含 XES 源文件的副本开始，使用标准库流式转换器：

```bash
python scripts/prepare_datasets.py --data-dir dataset --delete-xes
```

该命令将 BPIC 2020 D/I/Pe/Pr/R、Receipt 转为统一 CSV，写出
`Helpdesk_normalized.csv`，并将案例数、事件数、活动集合、时间戳和案例内
事件顺序写入 `dataset_conversion_report.json`。只有全部校验成功后才删除 6 个
XES 源文件；Helpdesk 原始 CSV 始终保留。省略 `--delete-xes` 可以在测试时保留
XES 文件。

# OrderFlow ML Agent Plan

## 项目定位

从零重建 OrderFlow 项目。旧的 A/B/C/D/E/F 场景识别只作为可选特征，不再作为核心预测目标。

新的核心目标是：

```text
用 Binance 全部交易对的 Price / OI / CVD 历史序列，
预测未来 N 根 K 线在扣除手续费后是否有交易价值。
```

最终模型输出不是“市场解释标签”，而是交易标签：

```text
LONG / SHORT / NO_TRADE
```

所有结果必须以扣费后的交易表现为准，而不是只看分类准确率。

## 当前数据范围

数据根目录：

```text
Binance/
```

当前共有 22 个交易对：

```text
1000BONKUSDT
1000PEPEUSDT
1000SHIBUSDT
AAVEUSDT
ADAUSDT
ANKRUSDT
ARBUSDT
ARCUSDT
ATOMUSDT
AVAXUSDT
AXSUSDT
BCHUSDT
BERAUSDT
BNBUSDT
BTCUSDT
CHZUSDT
CRVUSDT
DASHUSDT
DOGEUSDT
DOTUSDT
ENAUSDT
ENJUSDT
```

每个交易对都有：

```text
15m
1h
```

每个周期都至少包含以下关键文件：

```text
data__cg_price_history.parquet
data__cg_open_interest_history.parquet
data__cg_cvd_history.parquet
```

## 运行环境

后续开发、训练、回测统一使用 conda 环境：

```bash
conda activate quant
```

该环境应作为默认运行环境，假设其中已经包含项目需要的主要依赖，包括：

```text
pandas
numpy
scikit-learn
lightgbm
pyarrow
torch
```

如果在默认 shell 中检测不到 `torch` 或其他深度学习依赖，不代表项目不使用 GRU；应优先切换到 `quant` 环境后再运行相关脚本。

## 基本交易成本

默认手续费：

```text
开仓: 0.05%
平仓: 0.05%
往返: 0.10%
滑点: 0
```

所有策略评估必须报告扣费后结果。

## 总体路线

按以下顺序推进：

```text
1. 定义交易标签 LONG / SHORT / NO_TRADE
2. 用 HistGradientBoosting / LightGBM 跑快速表格基线
3. 同时做 GRU 时序模型
4. 在同一测试集上比较扣费后收益
```

LightGBM/HistGradientBoosting 是 sanity check，不是主方向。GRU 是深度学习主线。

## 阶段 1：数据层重建

### 目标

构建统一数据管道，支持所有交易对和周期。

### 输入

每个交易对的 parquet 文件：

```text
price
open_interest
cvd
```

可选扩展数据：

```text
taker_buy_sell_volume
funding_rate
liquidation
long_short_ratio
orderbook
```

### 输出

统一后的 K 线级数据集：

```text
dataset/{interval}/features.parquet
```

每一行至少包含：

```text
timestamp
coin
pair
interval
price
oi
cvd
```

### 关键规则

1. 三个数据源按 `ts` 合并。
2. 合并后必须重采样成真正的目标 K 线。
3. 不能因为 outer merge 产生伪高频样本。
4. 缺失值只允许使用过去值填充，不能使用未来值。
5. 每个交易对单独处理，再拼接成全市场数据集。

## 阶段 2：特征工程

### 基础特征

对每个交易对独立计算：

```text
price_return
oi_return
cvd_diff
cvd_pct_change
```

### 历史窗口特征

窗口建议：

```text
3, 5, 10, 20, 50, 100
```

计算：

```text
rolling_mean
rolling_std
rolling_min
rolling_max
rolling_zscore
momentum
up_ratio
down_ratio
```

### 订单流特征

```text
price_cvd_divergence
price_up_cvd_down
price_down_cvd_up
oi_up_price_up
oi_up_price_down
oi_down_price_up
oi_down_price_down
cvd_acceleration
oi_acceleration
```

### 场景特征

旧 A-F 场景可以作为辅助特征，但不能作为最终目标：

```text
last_scenario
scenario_streak
scenario_ratio_20
scenario_ratio_100
```

### 全市场特征

建议加入跨币种市场状态：

```text
BTC_return
BTC_oi_return
BTC_cvd_change
market_avg_return
market_breadth
sector_like_meme_avg_return
relative_strength_vs_BTC
relative_cvd_vs_BTC
```

注意：全市场特征也必须只使用当前及过去数据。

## 阶段 3：交易标签设计

核心标签不是场景，而是扣费后未来收益。

### 标签定义

对每一行，计算未来持有 N 根 K 线的收益：

```text
future_return_N = price[t + N] / price[t] - 1
```

默认持仓周期：

```text
1, 2, 4, 8
```

对 1h 数据代表：

```text
1h, 2h, 4h, 8h
```

### 手续费阈值

往返手续费：

```text
fee = open_fee + close_fee = 0.10%
```

加入安全边际：

```text
buffer = 0.05% 或 0.10%
```

标签：

```text
if future_return_N > fee + buffer:
    label = LONG
elif future_return_N < -(fee + buffer):
    label = SHORT
else:
    label = NO_TRADE
```

### 多标签实验

必须分别测试：

```text
horizon = 1
horizon = 2
horizon = 4
horizon = 8
```

并比较哪个 horizon 扣费后表现最好。

## 阶段 4：数据切分

必须按时间切分，禁止随机切分。

建议：

```text
训练集: 前 70%
验证集: 中间 15%
测试集: 最后 15%
```

全市场训练时，切分边界按时间戳统一，而不是每个交易对随机切。

必须保留两类测试：

```text
1. 时间外推测试：同一批交易对，测试未来时间段
2. 币种外推测试：留出部分交易对完全不参与训练
```

币种外推测试建议留出：

```text
BTC
ENA
DOGE
1000PEPE
```

具体留出组合后续可调整。

## 阶段 5：表格基线

### 模型

优先级：

```text
HistGradientBoostingClassifier
LightGBM
```

如果 LightGBM 未安装，先用 sklearn 的 HistGradientBoostingClassifier。

### 目标

建立强 sanity check：

```text
如果表格基线完全无效，优先检查标签和特征，不要急着调 GRU。
```

### 输出

必须报告：

```text
classification_report
confusion_matrix
每类 precision/recall
测试集交易次数
扣费后胜率
扣费后平均收益
扣费后累计收益
最大回撤
profit factor
```

## 阶段 6：GRU 模型

### 输入

过去 L 根 K 线的序列：

```text
lookback = 32, 64, 128
```

每根 K 线包含同一套特征。

输入形状：

```text
[batch, lookback, feature_dim]
```

### 输出

三分类：

```text
LONG
SHORT
NO_TRADE
```

### 模型结构初版

```text
GRU(input_dim, hidden_size=64, num_layers=1 or 2, dropout=0.1)
Linear(hidden_size, 3)
Softmax/CrossEntropyLoss
```

### 类别不平衡处理

必须处理 NO_TRADE 过多或 LONG/SHORT 不均衡问题：

```text
class_weight
focal_loss
weighted sampler
阈值调优
```

### 阈值决策

不要简单 argmax。

建议：

```text
只在 max_prob > threshold 时交易
否则 NO_TRADE
```

测试阈值：

```text
0.40, 0.45, 0.50, 0.55, 0.60, 0.65
```

## 阶段 7：统一回测引擎

所有模型必须走同一个回测函数。

输入：

```text
timestamp
coin
price_entry
price_exit
predicted_signal
confidence
fee
```

输出：

```text
gross_pnl
net_pnl
win
equity_curve
drawdown
```

必须支持：

```text
按全市场统计
按交易对统计
按年份/月统计
按 horizon 统计
按 confidence bucket 统计
```

核心评估指标：

```text
扣费后总收益
扣费后平均单笔收益
扣费后胜率
最大回撤
profit factor
交易次数
平均持仓周期
连续亏损次数
```

## 阶段 8：比较标准

LightGBM/HistGradientBoosting 与 GRU 必须在同一数据切分、同一标签、同一手续费下比较。

比较表：

```text
model
interval
horizon
lookback
threshold
trades
net_win_rate
avg_net_pnl
total_net_pnl
max_drawdown
profit_factor
```

只有当 GRU 在测试集扣费后优于表格基线，才继续投入更复杂深度模型。

## 推荐项目结构

```text
OrderFlow/
├─ Binance/
├─ agent.md
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ data_catalog.py
│  ├─ data_loader.py
│  ├─ feature_engineering.py
│  ├─ labeling.py
│  ├─ splits.py
│  ├─ backtest.py
│  ├─ metrics.py
│  ├─ train_hgb.py
│  ├─ train_gru.py
│  └─ config.py
├─ configs/
│  ├─ baseline_1h.yaml
│  └─ gru_1h.yaml
├─ datasets/
├─ models/
└─ results/
```

## 第一轮执行计划

### Step 1

创建数据目录扫描器：

```text
src/data_catalog.py
```

输出所有交易对、周期、关键文件是否完整。

### Step 2

创建统一数据加载器：

```text
src/data_loader.py
```

支持：

```text
load_pair_interval(pair, interval)
resample_to_bar(rule)
build_all_pairs_dataset(interval='1h')
```

### Step 3

创建标签模块：

```text
src/labeling.py
```

先实现：

```text
horizon = 4
fee = 0.001
buffer = 0.0005
```

### Step 4

创建表格基线：

```text
src/train_hgb.py
```

先跑 1h 全交易对。

### Step 5

创建统一回测：

```text
src/backtest.py
```

必须输出扣费后结果。

### Step 6

创建 GRU baseline：

```text
src/train_gru.py
```

先使用：

```text
lookback = 64
horizon = 4
interval = 1h
```

## 关键禁令

1. 禁止随机切分时间序列。
2. 禁止用未来数据填充缺失值。
3. 禁止只汇报准确率。
4. 禁止忽略手续费。
5. 禁止把 A-F 场景当最终交易目标。
6. 禁止只在 BTC 或单个小币种上得出全局结论。
7. 禁止模型调参时看测试集结果反复优化。

## 项目成功标准

第一阶段成功不是“模型很准”，而是建立可信评估框架。

最低可接受目标：

```text
1. 全 22 个交易对可批量生成统一数据集
2. LONG / SHORT / NO_TRADE 标签可配置
3. HGB/LightGBM 与 GRU 可在同一测试集比较
4. 回测报告包含扣费后收益和回撤
5. 能清楚判断模型是否真的覆盖手续费
```

真正进入策略优化的标准：

```text
测试集扣费后 avg_net_pnl > 0
profit_factor > 1.1
最大回撤可接受
交易次数足够但不过度频繁
跨多个交易对不是单点偶然有效
```

## 当前执行记录

### 已完成

1. 已扫描全量数据目录，确认共有 22 个交易对，且均包含 15m 与 1h 数据。
2. 已建立统一数据加载、特征工程、交易标签、时间切分、统一扣费回测。
3. 已跑全 22 个交易对的 1h / horizon=4 HGB 基线。
4. 已跑全 22 个交易对的 1h / horizon=4 GRU 基线，参数为 `lookback=64, epochs=10`。
5. 已完成 confidence 阈值过滤与 horizon 扫描。

### 环境备注

当前本机 `quant` 环境中可运行 `torch`，GRU 已使用 CUDA 训练。

`lightgbm` 当前未成功安装，因此表格基线暂时使用 sklearn `HistGradientBoostingClassifier`。后续如要换 LightGBM，只需要安装依赖后复用同一套脚本。

### 1h / 全 22 交易对 / 扣费后阶段结果

手续费默认：

```text
开仓 0.05%
平仓 0.05%
往返 0.10%
滑点 0
```

HGB horizon 扫描中，裸跑全部为负；加入 confidence 阈值后，只有少数高置信度组合转正：

```text
HGB h8  threshold=0.75 trades=860  avg_net_pnl=+0.0715% total_net_pnl=+0.6149 profit_factor=1.0785
HGB h12 threshold=0.80 trades=621  avg_net_pnl=+0.0631% total_net_pnl=+0.3918 profit_factor=1.0570
HGB h12 threshold=0.75 trades=1674 avg_net_pnl=+0.0049% total_net_pnl=+0.0824 profit_factor=1.0040
```

GRU 当前版本在 h2 / h4 / h8 / h12 均未扣费后转正。最佳接近盈亏平衡的是：

```text
GRU h2 threshold=0.80 trades=880 avg_net_pnl=-0.0069% total_net_pnl=-0.0610 profit_factor=0.9871
GRU h8 threshold=0.65 trades=31066 avg_net_pnl=-0.0323% total_net_pnl=-10.0331 profit_factor=0.9639
```

### 当前判断

1. confidence 过滤有效，能显著减少亏损；但 GRU 当前 argmax+confidence 版本还没有真正覆盖手续费。
2. horizon 拉长对表格模型有帮助，h8/h12 比 h2/h4 更接近可交易。
3. HGB 的正收益样本交易次数较少，暂时只能说明存在信号苗头，不能直接视为稳定策略。
4. 下一步优先改进标签与决策层，而不是盲目加深 GRU：

```text
class_weight / focal_loss
更严格 NO_TRADE 标签
按 pair 分层统计阈值
按月/年份检查稳定性
加入验证集阈值选择，避免测试集调参
```

## 当前执行记录 2：class_weight / focal_loss / 验证集阈值 / 稳定性

### 已完成代码改造

1. `src/train_gru.py` 已支持：

```text
--loss ce
--loss weighted_ce
--loss focal
--loss weighted_focal
--focal-gamma
```

默认使用 `weighted_focal`，类别权重只由训练集标签分布计算。

2. `src/train_gru.py` 与 `src/train_hgb.py` 均已改成：

```text
训练集: 拟合模型
验证集: 选择 confidence threshold
测试集: 只使用验证集选出的 threshold 评估
```

避免继续在测试集上挑阈值。

3. `src/backtest.py` 已增加：

```text
summarize_by_month
select_threshold
```

每次训练会输出：

```text
*_val_thresholds.csv
*_val_trades.csv
*_raw_trades.csv
*_trades.csv
*_by_pair.csv
*_by_month.csv
```

### 正式验证口径

先只跑最有希望的：

```text
interval=1h
horizon=8
lookback=64
epochs=10
all 22 pairs
fee=0.10%
```

### GRU weighted_focal 结果

验证集最佳阈值：

```text
threshold=0.50
val trades=708
val total_net_pnl=-0.8089
val avg_net_pnl=-0.1143%
val profit_factor=0.9246
```

测试集结果：

```text
trades=575
net_win_rate=54.43%
avg_net_pnl=+0.5723%
total_net_pnl=+3.2906
max_drawdown=-1.0069
profit_factor=1.6328
```

按月份拆分：

```text
2025-12 trades=20  total=-0.0767
2026-01 trades=85  total=+0.1868
2026-02 trades=156 total=+3.6650
2026-03 trades=245 total=-0.4332
2026-04 trades=69  total=-0.0513
```

结论：测试集正收益主要由 2026-02 贡献，稳定性还不够。

贡献最大的交易对：

```text
ARBUSDT       +0.6134
DOGEUSDT     +0.5248
BTCUSDT      +0.4294
1000PEPEUSDT +0.4164
AAVEUSDT     +0.3764
ENAUSDT      +0.3393
```

拖累最大的交易对：

```text
ENJUSDT  -0.2607
CHZUSDT  -0.1629
BCHUSDT  -0.1427
ANKRUSDT -0.0501
ATOMUSDT -0.0340
```

### HGB h8 验证集阈值基线

验证集最佳阈值：

```text
threshold=0.80
val trades=144
val total_net_pnl=+1.4631
val avg_net_pnl=+1.0161%
val profit_factor=2.0294
```

测试集结果：

```text
trades=265
net_win_rate=48.30%
avg_net_pnl=-0.0164%
total_net_pnl=-0.0433
profit_factor=0.9805
```

结论：HGB 在验证集很好，但测试集回落到略负，说明高置信度信号存在明显时间漂移。

### 当前新判断

1. `weighted_focal` 确实让 GRU 的交易频率大幅下降，并能在测试集挑出更高质量样本。
2. 但 GRU 的验证集阈值本身不是正收益，测试集正收益不能直接视为稳健策略。
3. HGB 出现相反现象：验证集很好，测试集略负，更说明不能依赖单段测试。
4. 下一步应该做 walk-forward 多窗口验证，而不是继续单次 train/val/test。

## 当前执行记录 3：walk-forward 多窗口验证

### 已完成代码改造

新增：

```text
src/splits.py::walk_forward_splits
src/walk_forward_hgb.py
```

walk-forward 每一折按如下流程执行：

```text
1. 训练窗口拟合模型
2. 验证窗口选择 confidence threshold
3. 测试窗口只使用验证集选出的 threshold
4. 窗口向后滚动，重复以上流程
```

输出文件包括：

```text
*_folds.csv
*_trades.csv
*_raw_trades.csv
*_by_pair.csv
*_by_month.csv
*_foldN_val_thresholds.csv
```

### 本轮正式运行

命令口径：

```text
conda run -n quant python -m src.walk_forward_hgb \
  --interval 1h \
  --bar-rule 1h \
  --horizon 8 \
  --model hgb \
  --train-bars 6000 \
  --val-bars 1000 \
  --test-bars 720 \
  --step-bars 720 \
  --min-val-trades 100
```

含义：

```text
训练约 250 天
验证约 42 天
测试约 30 天
每 30 天滚动一次
```

### 全 22 交易对结果

```text
folds=14
positive_folds=8 / 14
trades=15094
net_win_rate=50.86%
avg_net_pnl=+0.0405%
total_net_pnl=+6.1085
max_drawdown=-12.2380
profit_factor=1.0323
```

### 分折结果摘要

明显盈利折：

```text
fold 1  total=+5.6749 PF=1.3778
fold 9  total=+2.5096 PF=1.1543
fold 13 total=+4.0120 PF=2.4618
```

明显亏损折：

```text
fold 4  total=-4.5339 PF=0.5830
fold 10 total=-1.2580 PF=0.6550
fold 14 total=-0.9628 PF=0.8480
```

### 按月份稳定性

主要贡献：

```text
2025-02 total=+10.1134 PF=1.4921
2025-10 total=+2.3224  PF=1.1503
2026-02 total=+3.8747  PF=2.0610
```

主要拖累：

```text
2025-05 total=-4.9961 PF=0.5461
2025-11 total=-1.3989 PF=0.6177
2026-03 total=-1.1202 PF=0.7894
2025-03 total=-3.2722 PF=0.9597
```

### 按交易对稳定性

主要贡献：

```text
1000BONKUSDT total=+4.8688 PF=1.2009
DOTUSDT      total=+1.5225 PF=1.1530
ARBUSDT      total=+1.1628 PF=1.0803
AVAXUSDT     total=+0.9092 PF=1.0742
ENAUSDT      total=+0.7283 PF=1.0368
```

主要拖累：

```text
1000PEPEUSDT total=-1.8812 PF=0.9194
ARCUSDT      total=-1.5616 PF=0.5680
ADAUSDT      total=-0.6480 PF=0.9573
BNBUSDT      total=-0.3923 PF=0.9334
BTCUSDT      total=-0.2187 PF=0.9646
```

### 当前判断

walk-forward 后，HGB h8 不是完全随机：14 折里 8 折为正，整体扣费后也为正。

但边际很薄：

```text
profit_factor=1.0323
avg_net_pnl=+0.0405%
max_drawdown=-12.2380
```

这还达不到可上线策略标准。下一步应该优先做：

```text
1. 只保留 walk-forward 中跨月份更稳定的交易对池
2. 加入按市场状态过滤，避开 2025-05 / 2025-11 / 2026-03 这类失效段
3. 对 GRU 做同样 walk-forward，但先减少折数或 epochs 控制成本
4. 增加阈值选择约束，例如验证集 PF > 1 且 avg_net_pnl > 0，否则测试段不交易
```

## 当前执行记录 4：验证集门槛过滤

### 已完成代码改造

`src/walk_forward_hgb.py` 已加入验证集门槛参数：

```text
--gate-min-trades
--gate-min-profit-factor
--gate-min-avg-net-pnl
--gate-min-total-net-pnl
```

每一折流程变为：

```text
1. 在验证集选择 confidence threshold
2. 检查该 threshold 的验证集表现
3. 如果验证集不达标，该折测试段全部 NO_TRADE
4. 如果验证集达标，测试段使用验证集选出的 threshold
```

本轮门槛：

```text
验证集 trades >= 100
验证集 profit_factor >= 1
验证集 avg_net_pnl >= 0
验证集 total_net_pnl >= 0
```

### 全 22 交易对 h8 门槛版结果

```text
folds=14
gate_pass_folds=11 / 14
positive_folds=5 / 14
trades=14493
net_win_rate=50.48%
avg_net_pnl=+0.0111%
total_net_pnl=+1.6120
max_drawdown=-12.7225
profit_factor=1.0088
```

对比无门槛版本：

```text
无门槛: total_net_pnl=+6.1085, PF=1.0323, positive_folds=8/14
有门槛: total_net_pnl=+1.6120, PF=1.0088, positive_folds=5/14
```

### 关键观察

门槛过滤跳过了 3 折：

```text
fold 5
fold 7
fold 13
```

但这 3 折在无门槛测试中实际都是正收益，尤其：

```text
fold 13 无门槛测试 total_net_pnl=+4.0120
```

同时，门槛没有过滤掉明显亏损的：

```text
fold 4  total=-4.5339
fold 10 total=-1.2580
fold 14 total=-0.9628
```

### 当前判断

简单的“验证集 PF > 1 且 avg_net_pnl > 0”不能可靠预测下一段测试窗口是否有效，反而削弱了结果。

这说明问题不只是 threshold，而是需要识别市场状态是否延续。下一步不要继续简单提高门槛，应改为：

```text
1. 用验证集指标做特征，学习/规则化判断下一折是否交易
2. 加入市场状态过滤，例如波动率、趋势强度、BTC regime、全市场 breadth
3. 对亏损集中的月份和交易对做 regime 归因
4. 或者按交易对单独 gate，而不是整折统一 gate
```

## 下一步计划：高胜率方向

当前目标从“尽量多交易”转向“优先提高胜率与稳定性”。

### 需要新增的数据/特征

#### 1. 多空比

当前 CVD 中的 `taker_buy_vol / taker_sell_vol` 只代表主动买卖量，不等于账户/持仓多空比。

下一步应扫描 Binance 数据目录，确认是否存在以下数据：

```text
global long/short account ratio
top trader long/short account ratio
top trader long/short position ratio
```

如果存在，应接入特征工程。

重点组合特征：

```text
OI 上升 + 多空比上升 = 多头加杠杆
OI 上升 + 多空比下降 = 空头加杠杆
价格不涨 + 多空比过高 = 多头拥挤
价格不跌 + 多空比过低 = 空头拥挤
CVD 与多空比背离 = 主动成交和账户方向不一致
```

#### 2. 布林带与价格位置

布林带不是新数据，而是 Price 派生特征。它更适合作为高胜率过滤器，而不是单独策略。

需要新增：

```text
boll_mid
boll_upper
boll_lower
boll_percent_b
boll_band_width
distance_to_upper
distance_to_mid
distance_to_lower
```

用途：

```text
避免在价格贴近上轨时盲目做多
避免在价格贴近下轨时盲目做空
区分趋势行情和震荡行情
```

注意：

```text
趋势行情中突破上轨可能继续涨
震荡行情中上轨附近可能更适合反向
```

所以布林带特征必须与趋势/波动率状态一起判断。

#### 3. 波动率与趋势强度

高胜率模型需要知道当前环境是否适合预测。

建议新增：

```text
ATR
realized volatility
rolling high-low range
过去 N 根最大回撤
过去 N 根最大反弹
MA slope
price distance to MA
ADX 或简化趋势强度
RSI
rolling rank / stochastic position
```

### 新验证口径

新增特征后不能只看分类准确率，也不能只看总收益。

必须继续使用 walk-forward，并额外观察：

```text
扣费后胜率
avg_net_pnl
profit_factor
max_drawdown
交易次数
正收益折比例
按月份稳定性
按交易对稳定性
```

尤其要检查：

```text
胜率提高后，avg_net_pnl 是否同步提高
胜率提高后，profit_factor 是否同步提高
胜率提高是否只是靠交易次数过少
```

### 明天优先执行顺序

```text
1. 扫描 Binance 数据目录，确认是否存在多空比数据文件
2. 如果存在，接入 data_loader
3. 在 feature_engineering 中加入多空比特征
4. 加入布林带、ATR、RSI、趋势强度等 Price 派生特征
5. 重新跑 HGB h8 walk-forward
6. 对比新增特征前后的胜率、PF、avg_net_pnl、正收益折比例
7. 再决定是否把同样特征接入 GRU walk-forward
```

## 当前执行记录 5：研究初期缩小交易对范围

用户决定研究初期先只跑：

```text
BTCUSDT
BNBUSDT
```

原因：

```text
1. 全 22 个交易对训练和 walk-forward 太慢
2. 小币种噪声较大，早期不利于判断模型是否真的有效
3. 先用 BTC/BNB 做稳定基准，再决定是否扩展到小币种
```

当前数据目录检查结果：

```text
BTCUSDT 存在
BNBUSDT 存在
```

代码已支持指定交易对参数：

```text
--pairs BTCUSDT,BNBUSDT
```

已接入脚本：

```text
src/train_hgb.py
src/train_gru.py
src/walk_forward_hgb.py
src/walk_forward_direction_filter.py
src/scan_direction_filters.py
```

如果请求的交易对不存在，加载器会提示：

```text
Warning: requested pairs not found for interval=1h: ETHUSDT
```

当前 BTC/BNB 数据量：

```text
BTCUSDT: 17444 rows
BNBUSDT: 17431 rows
时间范围: 2024-04-18 07:00:00 -> 2026-04-15 02:00:00
```

### BTC/BNB 方向过滤阶段结果

长窗口正式命令曾运行超过 10 分钟未完成：

```text
conda run -n quant python -m src.walk_forward_direction_filter \
  --interval 1h \
  --bar-rule 1h \
  --horizon 8 \
  --model hgb \
  --pairs BTCUSDT,BNBUSDT \
  --train-bars 6000 \
  --val-bars 1000 \
  --test-bars 720 \
  --step-bars 720 \
  --min-val-candidates 20
```

因此先跑了研究期快速版：

```text
conda run -n quant python -m src.walk_forward_direction_filter \
  --interval 1h \
  --bar-rule 1h \
  --horizon 8 \
  --model hgb \
  --pairs BTCUSDT,BNBUSDT \
  --train-bars 3000 \
  --val-bars 500 \
  --test-bars 360 \
  --step-bars 360 \
  --max-folds 12 \
  --min-val-candidates 10
```

结果文件：

```text
results/direction_walk_forward_hist_gradient_boosting_1h_h8_tr3000_val500_te360_pairsBTCUSDT-BNBUSDT_summary.csv
results/direction_walk_forward_hist_gradient_boosting_1h_h8_tr3000_val500_te360_pairsBTCUSDT-BNBUSDT_folds.csv
results/direction_walk_forward_hist_gradient_boosting_1h_h8_tr3000_val500_te360_pairsBTCUSDT-BNBUSDT_trades.csv
```

最佳方向过滤候选：

```text
LONG_conf0.8_distance_to_ma_50_<=q0.1

含义:
模型预测 LONG
confidence >= 0.80
当前价格相对 MA50 的偏离处在验证集低分位区域
```

测试集结果：

```text
交易次数: 172
方向命中: 124 / 172 = 72.09%
平均单笔不扣费收益: 0.8206%
平均单笔扣费后收益: 0.7206%
累计不扣费收益: 1.4115
累计扣费后收益: 1.2395
正收益折: 8 / 12
```

按交易对拆分：

```text
BNBUSDT: 120 trades, 89 wins, 74.17%
BTCUSDT: 52 trades, 35 wins, 67.31%
```

阶段判断：

```text
1. BTC+BNB 上已经出现超过 70% 方向命中的候选过滤器。
2. 该信号主要来自 LONG，不是完整多空系统。
3. BNB 明显强于 BTC，BTC 单独还没有到 70%。
4. 样本数仍偏少，下一步不能急着扩大交易，应该先验证稳定性。
```

下一步优先：

```text
1. 给 direction filter 增加 confidence-only 候选，验证 BNB SHORT conf>=0.70 这类简单规则
2. 对 BTC/BNB 分别单独跑，避免 BNB 掩盖 BTC
3. 扩展 SHORT 侧候选: RSI 高位、距离布林下轨远、rally_from_low 高分位
4. 尝试安装 LightGBM 到 quant 环境，加速正式长窗口 walk-forward
5. 再跑 tr6000/val1000/te720 的正式版
```

### 口径修正：horizon 必须为 1

用户明确要求：

```text
horizon 必须改成 1
```

因此当前标准口径改为：

```text
bar interval: 1h
bar rule: 1h
horizon: 1

含义:
每根 bar 是 1 小时
预测下一根 1h bar
持仓周期约 1 小时
```

之前 horizon=8 的 72.09% 方向命中不能作为当前目标口径的有效结论，只能作为“更长持仓周期可能更容易预测”的参考。

已按 BTCUSDT + BNBUSDT 重跑 horizon=1 快速版：

```text
conda run -n quant python -m src.walk_forward_direction_filter \
  --interval 1h \
  --bar-rule 1h \
  --horizon 1 \
  --model hgb \
  --pairs BTCUSDT,BNBUSDT \
  --train-bars 3000 \
  --val-bars 500 \
  --test-bars 360 \
  --step-bars 360 \
  --max-folds 12 \
  --min-val-candidates 10
```

结果文件：

```text
results/direction_walk_forward_hist_gradient_boosting_1h_h1_tr3000_val500_te360_pairsBTCUSDT-BNBUSDT_summary.csv
results/direction_walk_forward_hist_gradient_boosting_1h_h1_tr3000_val500_te360_pairsBTCUSDT-BNBUSDT_folds.csv
results/direction_walk_forward_hist_gradient_boosting_1h_h1_tr3000_val500_te360_pairsBTCUSDT-BNBUSDT_trades.csv
```

horizon=1 当前结果：

```text
最佳候选:
LONG_conf0.8_rsi_50_<=q0.2

交易次数: 11
方向命中: 7 / 11 = 63.64%
平均单笔不扣费收益: 0.0951%
平均单笔扣费后收益: -0.0049%
累计扣费后收益: -0.000542
正收益折: 3 / 12
```

按交易对：

```text
BTCUSDT: 10 trades, 7 wins, 70.00%
BNBUSDT: 1 trade, 0 wins, 0.00%
```

阶段判断：

```text
1. horizon=1 后，默认高置信过滤器几乎不开仓。
2. 当前 70% 只出现在 BTC 的 10 笔极小样本，不能视为稳定。
3. BNB 在 horizon=1 下没有复现 horizon=8 的优势。
4. 下一步需要围绕 horizon=1 重新设计候选过滤器，不能沿用 horizon=8 的结论。
```

horizon=1 下一步优先：

```text
1. 给 direction filter 增加 confidence-only 候选，避免特征过滤过严导致无交易
2. 降低/扫描 confidence: 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
3. 分别跑 BTCUSDT 和 BNBUSDT，避免合并后信号互相稀释
4. 增加 SHORT 侧候选，尤其是 RSI 高位、rolling_rank 高位、rally_from_low 高分位
5. 对 horizon=1 单独做标签再平衡和阈值选择，因为 NO_TRADE 占比已经达到约 34.39%
```

### 口径修正：方向正确率、波动收益、扣费收益分开

用户指出：

```text
计算方向胜率不应该把手续费加进来。
应该拆成三层：
1. 方向正确率
2. 波动收益
3. 扣除手续费的收益
```

代码调整：

```text
src/labeling.py
- add_trade_labels 新增 label_mode
- label_mode=trade: 使用 open_fee + close_fee + buffer 作为 LONG/SHORT 标签门槛
- label_mode=direction: 使用 future_return 正负作为 LONG/SHORT 标签，手续费不参与方向标签

src/walk_forward_direction_filter.py
- 新增 --label-mode trade|direction
- 输出字段从 gross_win_rate 改为 direction_win_rate
- 同时保留 avg_gross_pnl / total_gross_pnl / avg_net_pnl / total_net_pnl
```

当前标准实验应使用：

```text
--horizon 1
--label-mode direction
```

已按 BTCUSDT + BNBUSDT 重跑纯方向标签：

```text
conda run -n quant python -m src.walk_forward_direction_filter \
  --interval 1h \
  --bar-rule 1h \
  --horizon 1 \
  --label-mode direction \
  --model hgb \
  --pairs BTCUSDT,BNBUSDT \
  --train-bars 3000 \
  --val-bars 500 \
  --test-bars 360 \
  --step-bars 360 \
  --max-folds 12 \
  --min-val-candidates 10
```

结果文件：

```text
results/direction_walk_forward_hist_gradient_boosting_1h_h1_direction_tr3000_val500_te360_pairsBTCUSDT-BNBUSDT_summary.csv
results/direction_walk_forward_hist_gradient_boosting_1h_h1_direction_tr3000_val500_te360_pairsBTCUSDT-BNBUSDT_folds.csv
results/direction_walk_forward_hist_gradient_boosting_1h_h1_direction_tr3000_val500_te360_pairsBTCUSDT-BNBUSDT_trades.csv
```

纯方向标签分布：

```text
SHORT: 17094, 49.02%
NO_TRADE: 41, 0.12%
LONG: 17738, 50.86%
```

最佳候选：

```text
LONG_conf0.8_drawdown_from_high_50_<=q0.1
```

三层结果：

```text
方向正确率: 46 / 79 = 58.23%
波动收益 avg_gross_pnl: -0.0539%
扣费收益 avg_net_pnl: -0.1539%
累计波动收益 total_gross_pnl: -0.042591
累计扣费收益 total_net_pnl: -0.121591
```

按交易对：

```text
BNBUSDT: 69 trades, 39 wins, direction_win_rate 56.52%, avg_gross_pnl -0.0928%
BTCUSDT: 10 trades, 7 wins, direction_win_rate 70.00%, avg_gross_pnl +0.2147%
```

阶段判断：

```text
1. 拆开手续费后，horizon=1 的真实方向能力目前只有 58.23%，远不到稳定 70%。
2. 虽然方向胜率略高于随机，但波动收益为负，说明错的时候亏得更大。
3. BTC 的 70% 仍然只有 10 笔，不能作为有效结论。
4. 当前模型/过滤器还不能交易，下一步应专门围绕 direction_win_rate 做候选扫描。
```

### BTC 单独过滤器实验

用户要求：

```text
每个交易对应该有自己的单独过滤器，先做 BTC。
```

代码调整：

```text
src/walk_forward_direction_filter.py
- 新增 --candidates-file，避免长候选列表和 >= / <= 在命令行中冲突

configs/btc_h1_direction_candidates.txt
- BTC horizon=1 纯方向候选列表
- 包含 LONG/SHORT、confidence 0.55-0.80、RSI/布林/MA/回撤/反弹等条件
```

BTC 单独实验命令：

```text
conda run -n quant python -m src.walk_forward_direction_filter \
  --interval 1h \
  --bar-rule 1h \
  --horizon 1 \
  --label-mode direction \
  --model hgb \
  --pairs BTCUSDT \
  --train-bars 3000 \
  --val-bars 500 \
  --test-bars 360 \
  --step-bars 360 \
  --max-folds 12 \
  --min-val-candidates 10 \
  --candidates-file configs\btc_h1_direction_candidates.txt
```

结果文件：

```text
results/direction_walk_forward_hist_gradient_boosting_1h_h1_direction_tr3000_val500_te360_pairsBTCUSDT_summary.csv
results/direction_walk_forward_hist_gradient_boosting_1h_h1_direction_tr3000_val500_te360_pairsBTCUSDT_folds.csv
results/direction_walk_forward_hist_gradient_boosting_1h_h1_direction_tr3000_val500_te360_pairsBTCUSDT_trades.csv
```

BTC 单独最佳候选：

```text
LONG_conf0.75_distance_to_boll_upper_14_<=q0.1

含义:
模型预测 LONG
confidence >= 0.75
distance_to_boll_upper_14 处在验证集最低 10% 区域
```

三层结果：

```text
交易次数: 61
方向正确率: 36 / 61 = 59.02%
平均波动收益: +0.1679%
累计波动收益: +0.102432
平均扣费后收益: +0.0679%
累计扣费后收益: +0.041432
正收益折: 6 / 12
```

按月份：

```text
2024-10: 2 trades, 100.00%, avg_gross +0.2255%
2024-11: 2 trades, 50.00%, avg_gross +0.4216%
2024-12: 12 trades, 83.33%, avg_gross +0.2457%
2025-01: 8 trades, 50.00%, avg_gross +0.1835%
2025-02: 20 trades, 50.00%, avg_gross +0.0716%
2025-03: 17 trades, 52.94%, avg_gross +0.1824%
```

阶段判断：

```text
1. BTC 单独过滤器目前没有达到 70% 方向正确率。
2. 最佳稳定候选方向正确率只有 59.02%，但波动收益和扣费后收益为正。
3. 70%+ 的结果只出现在极小样本，例如 7 笔，不应采信。
4. BTC 的下一步重点不是继续追高 confidence，而是增加更贴近下一小时方向的特征/候选。
```

### BTC 2025-02 拆分观察

以后所有结果固定拆成三层：

```text
1. 方向正确率
2. 波动收益，不扣手续费
3. 扣除手续费后的收益
```

BTC 最佳候选在 2025-02 的表现：

```text
交易次数: 20
方向正确率: 10 / 20 = 50.00%
平均波动收益: +0.0716%
累计波动收益: +0.014325
平均扣费后收益: -0.0284%
累计扣费后收益: -0.005675
```

盈亏幅度：

```text
盈利 10 笔: 平均 +0.5305%，合计 +0.053054
亏损 10 笔: 平均 -0.3873%，合计 -0.038729
```

按日期：

```text
2025-02-03: 1 trade, 100.00%, total_gross +0.001399
2025-02-25: 6 trades, 83.33%, total_gross +0.015327
2025-02-26: 5 trades, 40.00%, total_gross +0.002937
2025-02-28: 8 trades, 25.00%, total_gross -0.005339
```

阶段判断：

```text
1. 2025-02 的正波动收益不是来自方向胜率，而是来自盈亏比。
2. 该过滤器是 LONG + 远离布林上轨，本质更像超跌/弱势后的反弹捕捉。
3. 2 月交易集中在少数几天，尤其 2025-02-25 到 2025-02-28。
4. 2025-02-28 触发很多，但方向正确率只有 25%，说明极端波动/下跌延续时会连续误开多。
5. 若目标是高胜率，需要识别并过滤 2025-02-28 这类“继续下跌而非反弹”的环境。
```

### BTC 反弹过滤优化

针对 2025-02 的问题，做了两类优化：

```text
1. walk_forward_direction_filter 支持复合候选条件
   格式:
   SIDE:confidence:feature:op:quantile[:feature:op:quantile...]

2. 新增 BTC 反弹过滤候选文件
   configs/btc_h1_direction_rebound_filters.txt
```

优化思路：

```text
原始过滤器:
LONG + 远离布林上轨

新增二级过滤:
ma_slope_50 不要太差
realized_vol / ATR 不要太高
boll_percent_b / distance_to_boll_lower 不要太贴下轨
stochastic_position 不要太低
```

2025-02 的亏损交易更像“下跌延续”：

```text
ma_slope_50 更负
波动率更高
价格更贴近布林下轨
```

优化后实用候选 1：

```text
LONG_conf0.7_AND_distance_to_boll_upper_14_<=q0.1_AND_ma_slope_50_>=q0.5

交易次数: 46
方向正确率: 29 / 46 = 63.04%
平均波动收益: +0.1837%
累计波动收益: +0.084493
平均扣费后收益: +0.0837%
累计扣费后收益: +0.038493
正收益折: 6 / 12
```

问题：

```text
它完全过滤掉了 2025-02。
说明 ma_slope_50 对下跌延续过滤有效，但也可能过严。
```

优化后实用候选 2，更适合继续观察 2025-02：

```text
LONG_conf0.7_AND_distance_to_boll_upper_14_<=q0.1_AND_boll_percent_b_14_>=q0.2

交易次数: 62
方向正确率: 39 / 62 = 62.90%
平均波动收益: +0.1965%
累计波动收益: +0.121830
平均扣费后收益: +0.0965%
累计扣费后收益: +0.059830
正收益折: 7 / 12
```

候选 2 在 2025-02：

```text
交易次数: 16
方向正确率: 8 / 16 = 50.00%
平均波动收益: +0.1177%
平均扣费后收益: +0.0177%
累计扣费后收益: +0.002838
```

2025-02 按日期：

```text
2025-02-25: 5 trades, 80.00%, avg_net +0.1644%
2025-02-26: 4 trades, 50.00%, avg_net +0.0030%
2025-02-28: 7 trades, 28.57%, avg_net -0.0786%
```

阶段判断：

```text
1. 复合过滤能把 BTC 方向正确率从 59.02% 提升到约 63%。
2. 扣费后收益也改善，说明过滤不是单纯减少交易。
3. 但 2025-02-28 的方向问题还没有彻底解决。
4. ma_slope_50 能直接过滤掉这类环境，但也把 2025-02 整体过滤掉了。
5. 下一步应专门为 2025-02-28 增加“暴跌延续识别”：短周期 CVD/OI 恶化、连续阴线、ATR 扩张、下轨贴近度、价格创新低后未修复。
```

# OrderFlow ML

从原始 Binance parquet 数据重建订单流机器学习项目。

当前研究口径已经改为先拆开观察三层结果：

```text
1. 方向正确率
2. 波动收益，不扣手续费
3. 扣除手续费后的收益
```

默认成本：

```text
开仓 0.05%
平仓 0.05%
滑点 0
```

当前标准预测目标：

```text
bar: 1h
horizon: 1
label-mode: direction

含义:
用历史窗口预测下一根 1h bar 的方向。
方向正确率不加入手续费。
收益统计再分别看不扣费和扣费后。
```

## 当前结果

更新时间：2026-04-30

当前重点交易对：`BTCUSDT`

当前较实用的 BTC 过滤器：

```text
LONG_conf0.7_AND_distance_to_boll_upper_14_<=q0.1_AND_boll_percent_b_14_>=q0.2
```

含义：

```text
模型预测 LONG
confidence >= 0.70
价格处在远离布林上轨的低位区域
但不能太贴近布林下轨，避免一部分下跌延续
```

结果：

```text
交易次数: 62

1. 方向正确率:
39 / 62 = 62.90%

2. 波动收益:
平均单笔 +0.1965%
累计 +0.121830

3. 扣除手续费后的收益:
平均单笔 +0.0965%
累计 +0.059830
```

对比基础 BTC 过滤器：

```text
LONG_conf0.75_distance_to_boll_upper_14_<=q0.1

交易次数: 61
方向正确率: 36 / 61 = 59.02%
平均波动收益: +0.1679%
平均扣费后收益: +0.0679%
```

阶段结论：

```text
复合过滤后，BTC horizon=1 的方向正确率从约 59% 提升到约 63%。
扣费后收益也同步改善。
当前仍未达到稳定 70% 方向正确率。
下一步重点是识别 2025-02-28 这类下跌延续环境，减少超跌做多误触发。
```

相关文件：

```text
agent.md
configs/btc_h1_direction_candidates.txt
configs/btc_h1_direction_rebound_filters.txt
src/walk_forward_direction_filter.py
```

## 快速检查数据

```bash
python -m src.data_catalog
```

## 跑 BTC 1h 下一根方向过滤

```bash
python -m src.walk_forward_direction_filter \
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
  --candidates-file configs/btc_h1_direction_rebound_filters.txt
```

输出在 `results/`。

## 跑 1h 全市场 LightGBM/HGB 基线

```bash
python -m src.train_hgb --interval 1h --bar-rule 1h --horizon 1 --pairs BTCUSDT,BNBUSDT
```

输出在 `results/`。

## 构建缓存数据集

```bash
python -m src.build_dataset --interval 1h --bar-rule 1h --horizon 1 --buffer 0
```

输出在 `datasets/`。

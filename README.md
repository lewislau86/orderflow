# OrderFlow ML

从原始 Binance parquet 数据重建订单流机器学习项目。

核心目标不是预测 A-F 场景，而是预测扣除手续费后是否值得交易：

```text
LONG / SHORT / NO_TRADE
```

默认成本：

```text
开仓 0.05%
平仓 0.05%
滑点 0
```

## 快速检查数据

```bash
python -m src.data_catalog
```

## 跑 1h 全市场 LightGBM/HGB 基线

```bash
python -m src.train_hgb --interval 1h --bar-rule 1h --horizon 4 --buffer 0.0005
```

如果想先快速冒烟测试：

```bash
python -m src.train_hgb --interval 1h --bar-rule 1h --horizon 4 --buffer 0.0005 --max-pairs 3
```

输出在 `results/`。

## 构建缓存数据集

```bash
python -m src.build_dataset --interval 1h --bar-rule 1h --horizon 4 --buffer 0.0005
```

输出在 `datasets/`。

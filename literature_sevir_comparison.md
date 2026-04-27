# SEVIR 论文对标摘录（源码表格）

归档说明：本文件为快速阅读版；可复核与来源追踪请同时查看 literature_sevir_archive.md。

## 1) Earthformer (arXiv:2207.05833)

来源：论文源码文件 earthformer.tex 中 SEVIR 表格。

### 表 A：SEVIR 主表（CSI-M 与阈值，MSE）
- Earthformer: CSI-M=0.4419, CSI-219=0.1791, CSI-181=0.2848, CSI-160=0.3232, CSI-133=0.4271, CSI-74=0.6860, CSI-16=0.7513, MSE=3.6957e-3
- ConvLSTM: CSI-M=0.4185, CSI-219=0.1288, CSI-133=0.4052, CSI-74=0.6793, MSE=3.7532e-3
- PredRNN: CSI-M=0.4080, CSI-219=0.1312, CSI-133=0.3858, CSI-74=0.6713, MSE=3.9014e-3

### 表 B：SEVIR Challenge 口径（含 MAE）
- Earthformer: CSI-219=0.1480, CSI-181=0.2748, CSI-160=0.3126, CSI-133=0.4231, CSI-74=0.6886, CSI-16=0.7682, CSI-M3=0.6266, CSI-M6=0.4359, MSE=3.6702e-3, MAE=2.5112e-2
- ConvLSTM: CSI-219=0.1071, CSI-133=0.4155, CSI-74=0.6873, MSE=3.7532e-3, MAE=2.5898e-2

## 2) PreDiff (arXiv:2307.10422)

来源：论文源码文件 prediff.tex 中 SEVIR 表格。

### 表 C：SEVIR 主表（含 CRPS/FVD）
- PreDiff: FVD=33.05, CRPS=0.0246, CSI=0.4100, CSI-pool4=0.4624, CSI-pool16=0.6244
- Earthformer: FVD=690.7, CRPS=0.0304, CSI=0.4419, CSI-pool4=0.4567, CSI-pool16=0.5005
- ConvLSTM: FVD=659.7, CRPS=0.0332, CSI=0.4185, CSI-pool4=0.4452, CSI-pool16=0.5135

### 表 D：PreDiff 分阈值表（与本项目可对齐的阈值）
- PreDiff pool1: CSI-74=0.6740, CSI-133=0.4119, CSI-219=0.1154
- PreDiff pool4: CSI-74=0.6691, CSI-133=0.4807, CSI-219=0.2065
- PreDiff pool16: CSI-74=0.7789, CSI-133=0.6638, CSI-219=0.3865

## 3) SimCast (arXiv:2510.07953, ICME 2025)

来源：论文源码文件 camera_ready.tex 中 SEVIR 对比表。

### 表 E：SEVIR 主表（Deterministic）
- SimCast: CRPS=0.0270, SSIM=0.7252, HSS=0.5834, CSI-M(pool1/4/16)=0.4521/0.4750/0.4968, CSI-181(pool1/4/16)=0.3099/0.3343/0.3571, CSI-219(pool1/4/16)=0.2007/0.2517/0.3268
- SimVP: CRPS=0.0259, SSIM=0.7772, HSS=0.5280, CSI-M(pool1/4/16)=0.4153/0.4226/0.4530, CSI-181(pool1/4/16)=0.2532/0.2604/0.3000, CSI-219(pool1/4/16)=0.1338/0.1394/0.1685
- EarthFormer*: CRPS=0.0251, SSIM=0.7756, HSS=0.5411, CSI-M(pool1/4/16)=0.4310/0.4319/0.4351, CSI-181(pool1/4/16)=0.2622/0.2542/0.2562, CSI-219(pool1/4/16)=0.1448/0.1409/0.1481

### 表 F：SEVIR 主表（Probabilistic / Fusion）
- PreDiff: CRPS=0.0202, SSIM=0.7648, HSS=0.4914, CSI-M(pool1/4/16)=0.3875/0.3918/0.4157, CSI-181(pool1/4/16)=0.2076/0.2069/0.2264, CSI-219(pool1/4/16)=0.1032/0.1051/0.1213
- CasCast(EarthFormer): CRPS=0.0202, SSIM=0.7797, HSS=0.5602, CSI-M(pool1/4/16)=0.4401/0.4640/0.5225, CSI-181(pool1/4/16)=0.2879/0.3179/0.3900, CSI-219(pool1/4/16)=0.1851/0.2127/0.2841
- CasCast(SimCast): CRPS=0.0259, SSIM=0.7620, HSS=0.5771, CSI-M(pool1/4/16)=0.4467/0.4811/0.5501, CSI-181(pool1/4/16)=0.3049/0.3470/0.4252, CSI-219(pool1/4/16)=0.1856/0.2353/0.3387

## 4) SimVP 变体论文可比性说明

- SimVPv2 (arXiv:2211.12509): 论文主实验聚焦 MovingMNIST/TaxiBJ/WeatherBench/Caltech/KTH 等，源码检索未发现 SEVIR 的 CSI/CRPS 对标表。
- TAU (arXiv:2206.12126): 论文同样未包含 SEVIR nowcasting 的 CSI/POD/FAR/CRPS 表格，主要是通用视频预测与交通等基准。
- 结论：当前可直接用于 SEVIR 同口径对标的 SimVP 系数字值，优先采用 SimCast 论文主表中的 SimVP/SimCast 对照。

## 5) 本地实验（HF-SimVP/result/benchmark_summary_20260427_105934.csv）

- Baseline: CRPS=0.0391, MSE=0.0070, SSIM=0.6089, CSI-M/H/E-POOL1=0.5792/0.2845/0.2623
- Enhanced: CRPS=0.0309, MSE=0.0067, SSIM=0.7740, CSI-M/H/E-POOL1=0.6356/0.4053/0.5621

阈值映射：M/H/E 分别对应 74/133/219。

## 6) 口径注意事项

- Earthformer 与 PreDiff 的表格是官方论文口径，SEVIR 切分与实现细节可能与本项目不同。
- PreDiff 文中说明使用 downsampled SEVIR（受算力限制），与原始 384x384 训练评估不完全等价。
- 本项目 CRPS 在实现中等价于 MAE（point forecast 情形），与概率预测论文中的 CRPS 解释需谨慎对齐。

# SEVIR Literature Archive (Persistent)

This file is a persistent archive of extracted SEVIR benchmark numbers and provenance.
Updated: 2026-04-27

## A) Provenance and Local Sources

1. Earthformer (arXiv:2207.05833)
- Source bundle extracted at: /tmp/arxiv_src/earthformer_src
- Main source file: /tmp/arxiv_src/earthformer_src/earthformer.tex
- Key table labels used: table:sevir_csi, table:sevir_csi_challenge

2. PreDiff (arXiv:2307.10422)
- Source bundle extracted at: /tmp/arxiv_src
- Main source file: /tmp/arxiv_src/prediff.tex
- Key table labels used: table:sevir_csi, table:sevir_csi_thresholds, table:sevir_csi_p4_thresholds, table:sevir_csi_p16_thresholds

3. SimCast (arXiv:2510.07953)
- Source bundle extracted at: /tmp/arxiv_src/simcast_src
- Main source file: /tmp/arxiv_src/simcast_src/camera_ready.tex
- Key table label used: tab:eval_sevir

4. SimVP variants checked for SEVIR comparability
- SimVPv2 (arXiv:2211.12509), source path: /tmp/arxiv_src/simvpv2_src/tmm_simvpv2.tex
- TAU (arXiv:2206.12126), source path: /tmp/arxiv_src/tau_src/tau.tex
- Result: no SEVIR CSI/POD/FAR/CRPS benchmark table found in these two papers.

## B) Canonical Numbers (SEVIR)

### Earthformer (from earthformer.tex)
- CSI-M = 0.4419
- CSI-219 = 0.1791
- CSI-181 = 0.2848
- CSI-160 = 0.3232
- CSI-133 = 0.4271
- CSI-74 = 0.6860
- CSI-16 = 0.7513
- MSE = 3.6957e-3

Challenge-style table variant:
- CSI-M6 = 0.4359
- CSI-M3 = 0.6266
- MSE = 3.6702e-3
- MAE = 2.5112e-2

### PreDiff (from prediff.tex)
Main table:
- CRPS = 0.0246
- CSI = 0.4100
- CSI-pool4 = 0.4624
- CSI-pool16 = 0.6244

Threshold tables:
- pool1: CSI-74 = 0.6740, CSI-133 = 0.4119, CSI-219 = 0.1154
- pool4: CSI-74 = 0.6691, CSI-133 = 0.4807, CSI-219 = 0.2065
- pool16: CSI-74 = 0.7789, CSI-133 = 0.6638, CSI-219 = 0.3865

### SimCast (from camera_ready.tex)
Deterministic block:
- SimCast: CRPS = 0.0270, SSIM = 0.7252, HSS = 0.5834
  - CSI-M (pool1/4/16) = 0.4521 / 0.4750 / 0.4968
  - CSI-181 (pool1/4/16) = 0.3099 / 0.3343 / 0.3571
  - CSI-219 (pool1/4/16) = 0.2007 / 0.2517 / 0.3268
- SimVP baseline in the same table:
  - CRPS = 0.0259, SSIM = 0.7772, HSS = 0.5280
  - CSI-M (pool1/4/16) = 0.4153 / 0.4226 / 0.4530
  - CSI-181 (pool1/4/16) = 0.2532 / 0.2604 / 0.3000
  - CSI-219 (pool1/4/16) = 0.1338 / 0.1394 / 0.1685

Probabilistic/fusion block:
- PreDiff: CRPS = 0.0202, SSIM = 0.7648, HSS = 0.4914
- CasCast(EarthFormer): CRPS = 0.0202, SSIM = 0.7797, HSS = 0.5602
- CasCast(SimCast): CRPS = 0.0259, SSIM = 0.7620, HSS = 0.5771

## C) Local Experiment Snapshot

From /Users/amon/Desktop/code/HF-SimVP/result/benchmark_summary_20260427_105934.csv:
- Baseline: CRPS = 0.0391, MSE = 0.0070, SSIM = 0.6089
- Enhanced: CRPS = 0.0309, MSE = 0.0067, SSIM = 0.7740
- Threshold mapping in this project: M/H/E = 74/133/219

## D) Repro Commands (for future re-fetch)

1. SimCast source and table grep
- cd /tmp/arxiv_src && rm -rf simcast_src && mkdir -p simcast_src && cd simcast_src
- curl -L --fail -o simcast.tar.gz https://arxiv.org/e-print/2510.07953
- tar -xzf simcast.tar.gz
- grep -RIn "SEVIR\|CSI\|POD\|SUCR\|FAR\|MSE\|MAE\|CRPS\|table" . | head -n 260

2. Earthformer source and table grep
- cd /tmp/arxiv_src && rm -rf earthformer_src && mkdir -p earthformer_src && cd earthformer_src
- curl -L --fail -o earthformer.tar.gz https://arxiv.org/e-print/2207.05833
- tar -xzf earthformer.tar.gz
- grep -RIn "SEVIR\|CSI\|POD\|SUCR\|MSE\|MAE\|SSIM\|table" . | head -n 240

3. PreDiff source and table grep
- cd /tmp/arxiv_src && curl -L --fail -o prediff.tar.gz https://arxiv.org/e-print/2307.10422
- tar -xzf prediff.tar.gz
- grep -RIn "SEVIR\|CSI\|POD\|SSIM\|MSE\|MAE\|SUCR\|FAR\|table" . | head -n 200

## E) Notes

- Cross-paper direct comparison may be affected by evaluation protocol differences (pooling rule, averaging rule, challenge-vs-paper CSI definitions, downsample settings, split details).
- This archive is designed for fast reuse and traceability before writing final paper claims.

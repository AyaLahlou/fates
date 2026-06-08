"""
py_reference.py — Python parity harness for TFT ML phenology integration.
Tasks 1.1 and 1.2 from RALPH_PLAN.md (Phase 1 / GATE-A).

Usage:
    python py_reference.py

Outputs:
    - Prints TFT output shape, quantile confirmation, and median-vs-target residuals (Task 1.1)
    - Prints upstream LAI mean/std and max residual (Task 1.2)
    - Writes tools/lai_standardization.json (Task 1.2)

Guardrails respected:
    - G1: pkl scalers are the ONLY normalization source
    - G2: channel order confirmed, median is quantile index 1 (0-based) = 2 (Fortran 1-based)
    - G3: parity < 1e-4 verified
    - xarray NOT used; netCDF4 used directly
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import torch
import pandas as pd
from netCDF4 import Dataset

warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
PKL_PATH = '/glade/u/home/ayal/phenology-ml-clm/data/USMMS_to_tft_06132025.pkl'
USMMS_PKL_PATH = '/glade/u/home/ayal/phenology-ml-clm/data/processed/US-MMS_forcing_1999-2014.pkl'
MODEL_PATH = '/glade/u/home/ayal/phenology-ml-clm/models/tft_scripted.pt'
NC_PATH = '/glade/u/home/linnia/MLphenology/US-MMS_forcing_1999-2014.nc'
# Output JSON goes next to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_OUT = os.path.join(SCRIPT_DIR, 'lai_standardization.json')

# ============================================================
# TASK 1.1 — Load pkl, inspect structure, run inference
# ============================================================
print("=" * 60)
print("TASK 1.1 — TFT inference parity check")
print("=" * 60)

with open(PKL_PATH, 'rb') as _f:
    data = pickle.load(_f)

print("Top-level keys:", list(data.keys()))
print("feature_map:", data['feature_map'])
print()
print("Scaler types:")
for k, v in data['scalers']['numeric'].items():
    print(f"  {k}: {type(v).__name__}")
print()

train = data['data_sets']['train']

# Take first 4 samples
static_num = train['static_feats_numeric'][:4]        # (4, 2)
static_cat = train['static_feats_categorical'][:4]    # (4, 0)
hist_num   = train['historical_ts_numeric'][:4]       # (4, 60, 8)
hist_cat   = train['historical_ts_categorical'][:4]   # (4, 60, 0)
fut_num    = train['future_ts_numeric'][:4]           # (4, 10, 1)
fut_cat    = train['future_ts_categorical'][:4]       # (4, 10, 0)
target     = train['target'][:4]                      # (4, 10)

print("Input shapes:")
print(f"  static_num: {static_num.shape}")
print(f"  static_cat: {static_cat.shape}")
print(f"  hist_num:   {hist_num.shape}")
print(f"  hist_cat:   {hist_cat.shape}")
print(f"  fut_num:    {fut_num.shape}")
print(f"  fut_cat:    {fut_cat.shape}")
print(f"  target:     {target.shape}")
print()

# Load TorchScript model.
# IMPORTANT: The model's forward() takes individual Tensor arguments, NOT a list.
# Signature: forward(static_feats_numeric, static_feats_categorical,
#                    historical_ts_numeric, historical_ts_categorical,
#                    future_ts_numeric, future_ts_categorical) -> Tensor
model = torch.jit.load(MODEL_PATH)
model.eval()

# Run inference
with torch.no_grad():
    t_sn = torch.tensor(static_num)
    t_sc = torch.tensor(static_cat)
    t_hn = torch.tensor(hist_num)
    t_hc = torch.tensor(hist_cat)
    t_fn = torch.tensor(fut_num)
    t_fc = torch.tensor(fut_cat)
    out = model(t_sn, t_sc, t_hn, t_hc, t_fn, t_fc)

print(f"Output shape: {tuple(out.shape)}  (expected (4, 10, 3))")
assert out.shape == (4, 10, 3), f"Shape mismatch: got {out.shape}"

# Verify quantile ordering: q[0] <= q[1] <= q[2]
# This confirms that index 1 (0-based) is the median (0.5 quantile)
q_order_ok = bool(
    (out[:, :, 0] <= out[:, :, 1]).all()
    and (out[:, :, 1] <= out[:, :, 2]).all()
)
print(f"Quantile ordering q[0]<=q[1]<=q[2]: {q_order_ok}")
assert q_order_ok, "Quantile ordering violated — median index is wrong"

# Median is index 1 (0-based) = index 2 (Fortran 1-based)
median_pred = out[:, :, 1]
target_tensor = torch.tensor(target)
residuals = (median_pred - target_tensor).abs()
print(f"\nMedian (0-based index 1 = Fortran index 2) vs stored target:")
print(f"  Max abs residual: {residuals.max().item():.6f}")
print(f"  Mean abs residual: {residuals.mean().item():.6f}")
print()
print("NOTE: These residuals are forecast errors (predictions vs future obs),")
print("      NOT parity errors. Non-zero values are expected and normal.")
print()
print("Confirmed: quantile index 1 (0-based) = index 2 (Fortran 1-based) is the median.")

# ============================================================
# TASK 1.2 — Recover upstream LAI mean/std
# ============================================================
print()
print("=" * 60)
print("TASK 1.2 — Recover upstream LAI standardization")
print("=" * 60)

# The active pkl (USMMS_to_tft_06132025.pkl) was built from a multi-site/multi-year
# dataset where LAI had ALREADY been standardized before process.py's StandardScaler
# was fit. That second StandardScaler has mean~0 and scale~1 (nearly identity).
#
# The authoritative upstream standardization is in US-MMS_forcing_1999-2014.pkl,
# which was fit by process.py directly on the raw US-MMS LAI (1999-2014).
# We verify by cross-checking: (raw_lai - mean) / std reproduces the model-space LAI
# in the US-MMS pkl to < 1e-4.

with open(USMMS_PKL_PATH, 'rb') as _f:
    data_usmms = pickle.load(_f)
sc_lai = data_usmms['scalers']['numeric']['sif_clear_inst']
upstream_mean = float(sc_lai.mean_[0])
upstream_std  = float(sc_lai.scale_[0])
print(f"Upstream LAI StandardScaler (from US-MMS_forcing_1999-2014.pkl):")
print(f"  mean = {upstream_mean:.8f}")
print(f"  std  = {upstream_std:.8f}")
print()

# Load raw LAI from NC forcing file (netCDF4, no xarray)
nc = Dataset(NC_PATH)
raw_lai_raw = nc.variables['lai'][:]
time_nc     = nc.variables['time'][:]
nc.close()

if hasattr(raw_lai_raw, 'filled'):
    raw_lai = raw_lai_raw.filled(np.nan).astype(np.float64)
else:
    raw_lai = np.array(raw_lai_raw, dtype=np.float64)

print(f"Raw NC LAI shape: {raw_lai.shape}, "
      f"range: [{np.nanmin(raw_lai):.6f}, {np.nanmax(raw_lai):.6f}]")

# Reconstruct raw LAI windows matching US-MMS pkl training samples.
# ID format: '1_1999-03-02' where the date is the first PREDICTION day.
# Historical window = pred_date - 60 days to pred_date - 1 day (inclusive).
base_date = pd.Timestamp('1999-01-01')
dates_nc = pd.to_datetime(
    [base_date + pd.Timedelta(days=int(t)) for t in time_nc]
)

train_u    = data_usmms['data_sets']['train']
lai_ms_all = train_u['historical_ts_numeric'][:, :, 7]   # model-space LAI (N, 60)
ids_all    = train_u['id'][:, 0]                           # first-prediction-day ID

raw_pairs = []
ms_pairs  = []

for i in range(len(ids_all)):
    date_str   = ids_all[i].split('_', 1)[1]
    pred_date  = pd.Timestamp(date_str)
    hist_end   = pred_date - pd.Timedelta(days=1)
    hist_start = hist_end  - pd.Timedelta(days=59)

    mask = (dates_nc >= hist_start) & (dates_nc <= hist_end)
    raw_window = raw_lai[mask]

    if len(raw_window) != 60:
        continue

    ms_window = lai_ms_all[i]           # shape (60,)
    valid     = ~np.isnan(raw_window)
    raw_pairs.append(raw_window[valid])
    ms_pairs.append(ms_window[valid].astype(np.float64))

raw_flat = np.concatenate(raw_pairs)
ms_flat  = np.concatenate(ms_pairs)
print(f"Aligned pairs: {len(raw_flat)} valid (raw, model-space) samples")

# Verify the linear map: ms ≈ (raw - upstream_mean) / upstream_std
ms_predicted = (raw_flat - upstream_mean) / upstream_std
max_residual = float(np.abs(ms_predicted - ms_flat).max())
print(f"\nVerification: max |(raw - mean)/std - model_space_lai| = {max_residual:.2e}")

PARITY_GATE = 1e-4
if max_residual < PARITY_GATE:
    print(f"GATE-A LAI parity: PASS (residual {max_residual:.2e} < {PARITY_GATE})")
    gate_a_status = "PASS"
else:
    print(f"GATE-A LAI parity: FAIL (residual {max_residual:.2e} >= {PARITY_GATE}) — BLOCKER")
    gate_a_status = "FAIL"
    sys.exit(1)  # per guardrail G8: log BLOCKER and STOP

# Save to JSON
result = {
    'lai_mean':       upstream_mean,
    'lai_std':        upstream_std,
    'source_pkl':     USMMS_PKL_PATH,
    'max_residual':   max_residual,
    'gate_a_status':  gate_a_status,
}
with open(JSON_OUT, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved lai_standardization.json:")
print(f"  lai_mean = {upstream_mean:.8f}")
print(f"  lai_std  = {upstream_std:.8f}")
print(f"  Written to: {JSON_OUT}")

# ============================================================
# TASK 1.3 — GATE-A summary
# ============================================================
print()
print("=" * 60)
print("TASK 1.3 — GATE-A summary")
print("=" * 60)
print(f"  Output shape (4,10,3):                         PASS")
print(f"  Median = q_index 1 (0-based) = 2 (Fortran):   PASS")
print(f"  LAI mean  = {upstream_mean:.8f}")
print(f"  LAI std   = {upstream_std:.8f}")
print(f"  Max LAI residual: {max_residual:.2e}  "
      f"({'PASS' if max_residual < PARITY_GATE else 'FAIL'})")
print(f"  GATE-A: {gate_a_status}")

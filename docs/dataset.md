# Dataset: Azure Public Dataset V2 (2019 VM trace)

**Why 2019, not 2020.** The 2020 Azure VM release (`AzureTracesForPacking2020`) is a VM *allocation* trace (request times, lifetimes, VM type sizes) with no utilization time series, so it cannot be used for forecasting. The 2019 release (V2) is the latest Azure VM trace with utilization readings. The 2017 release (V1) has the same schema and is a possible second dataset for a robustness check.

**What it contains.** 30 consecutive days, 2,695,548 VMs, 1.94 B readings. Per VM, every 5 minutes: min / max / avg **CPU utilization** (%). There is **no memory utilization time series** in this trace; memory appears only as a per-VM size bucket in `vmtable`. The project therefore forecasts CPU (avg CPU as target; min/max available as extra input channels). If memory forecasting matters for the paper, Alibaba Cluster Trace 2018 (`machine_usage` / `container_usage`) has both CPU and memory.

**License / citation.** CC-BY 4.0. Cite Cortez et al., SOSP 2017 (`cortez2017resource` in `CITATIONS.bib`), as the dataset README requests.

**Hosting.** Files are GitHub release assets of `Azure/AzurePublicDataset` (tag `dataset-v2`); the old Azure Blob URLs redirect there. Links: `AzurePublicDatasetLinksV2.txt` in that repo.

## Our subset

The readings are split into 195 gzipped CSVs of ~856 MB each (156 GB total), ordered by time: each file holds about 3.7 hours of readings for every VM. A per-VM series therefore needs a pass over all 195 files. `scripts/fetch_azure2019.py` streams each file through `gzip -dc | awk` and keeps only the selected VMs, so nothing large touches disk (about 15-20 minutes with 4 cores and a fast link).

Selection (seed 612): VMs alive for the full 30 days, category Interactive or Delay-insensitive (the `Unknown` category is dropped), lifetime avg CPU >= 2% (drops near-idle VMs that make coverage trivially easy), 1000 per category. 157,526 VMs pass the filter.

`scripts/build_dataset.py` pivots the rows onto the 8,639-step 5-minute grid, forward-fills gaps of up to 6 steps (30 min), drops VMs with more than 2% missing, and keeps the pre-fill mask (`observed`) so evaluation skips filled targets.

## Files

| File | Where | Contents |
|---|---|---|
| `data/azure2019_cpu_2000vm.npz` | shared folder | full processed subset, `cpu [n_vm, 8639, 3]` float32 (min, max, avg CPU %), `observed`, VM ids, category, core / memory bucket |
| `data/sample/azure2019_cpu_sample100.npz` | shared folder | 100-VM random subset for quick experiments |
| raw filtered parts, `vmtable.csv.gz` | not kept | regenerate with `fetch_azure2019.py` (vmtable 440 MB, parts ~2 GB) |

## Splits

Chronological and shared by all VMs: days 0-18 train, 18-21 validation, 21-30 test. A window belongs to the split containing its first target step. A single global model is trained across VMs.

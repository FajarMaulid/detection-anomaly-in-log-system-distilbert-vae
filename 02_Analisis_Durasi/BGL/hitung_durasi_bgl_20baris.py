"""
Skrip: hitung_durasi_bgl_20baris.py
Tujuan: Menghitung rata-rata waktu yang dibutuhkan dataset BGL
        untuk menghasilkan / memproses 20 baris log.

Dua aspek yang dihitung:
  1. Durasi timestamp real-time (sliding window stride=1 dan block window stride=20)
  2. Waktu inferensi pipeline model (DistilBERT + VAE) dari laporan yang ada

Output: bgl_20baris_rata_rata_waktu.csv dan .xlsx
"""

import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime

# ─── Konfigurasi Path ────────────────────────────────────────────────────────
BASE_DIR     = r"c:\Users\Fajar M\Downloads\Output"
CSV_PATH     = os.path.join(BASE_DIR, "success_lines_BGL.csv")
INFER_JSON   = os.path.join(BASE_DIR, "Tes Inferensi", "inference_latency_report.json")
OUT_CSV      = os.path.join(BASE_DIR, "bgl_20baris_rata_rata_waktu.csv")
OUT_XLSX     = os.path.join(BASE_DIR, "bgl_20baris_rata_rata_waktu.xlsx")
WINDOW_SIZE  = 20

print("=" * 70)
print("  Kalkulasi Rata-rata Waktu Dataset BGL untuk 20 Baris")
print("=" * 70)

# ─── 1. Baca CSV dan ekstrak timestamp ────────────────────────────────────────
print(f"\n[1/4] Membaca dataset: {CSV_PATH}")
t0 = time.time()
df = pd.read_csv(CSV_PATH, usecols=["raw"])
print(f"      → {len(df):,} baris dibaca dalam {time.time()-t0:.1f}s")

# Ekstrak timestamp dari kolom raw (field ke-5, format: YYYY-MM-DD-HH.MM.SS.ffffff)
print("\n[2/4] Mengekstrak timestamp dari kolom 'raw'...")
t0 = time.time()

def parse_timestamp(raw_str):
    """Ekstrak timestamp dari string log BGL."""
    try:
        parts = str(raw_str).split()
        # Timestamp ada di indeks ke-4 (0-indexed), format: 2005-06-03-15.42.50.363779
        ts_str = parts[4]
        # Konversi format BGL ke datetime standar
        ts_str = ts_str.replace("-", " ", 3).replace(".", ":", 2)
        # Format sekarang: "2005 06 03 15:42:50.363779"
        parts2 = ts_str.split(" ")
        date_part = f"{parts2[0]}-{parts2[1]}-{parts2[2]}"
        time_part = parts2[3]
        dt_str = f"{date_part} {time_part}"
        return pd.to_datetime(dt_str, format="%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        return pd.NaT

timestamps = df["raw"].apply(parse_timestamp)
valid_mask = timestamps.notna()
timestamps = timestamps[valid_mask].reset_index(drop=True)
ts_seconds = timestamps.astype(np.int64) / 1e9   # konversi ke detik (epoch)

n_total = len(ts_seconds)
elapsed = time.time() - t0
print(f"      → {n_total:,} timestamp valid diekstrak dalam {elapsed:.1f}s")

# ─── 3. Hitung durasi window ──────────────────────────────────────────────────
print(f"\n[3/4] Menghitung durasi window ukuran {WINDOW_SIZE}...")
t0 = time.time()

# --- 3a. Sliding Window (stride=1) ---
# Durasi window ke-i = timestamp[i+19] - timestamp[i]
n_sliding = n_total - WINDOW_SIZE + 1
print(f"      → Sliding Window (stride=1): {n_sliding:,} window...")

ts_arr = ts_seconds.values
# Vektor durasi: selisih antara baris pertama dan ke-20 tiap window
durations_sliding = ts_arr[WINDOW_SIZE - 1:] - ts_arr[:n_sliding]

mean_sliding   = float(np.mean(durations_sliding))
median_sliding = float(np.median(durations_sliding))
std_sliding    = float(np.std(durations_sliding))
p10_sliding    = float(np.percentile(durations_sliding, 10))
p25_sliding    = float(np.percentile(durations_sliding, 25))
p75_sliding    = float(np.percentile(durations_sliding, 75))
p90_sliding    = float(np.percentile(durations_sliding, 90))
p95_sliding    = float(np.percentile(durations_sliding, 95))
p99_sliding    = float(np.percentile(durations_sliding, 99))
min_sliding    = float(np.min(durations_sliding))
max_sliding    = float(np.max(durations_sliding))

# --- 3b. Block Window (stride=20) ---
n_blocks = n_total // WINDOW_SIZE
print(f"      → Block Window (stride=20):  {n_blocks:,} blok...")

block_start_idx = np.arange(n_blocks) * WINDOW_SIZE
block_end_idx   = block_start_idx + WINDOW_SIZE - 1

# Pastikan tidak melebihi panjang array
valid_blocks = block_end_idx < n_total
block_start_idx = block_start_idx[valid_blocks]
block_end_idx   = block_end_idx[valid_blocks]
n_blocks        = len(block_start_idx)

durations_block = ts_arr[block_end_idx] - ts_arr[block_start_idx]

mean_block   = float(np.mean(durations_block))
median_block = float(np.median(durations_block))
std_block    = float(np.std(durations_block))
p10_block    = float(np.percentile(durations_block, 10))
p25_block    = float(np.percentile(durations_block, 25))
p75_block    = float(np.percentile(durations_block, 75))
p90_block    = float(np.percentile(durations_block, 90))
p95_block    = float(np.percentile(durations_block, 95))
p99_block    = float(np.percentile(durations_block, 99))
min_block    = float(np.min(durations_block))
max_block    = float(np.max(durations_block))

# --- 3c. Interval antar 1 baris (untuk referensi) ---
intervals = np.diff(ts_arr)
mean_1row  = float(np.mean(intervals))

elapsed = time.time() - t0
print(f"      → Selesai dalam {elapsed:.1f}s")

# ─── 4. Baca laporan inferensi pipeline ───────────────────────────────────────
print(f"\n[4/4] Membaca laporan inferensi: {INFER_JSON}")
with open(INFER_JSON, "r") as f:
    infer_data = json.load(f)

e2e = infer_data["end_to_end_latency_ms"]
stages = infer_data["stage_breakdown_ms"]

infer_mean_ms     = e2e["mean_ms"]
infer_median_ms   = e2e["median_ms"]
infer_p90_ms      = e2e["p90_ms"]
infer_p95_ms      = e2e["p95_ms"]
infer_p99_ms      = e2e["p99_ms"]
infer_max_ms      = e2e["max_ms"]
infer_n_samples   = e2e["n_samples"]
infer_throughput  = e2e["throughput_window_per_s"]
infer_window_size = infer_data["window_size"]

stage_drain3_ms   = stages["drain3_parsing"]["mean_ms"]
stage_bert_ms     = stages["distilbert_embedding"]["mean_ms"]
stage_vae_ms      = stages["scaling_vae"]["mean_ms"]

print(f"      → Window size inferensi: {infer_window_size}")
print(f"      → Jumlah sampel diukur:  {infer_n_samples:,}")
print(f"      → Rata-rata end-to-end:  {infer_mean_ms:.4f} ms")

# ─── 5. Susun hasil ke DataFrame ──────────────────────────────────────────────
print("\n[5/5] Menyusun dan menyimpan hasil...")

rows = [
    # ── Durasi Timestamp (dari data mentah BGL) ──
    {
        "Kategori"                 : "Timestamp BGL (real-time)",
        "Skenario"                 : f"Sliding Window (stride=1, window=20)",
        "Jumlah Window"            : n_sliding,
        "Rata-rata (detik)"        : round(mean_sliding, 6),
        "Rata-rata (ms)"           : round(mean_sliding * 1000, 4),
        "Rata-rata (menit)"        : round(mean_sliding / 60, 4),
        "Rata-rata (jam)"          : round(mean_sliding / 3600, 4),
        "Median / p50 (detik)"     : round(median_sliding, 6),
        "Std Dev (detik)"          : round(std_sliding, 6),
        "p10 (detik)"              : round(p10_sliding, 6),
        "p25 (detik)"              : round(p25_sliding, 6),
        "p75 (detik)"              : round(p75_sliding, 6),
        "p90 (detik)"              : round(p90_sliding, 6),
        "p95 (detik)"              : round(p95_sliding, 6),
        "p99 (detik)"              : round(p99_sliding, 6),
        "Minimum (detik)"          : round(min_sliding, 6),
        "Maksimum (detik)"         : round(max_sliding, 6),
        "Maksimum (jam)"           : round(max_sliding / 3600, 2),
        "Laju Log (baris/s)"       : round(1 / mean_1row, 2) if mean_1row > 0 else None,
    },
    {
        "Kategori"                 : "Timestamp BGL (real-time)",
        "Skenario"                 : f"Block Window (stride=20, window=20)",
        "Jumlah Window"            : n_blocks,
        "Rata-rata (detik)"        : round(mean_block, 6),
        "Rata-rata (ms)"           : round(mean_block * 1000, 4),
        "Rata-rata (menit)"        : round(mean_block / 60, 4),
        "Rata-rata (jam)"          : round(mean_block / 3600, 4),
        "Median / p50 (detik)"     : round(median_block, 6),
        "Std Dev (detik)"          : round(std_block, 6),
        "p10 (detik)"              : round(p10_block, 6),
        "p25 (detik)"              : round(p25_block, 6),
        "p75 (detik)"              : round(p75_block, 6),
        "p90 (detik)"              : round(p90_block, 6),
        "p95 (detik)"              : round(p95_block, 6),
        "p99 (detik)"              : round(p99_block, 6),
        "Minimum (detik)"          : round(min_block, 6),
        "Maksimum (detik)"         : round(max_block, 6),
        "Maksimum (jam)"           : round(max_block / 3600, 2),
        "Laju Log (baris/s)"       : round(1 / mean_1row, 2) if mean_1row > 0 else None,
    },
    # ── Waktu Inferensi Pipeline (model DistilBERT + VAE) ──
    {
        "Kategori"                 : "Inferensi Pipeline (DistilBERT+VAE)",
        "Skenario"                 : f"End-to-End (window={infer_window_size}, GPU CUDA, n={infer_n_samples:,})",
        "Jumlah Window"            : infer_n_samples,
        "Rata-rata (detik)"        : round(infer_mean_ms / 1000, 6),
        "Rata-rata (ms)"           : round(infer_mean_ms, 4),
        "Rata-rata (menit)"        : round(infer_mean_ms / 1000 / 60, 6),
        "Rata-rata (jam)"          : round(infer_mean_ms / 1000 / 3600, 8),
        "Median / p50 (detik)"     : round(infer_median_ms / 1000, 6),
        "Std Dev (detik)"          : None,
        "p10 (detik)"              : None,
        "p25 (detik)"              : None,
        "p75 (detik)"              : None,
        "p90 (detik)"              : round(infer_p90_ms / 1000, 6),
        "p95 (detik)"              : round(infer_p95_ms / 1000, 6),
        "p99 (detik)"              : round(infer_p99_ms / 1000, 6),
        "Minimum (detik)"          : None,
        "Maksimum (detik)"         : round(infer_max_ms / 1000, 6),
        "Maksimum (jam)"           : round(infer_max_ms / 1000 / 3600, 8),
        "Laju Log (baris/s)"       : round(infer_throughput * WINDOW_SIZE, 2),
    },
    # ── Breakdown per Stage ──
    {
        "Kategori"                 : "Inferensi Pipeline – Stage Drain3",
        "Skenario"                 : f"Drain3 Parsing (window={infer_window_size})",
        "Jumlah Window"            : infer_n_samples,
        "Rata-rata (detik)"        : round(stage_drain3_ms / 1000, 6),
        "Rata-rata (ms)"           : round(stage_drain3_ms, 4),
        "Rata-rata (menit)"        : round(stage_drain3_ms / 1000 / 60, 8),
        "Rata-rata (jam)"          : round(stage_drain3_ms / 1000 / 3600, 10),
        "Median / p50 (detik)"     : round(stages["drain3_parsing"]["median_ms"] / 1000, 6),
        "Std Dev (detik)"          : None,
        "p10 (detik)"              : None,
        "p25 (detik)"              : None,
        "p75 (detik)"              : None,
        "p90 (detik)"              : round(stages["drain3_parsing"]["p90_ms"] / 1000, 6),
        "p95 (detik)"              : round(stages["drain3_parsing"]["p95_ms"] / 1000, 6),
        "p99 (detik)"              : round(stages["drain3_parsing"]["p99_ms"] / 1000, 6),
        "Minimum (detik)"          : None,
        "Maksimum (detik)"         : round(stages["drain3_parsing"]["max_ms"] / 1000, 6),
        "Maksimum (jam)"           : round(stages["drain3_parsing"]["max_ms"] / 1000 / 3600, 10),
        "Laju Log (baris/s)"       : None,
    },
    {
        "Kategori"                 : "Inferensi Pipeline – Stage DistilBERT",
        "Skenario"                 : f"DistilBERT Embedding (window={infer_window_size})",
        "Jumlah Window"            : infer_n_samples,
        "Rata-rata (detik)"        : round(stage_bert_ms / 1000, 6),
        "Rata-rata (ms)"           : round(stage_bert_ms, 4),
        "Rata-rata (menit)"        : round(stage_bert_ms / 1000 / 60, 6),
        "Rata-rata (jam)"          : round(stage_bert_ms / 1000 / 3600, 8),
        "Median / p50 (detik)"     : round(stages["distilbert_embedding"]["median_ms"] / 1000, 6),
        "Std Dev (detik)"          : None,
        "p10 (detik)"              : None,
        "p25 (detik)"              : None,
        "p75 (detik)"              : None,
        "p90 (detik)"              : round(stages["distilbert_embedding"]["p90_ms"] / 1000, 6),
        "p95 (detik)"              : round(stages["distilbert_embedding"]["p95_ms"] / 1000, 6),
        "p99 (detik)"              : round(stages["distilbert_embedding"]["p99_ms"] / 1000, 6),
        "Minimum (detik)"          : None,
        "Maksimum (detik)"         : round(stages["distilbert_embedding"]["max_ms"] / 1000, 6),
        "Maksimum (jam)"           : round(stages["distilbert_embedding"]["max_ms"] / 1000 / 3600, 8),
        "Laju Log (baris/s)"       : None,
    },
    {
        "Kategori"                 : "Inferensi Pipeline – Stage VAE",
        "Skenario"                 : f"Scaling + VAE Inference (window={infer_window_size})",
        "Jumlah Window"            : infer_n_samples,
        "Rata-rata (detik)"        : round(stage_vae_ms / 1000, 6),
        "Rata-rata (ms)"           : round(stage_vae_ms, 4),
        "Rata-rata (menit)"        : round(stage_vae_ms / 1000 / 60, 8),
        "Rata-rata (jam)"          : round(stage_vae_ms / 1000 / 3600, 10),
        "Median / p50 (detik)"     : round(stages["scaling_vae"]["median_ms"] / 1000, 6),
        "Std Dev (detik)"          : None,
        "p10 (detik)"              : None,
        "p25 (detik)"              : None,
        "p75 (detik)"              : None,
        "p90 (detik)"              : round(stages["scaling_vae"]["p90_ms"] / 1000, 6),
        "p95 (detik)"              : round(stages["scaling_vae"]["p95_ms"] / 1000, 6),
        "p99 (detik)"              : round(stages["scaling_vae"]["p99_ms"] / 1000, 6),
        "Minimum (detik)"          : None,
        "Maksimum (detik)"         : round(stages["scaling_vae"]["max_ms"] / 1000, 6),
        "Maksimum (jam)"           : round(stages["scaling_vae"]["max_ms"] / 1000 / 3600, 10),
        "Laju Log (baris/s)"       : None,
    },
]

result_df = pd.DataFrame(rows)

# Simpan CSV
result_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# Simpan XLSX dengan formatting
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    result_df.to_excel(writer, index=False, sheet_name="Hasil")
    ws = writer.sheets["Hasil"]

    # Auto-fit lebar kolom
    for col in ws.columns:
        max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 50)

print(f"\n✔ Hasil disimpan:")
print(f"   CSV  → {OUT_CSV}")
print(f"   XLSX → {OUT_XLSX}")

# ─── Tampilkan Ringkasan di Terminal ─────────────────────────────────────────
print("\n" + "=" * 70)
print("  RINGKASAN HASIL")
print("=" * 70)

print("\n📊 A. Durasi Real-Time 20 Baris di Dataset BGL (dari timestamp log)")
print(f"   Window Size     : {WINDOW_SIZE} baris")
print(f"   Total baris log : {n_total:,}")
print()
print(f"   ┌─ Sliding Window (stride=1, {n_sliding:,} window) ─────────────────────┐")
print(f"   │  Rata-rata  : {mean_sliding:>12.4f} detik  ({mean_sliding*1000:>10.2f} ms, {mean_sliding/60:.4f} menit)")
print(f"   │  Median     : {median_sliding:>12.4f} detik")
print(f"   │  Std Dev    : {std_sliding:>12.4f} detik")
print(f"   │  p10 / p90  : {p10_sliding:.4f} / {p90_sliding:.4f} detik")
print(f"   │  Min / Max  : {min_sliding:.4f} / {max_sliding:.4f} detik")
print(f"   └──────────────────────────────────────────────────────────────┘")
print()
print(f"   ┌─ Block Window (stride=20, {n_blocks:,} blok) ───────────────────────┐")
print(f"   │  Rata-rata  : {mean_block:>12.4f} detik  ({mean_block*1000:>10.2f} ms, {mean_block/60:.4f} menit)")
print(f"   │  Median     : {median_block:>12.4f} detik")
print(f"   │  Std Dev    : {std_block:>12.4f} detik")
print(f"   │  p10 / p90  : {p10_block:.4f} / {p90_block:.4f} detik")
print(f"   │  Min / Max  : {min_block:.4f} / {max_block:.4f} detik")
print(f"   └──────────────────────────────────────────────────────────────┘")

print("\n⚡ B. Waktu Inferensi Pipeline (DistilBERT + VAE, GPU CUDA)")
print(f"   Window Size     : {infer_window_size} baris")
print(f"   Sampel diukur   : {infer_n_samples:,}")
print()
print(f"   ┌─ End-to-End ──────────────────────────────────────────────────┐")
print(f"   │  Rata-rata  : {infer_mean_ms:>12.4f} ms  ({infer_mean_ms/1000:.6f} detik)")
print(f"   │  Median     : {infer_median_ms:>12.4f} ms")
print(f"   │  p90 / p99  : {infer_p90_ms:.4f} / {infer_p99_ms:.4f} ms")
print(f"   │  Throughput  : {infer_throughput:.2f} window/s  (~{infer_throughput*WINDOW_SIZE:.0f} baris/s)")
print(f"   └──────────────────────────────────────────────────────────────┘")
print()
print(f"   ┌─ Breakdown per Stage ─────────────────────────────────────────┐")
print(f"   │  Drain3 Parsing    : {stage_drain3_ms:>8.4f} ms")
print(f"   │  DistilBERT Embed  : {stage_bert_ms:>8.4f} ms  ({stage_bert_ms/infer_mean_ms*100:.1f}%)")
print(f"   │  Scaling + VAE     : {stage_vae_ms:>8.4f} ms  ({stage_vae_ms/infer_mean_ms*100:.1f}%)")
print(f"   └──────────────────────────────────────────────────────────────┘")

print(f"\n✅ Selesai! Dibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

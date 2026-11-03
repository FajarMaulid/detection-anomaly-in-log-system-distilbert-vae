"""
Skrip: hitung_durasi_block_hdfs.py
Tujuan: Menghitung rata-rata dan statistik deskriptif durasi berjalannya
        suatu block ID dalam dataset HDFS.

Definisi "durasi block ID":
    Durasi = timestamp log TERAKHIR - timestamp log PERTAMA
    untuk setiap block ID yang sama.

Output:
    - hdfs_block_duration_summary.csv     (statistik ringkasan)
    - hdfs_block_duration_summary.xlsx    (Excel dengan formatting)
    - hdfs_block_duration_per_block.csv   (durasi per block ID)
"""

import sys
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Pastikan output UTF-8
sys.stdout.reconfigure(encoding="utf-8")

# ─── Konfigurasi Path ────────────────────────────────────────────────────────
HDFS_LOG     = r"c:\Users\Fajar M\Documents\Kode\Log System Detection Anomaly\Dataset\HDFS.log"
LABEL_CSV    = r"c:\Users\Fajar M\Documents\Kode\Log System Detection Anomaly\Dataset\anomaly_label.csv"
BASE_DIR     = r"c:\Users\Fajar M\Downloads\Output"
OUT_SUMMARY  = f"{BASE_DIR}\\hdfs_block_duration_summary.csv"
OUT_XLSX     = f"{BASE_DIR}\\hdfs_block_duration_summary.xlsx"
OUT_PER_BLK  = f"{BASE_DIR}\\hdfs_block_duration_per_block.csv"

print("=" * 70)
print("  Kalkulasi Durasi Block ID Dataset HDFS")
print("=" * 70)

# ─── 1. Baca anomaly label ────────────────────────────────────────────────────
print("\n[1/4] Membaca label block ID...")
label_df = pd.read_csv(LABEL_CSV)
label_map = dict(zip(label_df["BlockId"], label_df["Label"]))
print(f"      -> {len(label_map):,} block ID dengan label tersedia")
print(f"         Normal : {sum(1 for v in label_map.values() if v=='Normal'):,}")
print(f"         Anomaly: {sum(1 for v in label_map.values() if v=='Anomaly'):,}")

# ─── 2. Parse HDFS.log, kumpulkan min/max timestamp per block ─────────────────
print(f"\n[2/4] Parsing HDFS.log ({HDFS_LOG})...")
print("      (ini akan memakan waktu beberapa menit...)")

# Format timestamp: YYMMDD HHMMSS  -> "081109 203518"
# Blok regex: blk_[+-]?\d+
BLK_RE = re.compile(r"blk_[+-]?\d+")

# Simpan {blk_id: [min_ts_epoch, max_ts_epoch]}
blk_first = {}   # blk -> epoch detik pertama
blk_last  = {}   # blk -> epoch detik terakhir
blk_count = {}   # blk -> jumlah log lines

t0 = time.time()
n_lines   = 0
n_err     = 0
BATCH     = 1_000_000

with open(HDFS_LOG, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        n_lines += 1
        if n_lines % BATCH == 0:
            elapsed = time.time() - t0
            print(f"      -> {n_lines/1e6:.1f}M baris diproses ({elapsed:.0f}s, "
                  f"{n_lines/elapsed/1e3:.0f}k baris/s)...")

        parts = line.split(None, 2)  # split jadi max 3 bagian
        if len(parts) < 3:
            n_err += 1
            continue

        date_str = parts[0]  # "081109"
        time_str = parts[1]  # "203518"

        # Parse timestamp -> epoch detik
        try:
            dt = datetime.strptime(date_str + time_str, "%y%m%d%H%M%S")
            epoch = dt.timestamp()
        except ValueError:
            n_err += 1
            continue

        # Cari semua block ID di baris ini
        blks = BLK_RE.findall(line)
        for blk in blks:
            if blk not in blk_first:
                blk_first[blk] = epoch
                blk_last[blk]  = epoch
                blk_count[blk] = 1
            else:
                if epoch < blk_first[blk]:
                    blk_first[blk] = epoch
                if epoch > blk_last[blk]:
                    blk_last[blk] = epoch
                blk_count[blk] += 1

elapsed = time.time() - t0
print(f"      -> Selesai: {n_lines:,} baris dalam {elapsed:.1f}s")
print(f"      -> Gagal parse: {n_err:,} baris")
print(f"      -> Unique block ID ditemukan: {len(blk_first):,}")

# ─── 3. Hitung durasi per block ───────────────────────────────────────────────
print("\n[3/4] Menghitung durasi dan menyusun statistik...")

records = []
for blk in blk_first:
    duration_s = blk_last[blk] - blk_first[blk]
    label = label_map.get(blk, "Unknown")
    records.append({
        "block_id"      : blk,
        "label"         : label,
        "n_log_lines"   : blk_count[blk],
        "first_ts"      : datetime.fromtimestamp(blk_first[blk]).strftime("%Y-%m-%d %H:%M:%S"),
        "last_ts"       : datetime.fromtimestamp(blk_last[blk]).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s"    : round(duration_s, 3),
        "duration_ms"   : round(duration_s * 1000, 3),
        "duration_min"  : round(duration_s / 60, 6),
        "duration_hr"   : round(duration_s / 3600, 8),
    })

per_blk_df = pd.DataFrame(records)
per_blk_df.sort_values("block_id", inplace=True)
per_blk_df.reset_index(drop=True, inplace=True)

# Simpan per-block
per_blk_df.to_csv(OUT_PER_BLK, index=False, encoding="utf-8-sig")
print(f"      -> Per-block disimpan: {OUT_PER_BLK}")

# ─── 4. Hitung statistik deskriptif (Overall, Normal, Anomaly) ───────────────
def compute_stats(ser, label_name, n_blocks):
    """Hitung statistik deskriptif dari series durasi (detik)."""
    arr = ser.dropna().values
    if len(arr) == 0:
        return {}
    return {
        "Kategori"              : label_name,
        "Jumlah Block ID"       : n_blocks,
        "Rata-rata / Mean (s)"  : round(float(np.mean(arr)), 4),
        "Rata-rata / Mean (ms)" : round(float(np.mean(arr)) * 1000, 4),
        "Rata-rata / Mean (min)": round(float(np.mean(arr)) / 60, 6),
        "Median / p50 (s)"      : round(float(np.median(arr)), 4),
        "Standar Deviasi (s)"   : round(float(np.std(arr)), 4),
        "Minimum (s)"           : round(float(np.min(arr)), 4),
        "p10 (s)"               : round(float(np.percentile(arr, 10)), 4),
        "p25 (s)"               : round(float(np.percentile(arr, 25)), 4),
        "p75 (s)"               : round(float(np.percentile(arr, 75)), 4),
        "p90 (s)"               : round(float(np.percentile(arr, 90)), 4),
        "p95 (s)"               : round(float(np.percentile(arr, 95)), 4),
        "p99 (s)"               : round(float(np.percentile(arr, 99)), 4),
        "Maksimum (s)"          : round(float(np.max(arr)), 4),
        "Maksimum (jam)"        : round(float(np.max(arr)) / 3600, 4),
        "Rata-rata Baris/Block" : round(float(per_blk_df.loc[per_blk_df["label"]==label_name, "n_log_lines"].mean()), 2) if label_name != "Semua Block" else round(float(per_blk_df["n_log_lines"].mean()), 2),
    }

all_dur     = per_blk_df["duration_s"]
normal_dur  = per_blk_df.loc[per_blk_df["label"] == "Normal",  "duration_s"]
anomaly_dur = per_blk_df.loc[per_blk_df["label"] == "Anomaly", "duration_s"]
unknown_dur = per_blk_df.loc[per_blk_df["label"] == "Unknown", "duration_s"]

stats_rows = []
stats_rows.append(compute_stats(all_dur,     "Semua Block",   len(per_blk_df)))
stats_rows.append(compute_stats(normal_dur,  "Normal",  int((per_blk_df["label"]=="Normal").sum())))
stats_rows.append(compute_stats(anomaly_dur, "Anomaly", int((per_blk_df["label"]=="Anomaly").sum())))
if len(unknown_dur) > 0:
    stats_rows.append(compute_stats(unknown_dur, "Unknown", int((per_blk_df["label"]=="Unknown").sum())))

summary_df = pd.DataFrame(stats_rows)

# ─── 5. Simpan Output ─────────────────────────────────────────────────────────
print(f"\n[4/4] Menyimpan hasil...")

summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    # Sheet 1: Ringkasan statistik
    summary_df.to_excel(writer, index=False, sheet_name="Statistik Deskriptif")
    ws1 = writer.sheets["Statistik Deskriptif"]
    for col in ws1.columns:
        max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
        ws1.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

    # Sheet 2: Per block ID (sample 10.000 agar tidak terlalu besar)
    sample_df = per_blk_df.head(10000)
    sample_df.to_excel(writer, index=False, sheet_name="Per Block ID (10k sample)")
    ws2 = writer.sheets["Per Block ID (10k sample)"]
    for col in ws2.columns:
        max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
        ws2.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

print(f"   Summary CSV  -> {OUT_SUMMARY}")
print(f"   Summary XLSX -> {OUT_XLSX}")
print(f"   Per-Block    -> {OUT_PER_BLK}")

# ─── Tampilkan Ringkasan di Terminal ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  RINGKASAN STATISTIK DURASI BLOCK ID HDFS")
print("=" * 70)

for _, row in summary_df.iterrows():
    n = int(row["Jumlah Block ID"])
    print(f"\n  [{row['Kategori']}]  ({n:,} block ID)")
    print(f"    Rata-rata     : {row['Rata-rata / Mean (s)']:>12.4f} s"
          f"  ({row['Rata-rata / Mean (ms)']:>10.2f} ms, {row['Rata-rata / Mean (min)']:.4f} menit)")
    print(f"    Median (p50)  : {row['Median / p50 (s)']:>12.4f} s")
    print(f"    Std Dev       : {row['Standar Deviasi (s)']:>12.4f} s")
    print(f"    Min / Max     : {row['Minimum (s)']:.4f} / {row['Maksimum (s)']:.4f} s")
    print(f"    p10 / p25     : {row['p10 (s)']:.4f} / {row['p25 (s)']:.4f} s")
    print(f"    p75 / p90     : {row['p75 (s)']:.4f} / {row['p90 (s)']:.4f} s")
    print(f"    p95 / p99     : {row['p95 (s)']:.4f} / {row['p99 (s)']:.4f} s")
    print(f"    Rata-rata baris/block: {row['Rata-rata Baris/Block']:.2f}")

print(f"\nSelesai! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Dokumentasi Sistem Deteksi Anomali Log

Repositori ini mengimplementasikan pipeline deteksi anomali pada data log sistem terdistribusi skala besar (**HDFS** dan **BGL**) berbasis representasi semantik (*transformer embeddings*) dan *unsupervised deep generative model* (**$\beta$-VAE**), dipadukan dengan teori nilai ekstrem (**Extreme Value Theory / EVT**) untuk penentuan ambang batas anomali dinamis.

---

## 1. Ringkasan Isi Setiap Folder

| Folder | Deskripsi Singkat Isi & Peran |
| :--- | :--- |
| **`01_Pra_Pemrosesan/`** | Berisi notebook pra-pemrosesan data log mentah menggunakan parser Drain3, visualisasi hasil parsing, serta berkas statistik deskriptif panjang sekuens, frekuensi template, dan rasio anomali. |
| **`02_Analisis_Durasi/`** | Berisi skrip otomatisasi penghitungan rentang waktu/durasi log per unit analitik: subfolder **`BGL/`** (jendela geser 20 baris) dan **`HDFS/`** (berdasarkan siklus hidup `block_id`), lengkap dengan output CSV/XLSX. |
| **`03_Eksperimen_Model/`** | Berisi implementasi training dan evaluasi 4 konfigurasi utama: <br>• `Percobaan_1_HDFS_BERT`<br>• `Percobaan_3_HDFS_DistilBERT`<br>• `Percobaan_4_BGL_DistilBERT`<br>• `Percobaan_5_BGL_BERT`<br>• `Eksperimen_Awal_HDFS` (arsip artefak visualisasi awal).<br>Setiap folder memuat notebook eksperimen, bobot model terbaik (`vae_best.pt`), log hyperparameter search, metrik evaluasi, dan grafik proyeksi t-SNE. |
| **`04_Optimasi_Hyperparameter/`** | Memuat ringkasan gabungan pencarian hyperparameter via Optuna (`hp_trials_summary_combined.csv/.xlsx`) serta visualisasi analisis distribusi parameter, korelasi antarseed, dan parallel coordinates di subfolder `visualisasi/`. |
| **`05_Evaluasi_5_Seed/`** | Berisi hasil eksperimen multi-seed (seed 42 s.d. 46) untuk menguji stabilitas dan reliabilitas model terhadap variasi inisialisasi pada seluruh kombinasi dataset dan embedder, dirangkum dalam `hasil_rangkuman_5seed.csv/.xlsx`. |
| **`06_Uji_Statistik/`** | Berisi skrip uji signifikansi statistik (uji normalitas Shapiro-Wilk, homogenitas varians Levene, paired t-test / Wilcoxon signed-rank, dan Cohen's d) serta uji signifikansi matriks konfusi (McNemar / Wilcoxon), lengkap dengan laporan teks hasil ujinya. |
| **`07_Benchmark_Inferensi/`** | Berisi notebook dan visualisasi benchmark latensi waktu inferensi end-to-end (tahap Drain3, ekstraksi embedding, dan rekonstruksi VAE) untuk menguji kelayakan sistem terhadap Service Level Agreement (SLA) produksi. |

---

## 2. Struktur dan Kegunaan Kode Utama (`.ipynb` & `.py`)

### A. Pra-Pemrosesan Log
* **Berkas:** `01_Pra_Pemrosesan/statistik_pra_pemrosesan_log.ipynb`
* **Kegunaan:** Menjalankan pembersihan dan parsing log mentah menjadi format terstruktur menggunakan Drain3 Template Miner.
* **Struktur Kode:**
  1. *Data Ingestion*: Membaca raw log HDFS dan BGL.
  2. *Drain3 Parsing*: Ekstraksi template log dan penanganan variabel dinamis (*parameters*).
  3. *Grouping & Windowing*: Pengelompokan log berbasis `block_id` (HDFS) atau *sliding window* 20 baris (BGL).
  4. *Statistical Profiling*: Menghitung statistik deskriptif panjang sekuens, frekuensi kemunculan template unik, dan mengekspor visualisasi proporsi label normal vs. anomali.

---

### B. Analisis Durasi Sekuens
* **Berkas:**
  * `02_Analisis_Durasi/BGL/hitung_durasi_bgl_20baris.py`
  * `02_Analisis_Durasi/HDFS/hitung_durasi_block_hdfs.py`
* **Kegunaan:** Menghitung karakteristik temporal (durasi eksekusi) dari sekuens log untuk memastikan relevansi pemodelan jendela waktu pada sistem terdistribusi.
* **Struktur Kode:**
  1. *Timestamp Parsing*: Mengonversi format tanggal-waktu log mentah ke objek datetime mikrodetik.
  2. *Window Duration Calculation*: Menghitung selisih waktu (`timestamp_akhir - timestamp_awal`) pada tiap blok HDFS atau per jendela 20 baris BGL.
  3. *Aggregation*: Menghitung mean, median, standar deviasi, min, dan max durasi, lalu menyimpannya ke format CSV dan Excel.

---

### C. Pipeline Pelatihan & Evaluasi Model (Inti Eksperimen)
* **Berkas:**
  * `03_Eksperimen_Model/Percobaan_1_HDFS_BERT/percobaan_1_hdfs_bert.ipynb`
  * `03_Eksperimen_Model/Percobaan_3_HDFS_DistilBERT/percobaan_3_hdfs_distilbert.ipynb`
  * `03_Eksperimen_Model/Percobaan_4_BGL_DistilBERT/percobaan_4_bgl_distilbert.ipynb`
  * `03_Eksperimen_Model/Percobaan_5_BGL_BERT/percobaan_5_bgl_bert.ipynb`
* **Kegunaan:** Mengimplementasikan pipeline deteksi anomali end-to-end: ekstraksi fitur semantik, pencarian hyperparameter, pelatihan deep generative model, dan deteksi anomali menggunakan EVT.
* **Struktur Kode:**
  1. **Log Parsing & Sekuens**: Parsing template Drain3 dan pengelompokan event log.
  2. **Transformer Embedding**: Ekstraksi representasi vektor teks menggunakan BERT (`bert-base-uncased`) atau DistilBERT (`distilbert-base-uncased`) dengan *mean pooling* pada sekuens log.
  3. **Arsitektur $\beta$-VAE**:
     * `VAEEncoder`: Jaringan feed-forward multi-layer yang memproyeksikan vektor embedding ke parameter distribusi laten ($\mu, \log \sigma^2$).
     * *Reparameterization Trick*: Sampling vektor laten $z = \mu + \epsilon \odot \sigma$ di mana $\epsilon \sim \mathcal{N}(0, I)$.
     * `VAEDecoder`: Rekonstruksi vektor laten $z$ kembali ke dimensi embedding awal.
     * `vae_loss`: Kombinasi *Mean Squared Error* (MSE rekonstruksi) dan penalti divergensi Kullback-Leibler (KL) yang diatur oleh bobot $\beta$.
  4. **Optimasi Hyperparameter (Optuna)**: Tuning otomatis untuk parameter `latent_dim`, `batch_size`, `beta`, dan `learning_rate` selama 30 trial.
  5. **Training Final**: Pelatihan model VAE dengan konfigurasi terbaik hasil tuning menggunakan data berlabel normal.
  6. **Thresholding Anomali Dinamis (EVT / GPD)**:
     * Menghitung *reconstruction loss* pada data validasi.
     * Mengestimasi ambang batas (*threshold*) ekstrem menggunakan metode *Peaks-Over-Threshold* (POT) dengan *Generalized Pareto Distribution* (GPD) pada target FPR tertentu.
     * Membandingkan hasil EVT dengan baseline persentil empiris (p90, p95, p99).
  7. **Evaluasi & Visualisasi**: Evaluasi komparatif metrik (Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix) serta visualisasi separasi ruang laten via t-SNE.

---

### D. Uji Signifikansi Statistik
* **Berkas:**
  * `06_Uji_Statistik/analyze.py`
  * `06_Uji_Statistik/analyze_cm.py`
* **Kegunaan:** Memvalidasi secara ilmiah apakah perbedaan performa antar model (misal BERT vs. DistilBERT) signifikan secara statistik atau hanya karena kebetulan acak across 5 seeds.
* **Struktur Kode:**
  1. *Data Loading*: Membaca metrik dari `hasil_rangkuman_5seed.csv`.
  2. *Asumsi Statistik*: Menjalankan uji Shapiro-Wilk (normalitas) dan uji Levene (homogenitas varians).
  3. *Uji Hipotesis Komparatif*:
     * Menjalankan *Paired t-test* (jika terdistribusi normal) atau *Wilcoxon Signed-Rank Test* (jika non-parametrik).
     * Menghitung *Effect Size* menggunakan *Cohen's d*.
  4. *Analisis Matriks Konfusi (`analyze_cm.py`)*: Uji signifikansi terhadap distribusi True Positive, False Positive, True Negative, dan False Negative antar model.
  5. *Logging Output*: Menyimpan rangkuman keputusan statistik ke file `.txt`.

---

### E. Pengujian Latensi Inferensi Produksi
* **Berkas:** `07_Benchmark_Inferensi/tes_inferensi.ipynb`
* **Kegunaan:** Mengukur latensi komputasi *end-to-end* dalam skenario inferensi waktu nyata (*real-time inference*) untuk memastikan sistem mampu memenuhi SLA latensi produksi.
* **Struktur Kode:**
  1. *Artifact Loading*: Memuat model Drain3, tokenizer, embedder transformer, dan bobot VAE yang sudah terlatih.
  2. *Pipeline Profiling*: Mengukur waktu eksekusi breakdown per tahap:
     * Tahap 1: Parsing log mentah ke template (Drain3).
     * Tahap 2: Transformasi teks ke embedding (*Transformer Forward Pass*).
     * Tahap 3: Rekonstruksi & komputasi loss ($\beta$-VAE).
     * Tahap 4: Keputusan anomali (*EVT Threshold Checking*).
  3. *Batch Size Sensitivity*: Analisis throughput dan latensi per sampel pada variasi ukuran batch (1, 8, 16, 32, 64, 128, 256).
  4. *SLA Verdict*: Evaluasi latensi persentil ke-95 ($p_{95}$) terhadap ambang toleransi produksi dan ekspor visualisasi ke `inference_latency_benchmark.png`.

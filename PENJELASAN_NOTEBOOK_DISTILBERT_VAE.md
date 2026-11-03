# Penjelasan Kode Notebook Eksperimen DistilBERT + $\beta$-VAE

Dokumen ini menjelaskan secara padat dan terstruktur alur serta fungsi tiap bagian kode pada notebook eksperimen utama deteksi anomali log (**`percobaan_3_hdfs_distilbert.ipynb`** dan **`percobaan_4_bgl_distilbert.ipynb`**).

---

### 1. Instalasi Dependensi
* **Tindakan:** Mengunduh dan memasang pustaka eksternal yang dibutuhkan di lingkungan eksekusi (Kaggle/Colab/Lokal).
* **Komponen Utama:** Pemasangan `drain3` (log parser), `transformers` & `huggingface_hub` (model NLP), `optuna` (hyperparameter tuning), serta dependensi komputasi ilmiah (`torch`, `scikit-learn`, `scipy`).

---

### 2. Import Library & Cek Lingkungan
* **Tindakan:** Memuat modul Python standar, pustaka data science, dan memeriksa ketersediaan akselerasi perangkat keras.
* **Komponen Utama:** Pemeriksaan versi PyTorch, ketersediaan GPU via `torch.cuda.is_available()`, identifikasi tipe GPU (misal CUDA T4/P100), dan penentuan seed global untuk reproduksibilitas awal.

---

### 3. Konfigurasi Eksperimen (`CONFIG`)
* **Tindakan:** Menetapkan kamus konfigurasi terpusat yang mengatur seluruh parameter eksperimen.
* **Komponen Utama:** Pendeteksian lingkungan (`IS_KAGGLE`), penentuan nama dataset (`HDFS` atau `BGL`), ukuran jendela sekuens (*sliding window / block-id*), model transformer (`distilbert-base-uncased`), rasio pembagian data (*train-val-test*), batas epoch, serta target *False Positive Rate* (FPR) untuk EVT.

---

### 4. Parsing Log Mentah dengan Drain3
* **Tindakan:** Mengubah baris teks log mentah yang tidak terstruktur menjadi template log terstruktur secara *streaming*.
* **Komponen Utama:** Inisialisasi `TemplateMiner` dengan aturan regex khusus dataset untuk menyaring variabel dinamis (seperti IP address, block ID, ID numerik), menghasilkan ID template dan teks template bersih.

---

### 5. Pembentukan Sekuens Log
* **Tindakan:** Mengelompokkan aliran log individual menjadi satu kesatuan sekuens logis.
* **Komponen Utama:**
  * **HDFS:** Dikelompokkan berdasarkan identifier blok unik (`block_id`).
  * **BGL:** Dikelompokkan menggunakan *sliding window* berbasis jumlah baris (misal jendela 20 log).
  * Penggabungan teks template dalam tiap sekuens dan pelabelan biner (0 = Normal, 1 = Anomali).

---

### 6. Ekstraksi Embedding Semantik (DistilBERT + Mean Pooling)
* **Tindakan:** Mengonversi sekuens teks template menjadi vektor representasi berdimensi tinggi ($d = 768$).
* **Komponen Utama:** Kelas `DistilBERTEmbedder` yang memproses token sekuens melalui transformer *forward pass*, menerapkan *attention mask*, dan menghitung rata-rata bobot representasi (*mean pooling*). Dilengkapi mekanisme penyimpanan/pemuatan cache array numpy (`.npy`) agar tidak perlu ekstraksi ulang.

---

### 7. Arsitektur Model $\beta$-VAE
* **Tindakan:** Mendefinisikan arsitektur *deep generative model* PyTorch untuk mempelajari representasi pola log normal.
* **Komponen Utama:**
  * **`VAEEncoder`:** Jaringan feed-forward dengan *Layer Normalization/Dropout* yang memproyeksikan vektor embedding input ke parameter distribusi laten rata-rata ($\mu$) dan log-varians ($\log \sigma^2$).
  * **Reparameterization Trick:** Mengambil sampel vektor laten $z = \mu + \epsilon \odot \sigma$ dengan $\epsilon \sim \mathcal{N}(0, I)$ agar gradien backpropagation tetap dapat mengalir.
  * **`VAEDecoder`:** Merekonstruksi vektor laten $z$ kembali ke dimensi embedding awal ($d = 768$).
  * **`vae_loss`:** Menghitung total loss yang terdiri dari *Reconstruction Loss* (MSE) + regularisasi *Kullback-Leibler (KL) Divergence* yang dikalikan faktor bobot $\beta$.

---

### 8. Pembagian Data (Train / Val / Test) & DataLoader
* **Tindakan:** Menyiapkan partisi data untuk skenario *unsupervised / semi-supervised anomaly detection*.
* **Komponen Utama:**
  * Data latih (*training set*) **hanya** menggunakan data sekuens normal untuk mengajarkan model pola operasional sistem yang sehat.
  * Normalisasi fitur menggunakan `StandardScaler` (fit pada train normal, transform pada val dan test).
  * Pembuatan PyTorch `Dataset` dan `DataLoader` dengan shuffling untuk batching data.

---

### 9. Pencarian Hyperparameter (Optuna)
* **Tindakan:** Mengoptimalkan kombinasi hyperparameter model VAE secara otomatis dan terukur.
* **Komponen Utama:** Menjalankan 30 trial evaluasi menggunakan kerangka kerja Optuna. Ruang pencarian mencakup: `latent_dim` (dimensi ruang laten), `batch_size`, koefisien $\beta$, dan laju pembelajaran (*learning rate*). Fungsi objektif meminimalkan loss rekonstruksi data validasi.

---

### 10. Training VAE Final
* **Tindakan:** Melatih model VAE akhir menggunakan konfigurasi hyperparameter terbaik yang ditemukan oleh Optuna.
* **Komponen Utama:** Pelatihan epoch penuh pada seluruh data latih normal dengan optimizer Adam/AdamW, mekanisme *early stopping* berbasis validasi loss, penyimpanan riwayat loss (*learning curve*), dan serialisasi bobot model terbaik (`vae_best.pt`).

---

### 11. Deteksi Anomali & Ambang Batas EVT / GPD
* **Tindakan:** Menentukan kriteria pemisah antara data normal dan anomali berdasarkan galat rekonstruksi.
* **Komponen Utama:**
  * Penghitungan *Reconstruction Error* (MSE) antara embedding asli dan hasil rekonstruksi decoder VAE.
  * Penerapan teori nilai ekstrem (**Extreme Value Theory - EVT**) melalui pendekatan *Peaks-Over-Threshold* (POT). Ekor distribusi error difit ke *Generalized Pareto Distribution* (GPD) untuk mendapatkan nilai *threshold* anomali dinamis yang adaptif pada target FPR tertentu.
  * Pembuatan histogram perbandingan sebaran loss antara sampel normal dan anomali.

---

### 12. Evaluasi Komparatif & Metrik Kinerja
* **Tindakan:** Mengukur dan membandingkan performa deteksi model secara kuantitatif.
* **Komponen Utama:**
  * Pengujian performa pada split Train, Validasi, dan Uji.
  * Komparasi metode penentuan ambang batas: **EVT/GPD** dibandingkan dengan ambang batas persentil empiris (**p90, p95, p99**).
  * Perhitungan metrik evaluasi: Precision, Recall, F1-Score, ROC-AUC, PR-AUC, serta pembuatan visualisasi Matriks Konfusi (*Confusion Matrix*).

---

### 13. Evaluasi Clustering & Proyeksi Ruang Laten
* **Tindakan:** Menganalisis sifat keterpisahan (*separability*) representasi laten model terhadap kondisi anomali.
* **Komponen Utama:** Ekstraksi vektor laten $z$ untuk sampel normal dan anomali, reduksi dimensi non-linear ke 2D menggunakan **t-SNE**, dan visualisasi scatter plot untuk membuktikan bahwa ruang laten membentuk klaster normal yang rapat dan memisahkan anomali.

---

### 14. Penyimpanan Artefak Eksperimen
* **Tindakan:** Mendokumentasikan dan mengekspor seluruh artefak output eksperimen secara permanen.
* **Komponen Utama:** Penyimpanan ringkasan metrik ke berkas `summary.json`, tabel metrik CSV, log durasi inferensi/komputasi, ekspor seluruh grafik (ROC/PR curve, learning curve, t-SNE plot), serta pengarsipan berkas hasil pelatihan.

---

### 15. Inferensi End-to-End pada Data Baru
* **Tindakan:** Menyediakan fungsi modular untuk menguji kemampuan model memprediksi aliran log mentah baru secara langsung.
* **Komponen Utama:** Fungsi pipeline inferensi tunggal: menerima teks log mentah $\rightarrow$ parsing template via Drain3 $\rightarrow$ ekstraksi embedding DistilBERT $\rightarrow$ normalisasi $\rightarrow$ rekonstruksi $\beta$-VAE $\rightarrow$ evaluasi terhadap threshold EVT $\rightarrow$ menghasilkan keputusan (*Normal* / *Anomali*).

import pandas as pd
import scipy.stats as stats
import os

# Load data
file_path = "c:/Users/Fajar M/Downloads/Output/hasil_rangkuman_5seed.csv"
df = pd.read_csv(file_path)

output_folder = "c:/Users/Fajar M/Downloads/Output/Uji_Statistik"
output_file = os.path.join(output_folder, "hasil_uji_statistik.txt")

metrics = [
    '[Pct-p90-Test]  f1',
    '[Pct-p95-Test]  f1',
    '[Pct-p99-Test]  f1',
    '[EVT-Test]  f1',
    'train_total_sec',
    'ari',
    'nmi',
    'silhouette'
]

datasets = df['Dataset'].unique()

with open(output_file, 'w') as f:
    f.write("=== HASIL UJI STATISTIK DAN HIPOTESIS (Paired T-Test) ===\n\n")
    
    for dataset in datasets:
        f.write(f"--- Dataset: {dataset} ---\n")
        df_ds = df[df['Dataset'] == dataset]
        
        embedders = df_ds['Embedder'].unique()
        if len(embedders) == 2:
            emb1, emb2 = embedders[0], embedders[1]
            f.write(f"Membandingkan {emb1} vs {emb2} dengan 5 seed yang sama.\n\n")
            
            for metric in metrics:
                if metric in df.columns:
                    data1 = df_ds[df_ds['Embedder'] == emb1].sort_values(by='Seed')[metric].values
                    data2 = df_ds[df_ds['Embedder'] == emb2].sort_values(by='Seed')[metric].values
                    
                    if len(data1) == len(data2) and len(data1) > 1:
                        # Paired T-test
                        stat, p_val = stats.ttest_rel(data1, data2)
                        
                        f.write(f"Metrik: {metric}\n")
                        f.write(f"  Rata-rata {emb1}: {data1.mean():.4f}\n")
                        f.write(f"  Rata-rata {emb2}: {data2.mean():.4f}\n")
                        f.write(f"  T-statistic: {stat:.4f}, p-value: {p_val:.4f}\n")
                        
                        alpha = 0.05
                        if p_val < alpha:
                            f.write(f"  Kesimpulan: Tolak H0. Terdapat perbedaan signifikan antara {emb1} dan {emb2} (p < {alpha}).\n")
                        else:
                            f.write(f"  Kesimpulan: Gagal Tolak H0. Tidak terdapat perbedaan signifikan antara {emb1} dan {emb2} (p >= {alpha}).\n")
                        f.write("\n")
        else:
            f.write(f"Tidak dapat membandingkan untuk dataset {dataset} karena embedder unik bukan 2.\n")
        f.write("="*50 + "\n\n")

print(f"Selesai! Hasil disimpan di {output_file}")

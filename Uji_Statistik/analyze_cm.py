import pandas as pd
import scipy.stats as stats
import os

file_path = "c:/Users/Fajar M/Downloads/Output/hasil_rangkuman_5seed.csv"
df = pd.read_csv(file_path)

output_folder = "c:/Users/Fajar M/Downloads/Output/Uji_Statistik"
output_file = os.path.join(output_folder, "hasil_uji_statistik_cm.txt")

metrics = [
    '[EVT-Test]  accuracy',
    '[EVT-Test]  precision',
    '[EVT-Test]  recall',
    '[EVT-Test]  f1'
]

datasets = df['Dataset'].unique()

with open(output_file, 'w') as f:
    for dataset in datasets:
        f.write(f"Dataset: {dataset}\n")
        df_ds = df[df['Dataset'] == dataset]
        embedders = df_ds['Embedder'].unique()
        emb1, emb2 = embedders[0], embedders[1]
        
        for metric in metrics:
            data1 = df_ds[df_ds['Embedder'] == emb1].sort_values(by='Seed')[metric].values
            data2 = df_ds[df_ds['Embedder'] == emb2].sort_values(by='Seed')[metric].values
            stat, p_val = stats.ttest_rel(data1, data2)
            f.write(f"{metric} | {emb1}: {data1.mean():.4f} | {emb2}: {data2.mean():.4f} | p-value: {p_val:.4f}\n")
        f.write("\n")

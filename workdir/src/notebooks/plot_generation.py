import pandas as pd
import matplotlib.subplots as plt
import seaborn as sns

# Tus datos estructurados
data = {
    'Dataset': ['Rimone', 'Rimone', 'Rimone', 'Refuge', 'Refuge', 'Refuge', 'Refuge', 'Refuge', 'Refuge'],
    'Attack': ['FGSM', 'BIM', 'PGD', 'FGSM', 'BIM', 'PGD', 'PGD', 'PGD', 'PGD'],
    'Epsilon': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.05, 0.02],
    'Precision': [8.514, 8.514, 8.514, 7.546, 7.546, 7.546, 7.546, 7.546, 7.546],
    'Precision_after': [7.256, 2.878, 3.130, 5.060, 2.576, 2.685, 3.814, 5.155, 6.515]
}
df = pd.DataFrame(data)

# --- GRÁFICA 1: Barras para Epsilon 0.2 ---
sns.set_theme(style="whitegrid")
df_eps02 = df[df['Epsilon'] == 0.2]

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35
x = range(len(df_eps02))

ax.bar([i - width/2 for i in x], df_eps02['Precision'], width, label='Precisión Original', color='skyblue')
ax.bar([i + width/2 for i in x], df_eps02['Precision_after'], width, label='Precisión tras el Ataque', color='salmon')

ax.set_ylabel('Precisión')
ax.set_title('Impacto de Diferentes Ataques con ε=0.2', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f"{d}\n({a})" for d, a in zip(df_eps02['Dataset'], df_eps02['Attack'])])
ax.legend()
plt.show()

# --- GRÁFICA 2: Líneas para Epsilon variante en Refuge (PGD) ---
df_refuge_pgd = df[(df['Dataset'] == 'Refuge') & (df['Attack'] == 'PGD')].sort_values('Epsilon')

plt.figure(figsize=(8, 5))
plt.plot(df_refuge_pgd['Epsilon'], df_refuge_pgd['Precision_after'], marker='o', linestyle='-', color='red', linewidth=2, markersize=8, label='Precisión tras Ataque PGD')
plt.axhline(y=df_refuge_pgd['Precision'].iloc[0], color='blue', linestyle='--', label='Precisión Original')
plt.xlabel('Epsilon (ε)', fontsize=12)
plt.ylabel('Precisión', fontsize=12)
plt.title('Efecto del valor de Epsilon en la Precisión\n(Dataset Refuge, Ataque PGD)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
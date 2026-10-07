import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42  # editable text in PDF (no Type 3 fonts)
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

df = pd.read_csv('results/dim_angle_vs_n_dense500.csv')

fig, ax = plt.subplots(figsize=(6, 4))

ax.fill_between(df['n'], df['min'], df['max'], color='#2a78d6', alpha=0.18, label='min-max range (200 draws)')
ax.plot(df['n'], df['min'], color='#2a78d6', linewidth=1.0, linestyle='--', alpha=0.8)
ax.plot(df['n'], df['max'], color='#2a78d6', linewidth=1.0, linestyle='--', alpha=0.8)
ax.plot(df['n'], df['mean'], color='#2a78d6', linewidth=2.0, label='mean angle')

ax.axhline(10, color='gray', linewidth=1.0, linestyle=':', alpha=0.7)
ax.text(df['n'].max(), 10, ' 10° ref.', va='center', ha='left', fontsize=8, color='gray')

ax.set_xlabel('sample size $n$ (n positive + n negative statements)')
ax.set_ylabel('angle to full-data DiM direction (degrees)')
ax.set_title('DiM direction angle vs. subsample size\n(Llama3-8B-Instruct, layer 13 / head 28)', fontsize=11)
ax.set_xlim(1, 500)
ax.set_ylim(0, 110)
ax.legend(frameon=False, fontsize=9, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.25, linewidth=0.5)

fig.tight_layout()
out_path = 'figures/dim_angle_vs_n.pdf'
fig.savefig(out_path)
print(f"Saved to {out_path}")

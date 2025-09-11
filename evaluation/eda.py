import pandas as pd
import matplotlib.pyplot
import seaborn as sns
from pathlib import Path

images_dir = Path.cwd() / "dataset/BTXRD/images"
metadata_path = Path.cwd() / "dataset/BTXRD/dataset.xlsx"

df = pd.read_excel(metadata_path, engine='openpyxl')

matplotlib.pyplot.figure(figsize=(10, 7))
sns.set_style("whitegrid")
ax = sns.countplot(x='tumor', data=df, hue='tumor', palette='viridis', legend=False)
ax.set_title('Tumor vs Normal Distribution', fontsize=16, pad=20)
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Number of Images', fontsize=12)
ax.set_xticklabels(['Normal Images', 'Tumor Images'], fontsize=11)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
matplotlib.pyplot.show()

anatomy_cols = ['hand', 'ulna', 'radius', 'humerus', 'foot', 'tibia', 'fibula', 'femur', 'hip bone']
anatomy_counts = df[anatomy_cols].sum().sort_values(ascending=False)
matplotlib.pyplot.figure(figsize=(12, 8))
sns.barplot(x=anatomy_counts.values, y=anatomy_counts.index, hue=anatomy_counts.index, palette='crest', legend=False)
matplotlib.pyplot.title('Distribution by Anatomy', fontsize=16, pad=20)
matplotlib.pyplot.xlabel('Number of Images', fontsize=12)
matplotlib.pyplot.ylabel('Anatomy Class', fontsize=12)
matplotlib.pyplot.show()


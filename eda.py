import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#%% Load Dataset

DATASET_EXCEL = "dataset.xlsx"

project_dir = os.getcwd()
data_dir = os.path.join(project_dir, "dataset", "BTXRD")
images_dir = os.path.join(data_dir, "images")
metadata_path = os.path.join(data_dir, DATASET_EXCEL)

#Read Excel
df = pd.read_excel(metadata_path, engine='openpyxl')

#%% Tumor vs Normal Distribution

print("Tumor vs Normal Distribution")
plt.figure(figsize=(10, 7))
sns.set_style("whitegrid")
ax = sns.countplot(x='tumor', data=df, palette='viridis')
ax.set_title('Tumor vs Normal Distribution', fontsize=16, pad=20)
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Number of Images', fontsize=12)
ax.set_xticklabels(['Normal Images', 'Tumor Images'], fontsize=11)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.show()

#%% Anatomy Class Distribution
print("Anatomy Class Distribution")
anatomy_cols = ['hand', 'ulna', 'radius', 'humerus', 'foot', 'tibia', 'fibula', 'femur', 'hip bone']
anatomy_counts = df[anatomy_cols].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=anatomy_counts.values, y=anatomy_counts.index, palette='crest')
plt.title('Tumor vs Normal Distribution', fontsize=16, pad=20)
plt.xlabel('Number of Images', fontsize=12)
plt.ylabel('Anatomy Class', fontsize=12)
plt.show()

#%% Image Dimensions Distribution
print("Image Dimensions Distribution")
width = []
height = []
for image_filename in df['image_id']:
    image_path = os.path.join(images_dir, image_filename)
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            h, w, _ = img.shape
            height.append(h)
            width.append(w)

plt.figure(figsize=(16, 7))
plt.subplot(1, 2, 1)
sns.histplot(width, bins=50, kde=True, color='royalblue')
plt.title('Image Width Distribution', fontsize=14)
plt.xlabel('Width (pixels)')
plt.ylabel('Frequency', fontsize=12)
plt.subplot(1, 2, 2)
sns.histplot(height, bins=50, kde=True, color='forestgreen')
plt.title('Image Height Distribution', fontsize=14)
plt.xlabel('Height (pixels)')
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout(pad=3.0)
plt.show()

#%% t-Distributed Stochastic Neighbor

image_features = []
labels = []
    
# Dataset Loop
for row_index, row in df.iterrows():
        image_path = os.path.join(images_dir, row['image_id'])
        if os.path.exists(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))
                image_features.append(img_resized.flatten())
                labels.append(row['tumor'])

X = np.array(image_features)
y = np.array(labels)

# Standarisation
X_scaled = StandardScaler().fit_transform(X)

# Applying Principal Component Analysis
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# t-Distributed Stochastic Neighbor
tsne = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=5000, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Create a DataFrame for plotting
tsne_dataframe = pd.DataFrame(data=X_tsne, columns=['tsne-1', 'tsne-2'])
tsne_dataframe['label'] = y

plt.figure(figsize=(12, 10))
sns.scatterplot(
    x="tsne-1", y="tsne-2",
    hue="label",
    palette=sns.color_palette("hls", 2),
    data=tsne_dataframe,
    legend="full",
    alpha=0.6
)
plt.title('t-Distributed Stochastic Neighbor', fontsize=16, pad=20)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
plt.legend(legend_handles, ['Normal', 'Tumor'], title='Class')

plt.show()


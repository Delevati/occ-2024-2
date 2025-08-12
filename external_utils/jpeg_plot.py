import os
import matplotlib.pyplot as plt
from PIL import Image

# Caminho da pasta onde estão os JPEGs de MG
mg_folder = "/Volumes/luryand/selecoes_final"
mg_imgs = [os.path.join(mg_folder, f) for f in os.listdir(mg_folder) if f.startswith("MG_") and f.endswith(".jpg")]

# Ordena numericamente pelo número do mosaico (assumindo nome tipo MG_mosaic_1.jpg)
mg_imgs.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].replace('.jpg','').replace('mosaic','').replace('_','')))

# Garante até 6 imagens
mg_imgs = mg_imgs[:6]
while len(mg_imgs) < 6:
    mg_imgs.append(None)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
idx = 0
for row in range(2):
    for col in range(3):
        ax = axes[row][col]
        if mg_imgs[idx] is not None:
            with Image.open(mg_imgs[idx]) as img:
                img = img.convert("RGB")
                img.thumbnail((400, 400), Image.LANCZOS)
                ax.imshow(img)
                # ax.set_title(os.path.basename(mg_imgs[idx]))
        ax.axis('off')
        idx += 1

plt.tight_layout()
plt.savefig(os.path.join(mg_folder, "mosaic_mg_plot.jpg"), dpi=150)
plt.close()
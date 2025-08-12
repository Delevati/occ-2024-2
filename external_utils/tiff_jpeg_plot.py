import os
from PIL import Image
import matplotlib.pyplot as plt
import glob

Image.MAX_IMAGE_PIXELS = None

def tiff_to_jpeg(input_path, output_path, quality=85):
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=quality, optimize=True)

def plot_all_mosaics(jpeg_dict, output_plot, max_size=(400, 400)):
    # Ordena regiões, RS por último
    regions = sorted([r for r in jpeg_dict if r != "RS"]) + (["RS"] if "RS" in jpeg_dict else [])
    total_rows = sum((len(jpeg_dict[r]) + 5) // 6 for r in regions)
    fig, axes = plt.subplots(total_rows, 6, figsize=(6*4, total_rows*4))
    if total_rows == 1:
        axes = [axes]
    idx = 0
    for region in regions:
        imgs = jpeg_dict[region]
        row_imgs = [imgs[i:i+6] for i in range(0, len(imgs), 6)]
        for row in row_imgs:
            for col in range(6):
                ax = axes[idx][col] if total_rows > 1 else axes[col]
                if col < len(row):
                    img_path = row[col]
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        img.thumbnail(max_size, Image.LANCZOS)
                        ax.imshow(img)
                        ax.set_title(f"{region} - {os.path.basename(img_path).split('_')[-1].replace('.jpg','')}")
                else:
                    ax.axis('off')
                ax.axis('off')
            idx += 1
        if region == "RS" and len(imgs) == 5:
            for col in range(5, 6):
                ax = axes[idx-1][col] if total_rows > 1 else axes[col]
                ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()

# Caminho dos arquivos
base_path = "/Volumes/luryand/mosaicos_selecionados"
tiff_files = glob.glob(os.path.join(base_path, "*_mosaic_*.tif"))

# Agrupa por região
jpeg_dict = {}
for tif in tiff_files:
    region = os.path.basename(tif).split('_')[0]
    jpeg_path = tif.replace(".tif", ".jpg")
    tiff_to_jpeg(tif, jpeg_path, quality=150)
    jpeg_dict.setdefault(region, []).append(jpeg_path)

# Ordena os mosaicos de cada região pelo nome (ex: mosaic_1, mosaic_2, ...)
for region in jpeg_dict:
    jpeg_dict[region].sort(key=lambda x: int(os.path.basename(x).split('_')[-1].replace('.jpg','').replace('mosaic_','')))

# Plota todos os mosaicos em um único plot
# plot_all_mosaics(jpeg_dict, os.path.join(base_path, "mosaic_golden_plot.jpg"), max_size=(400, 400))

if "MG" in jpeg_dict:
    mg_dict = {"MG": jpeg_dict["MG"]}
    plot_all_mosaics(mg_dict, os.path.join(base_path, "mosaic_mg_plot.jpg"), max_size=(400, 400))
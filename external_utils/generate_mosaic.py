import os
import json
import glob
import subprocess
import rasterio
from rasterio.shutil import copy
import tempfile
import zipfile
from tqdm import tqdm

IMAGES_BASE_DIR = "/Volumes/luryand/nova_busca"
OUTPUT_DIR = "/Volumes/luryand/mosaicos_selecionados"

def find_tci_10m_file(safe_dir):
    """Procura o arquivo TCI 10m dentro da estrutura .SAFE"""
    pattern = os.path.join(
        safe_dir, "GRANULE", "*", "IMG_DATA", "R10m", "*_TCI_10m.jp2"
    )
    files = glob.glob(pattern)
    return files[0] if files else None

def convert_jp2_to_tif(jp2_path, tif_path):
    """Converte JP2 para TIFF usando rasterio"""
    with rasterio.open(jp2_path) as src:
        copy(src, tif_path, driver='GTiff')

def mosaic_images(image_list, shapefile, output_path):
    temp_vrt = output_path.replace('.tif', '_temp.vrt')
    subprocess.run([
        "gdalbuildvrt", temp_vrt, *image_list
    ], check=True)
    subprocess.run([
        "gdalwarp", "-overwrite", "-cutline", shapefile, "-crop_to_cutline",
        "-of", "GTiff", "-dstnodata", "255", temp_vrt, output_path
    ], check=True)
    os.remove(temp_vrt)

def process_area(json_path, shapefile_path):
    area_name = os.path.splitext(os.path.basename(json_path))[0].split('_')[0]

    with open(json_path, "r") as f:
        log_data = json.load(f)
    # print(f"Log lido para {area_name}: {log_data}")

    with tempfile.TemporaryDirectory(dir="/Volumes/luryand/tmp") as tmpdir:
        for group in log_data.get("mosaic_groups", []):
            group_id = group.get("group_id", "unknown")
            band_files = []
            for img_name in group.get("images", []):
                zip_path = os.path.join(IMAGES_BASE_DIR,area_name,img_name)
                if not os.path.exists(zip_path):
                    print(f"AVISO: ZIP não encontrado: {zip_path}")
                    continue
                print(f"Procurando ZIP: {zip_path}")

                extract_dir = os.path.join(tmpdir, img_name.replace('.SAFE.zip', ''))
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                safe_dirs = [d for d in os.listdir(extract_dir) if d.endswith('.SAFE')]
                if not safe_dirs:
                    print(f"Nenhuma pasta .SAFE encontrada em {zip_path}")
                    continue
                safe_dir = os.path.join(extract_dir, safe_dirs[0])
                tci_file = find_tci_10m_file(safe_dir)
                print(f"Arquivo TCI buscado: {tci_file}")
                if tci_file:
                    tif_file = tci_file.replace('.jp2', '.tif')
                    convert_jp2_to_tif(tci_file, tif_file)
                    band_files.append(tif_file)
                else:
                    print(f"TCI não encontrado em: {safe_dir}")

            if not band_files:
                print(f"Nenhum arquivo TCI encontrado para {area_name} - {group_id}")
                continue

            output_path = os.path.join(OUTPUT_DIR, f"{area_name}_mosaic_{group_id}.tif")
            mosaic_images(band_files, shapefile_path, output_path)
            print(f"Mosaico gerado para {area_name} - {group_id}: {output_path}")

def main():
    json_dir = "/Users/luryand/Documents/encode-image/coverage_otimization/code/output_log_cplex"
    shapefile_base = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture"
    json_files = glob.glob(os.path.join(json_dir, "*.json"))

    for json_path in tqdm(json_files, desc="Áreas (JSONs)", unit="área"):
        area_name = os.path.splitext(os.path.basename(json_path))[0].split('_')[0]
        shapefile_dir = os.path.join(shapefile_base, area_name)
        shapefiles = glob.glob(os.path.join(shapefile_dir, "*.shp"))
        if not shapefiles:
            print(f"AVISO: Nenhum shapefile encontrado para {area_name} em {shapefile_dir}")
            continue
        shapefile_path = shapefiles[0]
        print(f"\nProcessando {json_path} com shapefile {shapefile_path}")
        process_area(json_path, shapefile_path)

if __name__ == "__main__":
    main()
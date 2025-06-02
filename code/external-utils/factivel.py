import os
import json
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import geopandas as gpd
from tqdm import tqdm
import copy
import numpy as np
from rasterio.coords import BoundingBox
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from shapely.geometry import mapping

CPLEX_JSON = "/Volumes/luryand/coverage_otimization_pe-pi-ce/results/cplex_selected_mosaic_groups.json"
METADATA_JSON = "/Volumes/luryand/coverage_otimization_pe-pi-ce/metadata/all_processed_images_log.json"
AOI_SHP_PATH = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/PI-PE-CE/ucs_pe-pi-ce_31984.shp"
EXCLUDE = {"mosaic_1", "mosaic_6"}
OUTPUT_DIR = "/Volumes/luryand/cplex_mosaics_clean"
TARGET_CRS = "EPSG:31984"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fix_bounds_format(metadata_lookup):
    fixed_metadata = copy.deepcopy(metadata_lookup)
    for filename, meta in fixed_metadata.items():
        if 'bounds' in meta and isinstance(meta['bounds'], list):
            bounds_list = meta['bounds']
            if len(bounds_list) == 4:
                try:
                    meta['bounds'] = BoundingBox(
                        left=bounds_list[0],
                        bottom=bounds_list[1],
                        right=bounds_list[2],
                        top=bounds_list[3]
                    )
                except Exception:
                    pass
        elif 'bounds' in meta and isinstance(meta['bounds'], dict):
            bounds_dict = meta['bounds']
            if all(k in bounds_dict for k in ['left', 'bottom', 'right', 'top']):
                try:
                    meta['bounds'] = BoundingBox(**bounds_dict)
                except Exception:
                    pass
    return fixed_metadata

def plot_mosaic_single(mosaic, fixed_metadata, aoi_gdf_union, output_path):
    component_images = mosaic.get('images', [])
    verified_images = []
    for img_name in component_images:
        if img_name in fixed_metadata:
            img_meta = fixed_metadata[img_name]
            tci_path = img_meta.get('tci_path') or img_meta.get('temp_tci_path')
            if tci_path and os.path.exists(tci_path):
                verified_images.append(img_name)
    if not verified_images:
        logging.warning(f"Nenhuma imagem válida para o mosaico {mosaic.get('group_id')}")
        return

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    aoi_geometry = aoi_gdf_union.geometry.values[0]
    aoi_geojson = [mapping(aoi_geometry)]

    # Preenche a AOI com preto antes de plotar as imagens
    if hasattr(aoi_geometry, "geoms"):
        # MultiPolygon
        for poly in aoi_geometry.geoms:
            ax.add_patch(plt.Polygon(np.array(poly.exterior.coords), closed=True, facecolor='black', edgecolor=None, zorder=0))
    else:
        # Polygon
        ax.add_patch(plt.Polygon(np.array(aoi_geometry.exterior.coords), closed=True, facecolor='black', edgecolor=None, zorder=0))

    # Plote cada imagem mascarada diretamente no eixo
    for img_name in verified_images:
        try:
            with rasterio.open(fixed_metadata[img_name]['tci_path']) as src:
                out_image, out_transform = mask(
                    src, aoi_geojson, crop=True, nodata=0, filled=True, all_touched=True
                )
                show(out_image, transform=out_transform, ax=ax)
        except Exception as e:
            logging.warning(f"Erro ao plotar {img_name}: {e}")

    # AOI boundary
    aoi_gdf_union.boundary.plot(ax=ax, color='red', linewidth=1.0, linestyle='-', zorder=100)

    # Eixos formatados curtos
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}"))
    ax.set_frame_on(False)
    data_str = mosaic.get("time_window_start", "")
    if data_str:
        try:
            data_fmt = data_str[:10]
        except Exception:
            data_fmt = data_str
        ax.set_title(data_fmt, fontsize=15)
    else:
        ax.set_title("", fontsize=15)
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    logging.info(f"Salvo: {output_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    aoi_gdf = gpd.read_file(AOI_SHP_PATH)
    if aoi_gdf.crs.to_epsg() != 31984:
        aoi_gdf = aoi_gdf.to_crs(TARGET_CRS)
    aoi_geometry_union = aoi_gdf.union_all()
    aoi_gdf_union = gpd.GeoDataFrame(geometry=[aoi_geometry_union], crs=TARGET_CRS)
    with open(CPLEX_JSON, "r") as f:
        mosaics = json.load(f)
    with open(METADATA_JSON, "r") as f:
        all_metadata = json.load(f)
    metadata_lookup = {img['filename']: img for img in all_metadata if img.get('filename')}
    fixed_metadata = fix_bounds_format(metadata_lookup)
    mosaics_to_plot = [m for m in mosaics if m.get("group_id") not in EXCLUDE]
    mosaics_to_plot.sort(key=lambda x: x.get("time_window_start", ""))
    for mosaic in tqdm(mosaics_to_plot, desc="Plotando mosaicos"):
        group_id = mosaic.get("group_id", "mosaic")
        date_str = mosaic.get("time_window_start", "")[:10].replace("-", "")
        output_path = os.path.join(OUTPUT_DIR, f"{date_str}_{group_id}_simple.png")
        plot_mosaic_single(mosaic, fixed_metadata, aoi_gdf_union, output_path)

if __name__ == "__main__":
    main()
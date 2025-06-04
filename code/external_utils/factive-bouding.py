import os
import json
import sys
import matplotlib.pyplot as plt
import geopandas as gpd
import logging
import numpy as np
from matplotlib import rcParams
import matplotlib.patches as patches
import matplotlib.ticker as mticker
from shapely.geometry import mapping, box, Polygon
from shapely.ops import unary_union
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.coords import BoundingBox

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parâmetros
CPLEX_JSON = "/Volumes/luryand/coverage_otimization_pe-pi-ce/results/cplex_selected_mosaic_groups.json"
METADATA_JSON = "/Volumes/luryand/coverage_otimization_pe-pi-ce/metadata/all_processed_images_log.json"
AOI_SHP_PATH = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/PI-PE-CE/ucs_pe-pi-ce_31984.shp"
OUTPUT_DIR = "/Volumes/luryand/cplex_mosaics_boxes"

# CRS para usar em todos os dados (SIRGAS 2000 / UTM zone 24S)
TARGET_CRS = "EPSG:31984"

def fix_bounds_format(metadata_lookup):
    """Corrige o formato dos bounds no metadata."""
    fixed_metadata = {}
    
    for filename, meta in metadata_lookup.items():
        fixed_metadata[filename] = meta.copy()
        if 'bounds' in meta and isinstance(meta['bounds'], list):
            bounds_list = meta['bounds']
            if len(bounds_list) == 4:
                try:
                    fixed_metadata[filename]['bounds'] = BoundingBox(
                        left=bounds_list[0],
                        bottom=bounds_list[1],
                        right=bounds_list[2],
                        top=bounds_list[3]
                    )
                except Exception as e:
                    logging.error(f"Erro ao converter bounds para {filename}: {e}")
        elif 'bounds' in meta and isinstance(meta['bounds'], dict):
            bounds_dict = meta['bounds']
            if all(k in bounds_dict for k in ['left', 'bottom', 'right', 'top']):
                try:
                    fixed_metadata[filename]['bounds'] = BoundingBox(**bounds_dict)
                except Exception as e:
                    logging.error(f"Erro ao converter bounds dict para {filename}: {e}")
    
    return fixed_metadata

def plot_mosaic_with_boxes(mosaic_id, mosaics, metadata_lookup, aoi_gdf, output_path):
    """
    Plota um mosaico específico com caixas delimitadoras para cada imagem componente.
    
    Args:
        mosaic_id: ID do mosaico a ser plotado
        mosaics: Lista de todos os mosaicos
        metadata_lookup: Dicionário com metadados das imagens
        aoi_gdf: GeoDataFrame com a AOI
        output_path: Caminho para salvar a imagem
    """
    # Encontrar o mosaico específico
    target_mosaic = None
    for mosaic in mosaics:
        if mosaic.get("group_id") == mosaic_id:
            target_mosaic = mosaic
            break
    
    if not target_mosaic:
        logging.error(f"Mosaico {mosaic_id} não encontrado!")
        return False
    
    # Obter data e imagens do mosaico
    date_str = target_mosaic.get("time_window_start", "")
    short_date = date_str[:10] if date_str else ""
    component_images = target_mosaic.get('images', [])
    
    logging.info(f"Plotando mosaico {mosaic_id} com {len(component_images)} imagens e bounding boxes")
    
    # Configurações visuais
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    rcParams['font.size'] = 12
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    
    # Extrair geometria da AOI
    aoi_geometry = aoi_gdf.geometry.values[0]
    aoi_bounds = aoi_gdf.total_bounds
    aoi_geojson = [mapping(aoi_geometry)]
    
    # Calcular centro e limites da AOI
    aoi_center_x = (aoi_bounds[0] + aoi_bounds[2]) / 2
    aoi_center_y = (aoi_bounds[1] + aoi_bounds[3]) / 2
    aoi_width = aoi_bounds[2] - aoi_bounds[0]
    aoi_height = aoi_bounds[3] - aoi_bounds[1]
    margin_percent = 0.1
    plot_width = aoi_width * (1 + margin_percent)
    plot_height = aoi_height * (1 + margin_percent)
    plot_minx = aoi_center_x - plot_width / 2
    plot_miny = aoi_center_y - plot_height / 2
    plot_maxx = aoi_center_x + plot_width / 2
    plot_maxy = aoi_center_y + plot_height / 2
    
    # Lista para armazenar informações das imagens para bounding boxes
    image_boxes = []
    
    # Cores distintas para cada imagem
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plotar as imagens mascaradas pela AOI
    for idx, img_name in enumerate(component_images):
        if img_name not in metadata_lookup:
            continue
            
        img_meta = metadata_lookup[img_name]
        tci_path = img_meta.get('tci_path') or img_meta.get('temp_tci_path')
        
        if not tci_path or not os.path.exists(tci_path):
            continue
            
        logging.info(f"  Plotando {img_name} com bounding box")
        
        try:
            with rasterio.open(tci_path) as src:
                # Guardar informações sobre a imagem para bounding box
                file_bounds = None
                
                # Processar a imagem
                if src.crs.to_epsg() != 31984:
                    # Reprojetar para EPSG:31984
                    transform, width, height = calculate_default_transform(
                        src.crs, TARGET_CRS, src.width, src.height, *src.bounds
                    )
                    
                    # Criar array para a imagem reprojetada
                    reprojected = np.zeros((src.count, height, width), dtype=src.dtypes[0])
                    
                    # Aplicar a reprojeção
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=reprojected[i-1],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=TARGET_CRS,
                            resampling=Resampling.nearest
                        )
                    
                    # Calcular bounds para a imagem reprojetada
                    file_bounds = [
                        transform[2],                    # left
                        transform[5] + height * transform[4],  # bottom
                        transform[2] + width * transform[0],   # right
                        transform[5]                     # top
                    ]
                    
                    # Criar MemoryFile e aplicar máscara
                    mem_file = rasterio.MemoryFile()
                    with mem_file.open(
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=src.count,
                        dtype=src.dtypes[0],
                        crs=TARGET_CRS,
                        transform=transform
                    ) as mem_ds:
                        for i in range(src.count):
                            mem_ds.write(reprojected[i], i + 1)
                        
                        # Aplicar máscara da AOI
                        try:
                            out_image, out_transform = mask(mem_ds, aoi_geojson, crop=True, 
                                                         nodata=0, filled=True, all_touched=True)
                            show(out_image, transform=out_transform, ax=ax)
                        except Exception as mask_error:
                            logging.error(f"  Erro ao aplicar máscara AOI: {mask_error}")
                else:
                    # Se já está no CRS correto
                    file_bounds = [
                        src.bounds.left,
                        src.bounds.bottom,
                        src.bounds.right,
                        src.bounds.top
                    ]
                    
                    # Aplicar máscara diretamente
                    try:
                        out_image, out_transform = mask(src, aoi_geojson, crop=True, 
                                                      nodata=0, filled=True, all_touched=True)
                        show(out_image, transform=out_transform, ax=ax)
                    except Exception as mask_error:
                        logging.error(f"  Erro ao aplicar máscara AOI: {mask_error}")
                
                # Adicionar informação da imagem para o bounding box
                if file_bounds:
                    # Obter nome simplificado para o rótulo
                    simple_name = os.path.basename(img_name).split('_')[0]
                    color = colors[idx % len(colors)]
                    
                    image_boxes.append({
                        'bounds': file_bounds,
                        'name': simple_name,
                        'color': color,
                        'original_name': img_name
                    })
                
        except Exception as e:
            logging.error(f"  Erro ao plotar {img_name}: {e}")
    
    # Calcular a porcentagem de cobertura da AOI
    aoi_area = aoi_gdf.area.values[0]  # Área total da AOI
    
    # Criar polígonos a partir dos bounding boxes e calcular interseção com AOI
    coverage_polygons = []
    for img_info in image_boxes:
        bounds = img_info['bounds']
        left, bottom, right, top = bounds
        img_box = box(left, bottom, right, top)
        # Interseção com a AOI
        intersection = img_box.intersection(aoi_geometry)
        if not intersection.is_empty:
            coverage_polygons.append(intersection)
    
    # União de todos os polígonos (para não contar áreas sobrepostas múltiplas vezes)
    if coverage_polygons:
        total_coverage = unary_union(coverage_polygons)
        coverage_area = total_coverage.area
        coverage_percent = (coverage_area / aoi_area) * 100
    else:
        coverage_percent = 0.0
    
    # Plotar a AOI
    aoi_gdf.boundary.plot(ax=ax, color='#FF3333', linewidth=1.0, linestyle='-', zorder=100)
    
    # Plotar os bounding boxes das imagens SEM as etiquetas centrais
    for img_info in image_boxes:
        bounds = img_info['bounds']
        left, bottom, right, top = bounds
        width = right - left
        height = top - bottom
        
        # Criar retângulo para o bounding box
        rect = patches.Rectangle(
            (left, bottom), width, height,
            linewidth=2,
            edgecolor=img_info['color'],
            facecolor='none',
            linestyle='--',
            zorder=50
        )
        ax.add_patch(rect)
        
        # NÃO adicionar mais os rótulos S2B no centro
    
    # Definir os limites do plot
    ax.set_xlim(plot_minx, plot_maxx)
    ax.set_ylim(plot_miny, plot_maxy)
    
    # Configurar eixos
    ax.grid(False)
    ax.set_frame_on(True)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    
    # Adicionar título (data) com porcentagem de cobertura
    if short_date:
        try:
            year, month, day = short_date.split('-')
            formatted_date = f"{day}/{month}/{year}"
        except:
            formatted_date = short_date
        
        # Adicionar informação de cobertura ao título
        plt.title(f"{formatted_date} - Mosaico {mosaic_id} - Cobertura: {coverage_percent:.1f}%", 
                fontsize=16, fontweight='bold')

    # Aplicar tight_layout antes de salvar
    plt.tight_layout()

    # Salvar com alta qualidade
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2,
            transparent=False, facecolor='white')
    
    logging.info(f"Plot com bounding boxes salvo em: {output_path}")
    return True

def main():
    """Função principal para plotar mosaico específico com bounding boxes."""
    # Criar diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Verificar se foi fornecido um ID de mosaico
    if len(sys.argv) < 2:
        print("Uso: python plot_mosaic_with_boxes.py <mosaic_id>")
        
        # Listar mosaicos disponíveis
        with open(CPLEX_JSON, "r") as f:
            mosaics = json.load(f)
        
        print("\nMosaicos disponíveis:")
        for mosaic in mosaics:
            group_id = mosaic.get("group_id")
            date = mosaic.get("time_window_start", "")[:10]
            num_images = len(mosaic.get('images', []))
            print(f"  {group_id} - Data: {date} - {num_images} imagens")
        
        return
    
    mosaic_id = sys.argv[1]
    
    # Carregar a AOI
    aoi_gdf = gpd.read_file(AOI_SHP_PATH)
    if aoi_gdf.crs.to_epsg() != 31984:
        aoi_gdf = aoi_gdf.to_crs(TARGET_CRS)
    
    # Unir geometrias da AOI
    aoi_geometry_union = aoi_gdf.union_all ()
    aoi_gdf_union = gpd.GeoDataFrame(geometry=[aoi_geometry_union], crs=TARGET_CRS)
    
    # Carregar mosaicos e metadados
    with open(CPLEX_JSON, "r") as f:
        mosaics = json.load(f)
    
    with open(METADATA_JSON, "r") as f:
        all_metadata = json.load(f)
    
    # Criar lookup e corrigir bounds
    metadata_lookup = {img['filename']: img for img in all_metadata if img.get('filename')}
    fixed_metadata = fix_bounds_format(metadata_lookup)
    
    # Criar o plot
    output_path = os.path.join(OUTPUT_DIR, f"{mosaic_id}_with_boxes.png")
    plot_mosaic_with_boxes(mosaic_id, mosaics, fixed_metadata, aoi_gdf_union, output_path)

if __name__ == "__main__":
    main()
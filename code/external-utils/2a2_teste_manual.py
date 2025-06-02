import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.mask import mask
from pathlib import Path
import zipfile
import shutil
import math
import logging
import traceback
from shapely.geometry import shape, box, mapping
import geopandas as gpd
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# Diretório das imagens Sentinel-2
IMAGE_DIR = "/Volumes/luryand/nova_busca"

# Diretório temporário personalizado
TEMP_DIR = "/Volumes/luryand/temp_mosaics"
os.makedirs(TEMP_DIR, exist_ok=True)

# DEFINIÇÃO FIXA DOS ARQUIVOS - MUDE AQUI
JSON_FILE = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/PI-PE-CE/optimization_parameters-PI-PE-CE-precalc2.json"
SHP_FILE = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/PI-PE-CE/ucs_pe-pi-ce_31984.shp"

def extract_tci_from_zip(zip_path, temp_dir):
    """Extrai arquivo TCI de uma imagem Sentinel-2 ZIP."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            tci_files = [f for f in zip_ref.namelist() if 'TCI_10m.jp2' in f]
            if not tci_files:
                return None
            
            tci_file = tci_files[0]
            extract_path = os.path.join(temp_dir, os.path.basename(tci_file))
            
            with zip_ref.open(tci_file) as source, open(extract_path, 'wb') as target:
                shutil.copyfileobj(source, target)
            
            return extract_path
    except Exception as e:
        logging.error(f"Erro ao extrair TCI de {zip_path}: {e}")
        return None

def find_image_path(image_name):
    """Encontra o caminho da imagem em qualquer subdiretório do diretório base."""
    base_name = image_name.replace('.zip', '').replace('.SAFE', '')
    
    # Procurar em todos os subdiretórios
    for root, dirs, files in os.walk(IMAGE_DIR):
        # Tentar diferentes variações do nome
        for variant in [f"{base_name}.zip", f"{base_name}.SAFE.zip", f"{base_name}.SAFE"]:
            path = os.path.join(root, variant)
            if os.path.exists(path):
                return path
    
    logging.warning(f"Imagem não encontrada: {image_name}")
    return None

def get_tci_path(image_name, temp_dir):
    """Obtém o caminho do arquivo TCI para a imagem."""
    image_path = find_image_path(image_name)
    if not image_path:
        return None
    
    # Extrair TCI do ZIP ou encontrar no SAFE
    if image_path.endswith('.zip'):
        tci_path = extract_tci_from_zip(image_path, temp_dir)
    else:  # SAFE directory
        tci_files = list(Path(image_path).glob("**/TCI_10m.jp2"))
        tci_path = str(tci_files[0]) if tci_files else None
        
    if not tci_path:
        logging.warning(f"TCI não encontrado para {image_name}")
    
    return tci_path

def get_image_bbox(image_name, temp_dir):
    """Obtém o bounding box da imagem em coordenadas geográficas."""
    tci_path = get_tci_path(image_name, temp_dir)
    if not tci_path:
        return None
    
    try:
        with rasterio.open(tci_path) as src:
            # Obter os limites da imagem
            bounds = src.bounds
            
            # Converter para coordenadas WGS84 (lat/lon)
            bbox_wgs84 = transform_bounds(src.crs, 'EPSG:4326', 
                                         bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            # Criar um polígono com os limites
            bbox_poly = box(bbox_wgs84[0], bbox_wgs84[1], bbox_wgs84[2], bbox_wgs84[3])
            return bbox_poly
    except Exception as e:
        logging.error(f"Erro ao obter bbox para {image_name}: {e}")
        return None

def transform_aoi_to_raster_crs(aoi_gdf, raster_crs):
    """Transforma o GeoDataFrame da AOI para o CRS do raster."""
    if aoi_gdf is None:
        return None
    
    # Converter a AOI para o mesmo CRS da imagem
    try:
        aoi_transformed = aoi_gdf.to_crs(raster_crs)
        return aoi_transformed
    except Exception as e:
        logging.error(f"Erro ao transformar AOI para CRS {raster_crs}: {e}")
        return None

def calculate_mosaic_area(image_names, temp_dir, aoi_gdf=None):
    """Calcula a área coberta por todas as imagens do mosaico."""
    total_area = 0
    
    # Para cada imagem no mosaico
    for image_name in image_names:
        tci_path = get_tci_path(image_name, temp_dir)
        if not tci_path:
            continue
            
        try:
            with rasterio.open(tci_path) as src:
                # Transformar AOI para o CRS da imagem
                if aoi_gdf is not None:
                    aoi_transformed = transform_aoi_to_raster_crs(aoi_gdf, src.crs)
                    if aoi_transformed is not None:
                        aoi_geojson = [mapping(geom) for geom in aoi_transformed.geometry]
                    else:
                        aoi_geojson = None
                else:
                    aoi_geojson = None
                
                # Se temos AOI, mascarar a imagem
                if aoi_geojson:
                    try:
                        out_image, out_transform = mask(
                            src, aoi_geojson, crop=True, 
                            nodata=0, filled=True, all_touched=True
                        )
                        
                        # Criar máscara de pixels válidos
                        valid_mask = (out_image[0] > 0) | (out_image[1] > 0) | (out_image[2] > 0)
                        
                        # Calcular tamanho do pixel em metros quadrados
                        pixel_x_size = abs(out_transform[0])
                        pixel_y_size = abs(out_transform[4])
                    except ValueError:
                        # Se não houver sobreposição, pular esta imagem
                        logging.warning(f"Imagem {image_name} não se sobrepõe à AOI")
                        continue
                else:
                    # Sem AOI, considerar a imagem completa
                    height, width = src.shape
                    valid_mask = np.ones((height, width), dtype=bool)
                    
                    # Tamanho do pixel
                    pixel_x_size = abs(src.transform[0])
                    pixel_y_size = abs(src.transform[4])
                
                # Área do pixel
                pixel_area = pixel_x_size * pixel_y_size
                
                # Área desta imagem
                valid_pixel_count = np.sum(valid_mask)
                area_m2 = valid_pixel_count * pixel_area
                
                # Adicionar à área total (simplificação: não considera sobreposição)
                total_area += area_m2
                
        except Exception as e:
            logging.error(f"Erro ao calcular área para {image_name}: {e}")
    
    return total_area

def visualize_mosaics():
    """Visualiza os mosaicos reais e compara com métricas PIE."""
    
    # Carregar dados do JSON
    with open(JSON_FILE, 'r') as f:
        json_data = json.load(f)
    
    # Verificar formato do JSON e extrair mosaicos
    if isinstance(json_data, list):
        # Formato esperado: lista de mosaicos
        mosaics = json_data
    elif isinstance(json_data, dict) and 'mosaic_groups' in json_data:
        # Formato optimization_parameters: dicionário com mosaic_groups
        mosaics = json_data.get('mosaic_groups', [])
    else:
        # Tentar encontrar a estrutura de mosaicos
        mosaics = []
        # Procurar por qualquer lista no JSON que possa conter mosaicos
        for key, value in json_data.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict) and 'images' in value[0]:
                    mosaics = value
                    break
        
        if not mosaics:
            logging.error(f"Não foi possível identificar mosaicos no arquivo JSON: {JSON_FILE}")
            logging.info(f"Estrutura do JSON: {list(json_data.keys())}")
            return
    
    region = os.path.basename(JSON_FILE).split('optimization_parameters-')[1].split('.json')[0] \
        if 'optimization_parameters-' in os.path.basename(JSON_FILE) else "RS"
    
    logging.info(f"Visualizando {len(mosaics)} mosaicos para região: {region}")
    
    # Carregar shapefile da AOI usando o path fixo
    if os.path.exists(SHP_FILE):
        aoi_gdf = gpd.read_file(SHP_FILE)
        # Manter no CRS original para transformação posterior
        aoi_gdf_wgs84 = aoi_gdf.to_crs('EPSG:4326')  # Para visualização no matplotlib
        logging.info(f"AOI carregada do shapefile: {SHP_FILE}")
    else:
        aoi_gdf = None
        aoi_gdf_wgs84 = None
        logging.warning(f"AOI não encontrada em: {SHP_FILE}")
    
    # Determinar layout do grid
    n_mosaics = 3
    cols = 4
    rows = math.ceil(n_mosaics / cols)
    
    # Criar figura
    fig = plt.figure(figsize=(16, 6 * rows))
    plt.suptitle(f"Mosaicos e Análise PIE vs Cobertura Real: {region}", fontsize=16)
    
    # Limpar qualquer arquivo antigo que possa estar no diretório
    try:
        for file in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        logging.error(f"Erro ao limpar diretório temporário: {e}")
    
    try:
        # Para cada mosaico definido no JSON
        for i, mosaic_data in enumerate(mosaics):
            if i >= n_mosaics:
                break
                
            # Extrair imagens deste mosaico
            image_names = mosaic_data.get('images', [])
            num_images = len(image_names)
            
            if num_images == 0:
                continue
                
            # Criar subplot para este mosaico
            ax = plt.subplot(rows, cols, i+1)
            
            logging.info(f"Processando mosaico {i+1} com {num_images} imagens...")
            
            # Preencher a AOI com preto como fundo antes de plotar as imagens
            if aoi_gdf_wgs84 is not None:
                for geom in aoi_gdf_wgs84.geometry:
                    if hasattr(geom, "geoms"):
                        # MultiPolygon
                        for poly in geom.geoms:
                            ax.add_patch(plt.Polygon(np.array(poly.exterior.coords), 
                                                     closed=True, facecolor='black', 
                                                     edgecolor=None, zorder=0))
                    else:
                        # Polygon
                        ax.add_patch(plt.Polygon(np.array(geom.exterior.coords), 
                                                 closed=True, facecolor='black', 
                                                 edgecolor=None, zorder=0))
            
            # Lista para armazenar os limites de todas as imagens para ajustar o view
            all_bounds = []
            
            # Plotar cada imagem individualmente
            for j, image_name in enumerate(image_names):
                tci_path = get_tci_path(image_name, TEMP_DIR)
                if tci_path and os.path.exists(tci_path):
                    try:
                        with rasterio.open(tci_path) as src:
                            # Armazenar os limites da imagem
                            img_bounds = transform_bounds(src.crs, 'EPSG:4326', 
                                                         src.bounds.left, src.bounds.bottom, 
                                                         src.bounds.right, src.bounds.top)
                            all_bounds.append(img_bounds)
                            
                            # Se tivermos AOI, mascarar a imagem - transforma AOI para o CRS da imagem
                            if aoi_gdf is not None:
                                aoi_transformed = transform_aoi_to_raster_crs(aoi_gdf, src.crs)
                                if aoi_transformed is not None:
                                    try:
                                        aoi_geojson = [mapping(geom) for geom in aoi_transformed.geometry]
                                        out_image, out_transform = mask(
                                            src, aoi_geojson, crop=True, 
                                            nodata=0, filled=True, all_touched=True
                                        )
                                        show(out_image, transform=out_transform, ax=ax)
                                    except ValueError:
                                        # Se não houver sobreposição, mostrar a imagem completa
                                        logging.warning(f"Imagem {image_name} não se sobrepõe à AOI, mostrando completa")
                                        show(src, ax=ax)
                                else:
                                    show(src, ax=ax)
                            else:
                                show(src, ax=ax)
                            
                            # Adicionar o bounding box para referência
                            bbox = get_image_bbox(image_name, TEMP_DIR)
                            if bbox:
                                x, y = bbox.exterior.xy
                                ax.plot(x, y, color='yellow', linewidth=1.5)
                                ax.text(x[0], y[0], f"{j+1}", fontsize=8, 
                                        color='yellow', weight='bold', backgroundcolor='black')
                    except Exception as e:
                        logging.error(f"Erro ao plotar {image_name}: {e}")
                        traceback.print_exc()
            
            # Calcular área aproximada do mosaico
            actual_area = calculate_mosaic_area(image_names, TEMP_DIR, aoi_gdf)
            
            # Mostrar a AOI se disponível
            if aoi_gdf_wgs84 is not None:
                aoi_gdf_wgs84.boundary.plot(ax=ax, color='red', linewidth=2)
            
            # Ajustar os limites da visualização para mostrar todas as imagens
            if all_bounds:
                min_x = min(bounds[0] for bounds in all_bounds)
                min_y = min(bounds[1] for bounds in all_bounds)
                max_x = max(bounds[2] for bounds in all_bounds)
                max_y = max(bounds[3] for bounds in all_bounds)
                
                # Adicionar uma pequena margem
                margin = 0.05
                width = max_x - min_x
                height = max_y - min_y
                ax.set_xlim(min_x - margin * width, max_x + margin * width)
                ax.set_ylim(min_y - margin * height, max_y + margin * height)
            
            # Adicionar título
            title = f"Mosaico {i+1}"
            if 'group_id' in mosaic_data:
                title = f"Mosaico {mosaic_data['group_id']}"
            if 'time_window_start' in mosaic_data:
                title += f"\n{mosaic_data['time_window_start'][:10]}"
                
            ax.set_title(title)
            
            # Extrair métricas do JSON
            json_area = mosaic_data.get('geometric_coverage_m2', 0)
            raw_area = mosaic_data.get('total_individual_area', 0)
            intersections = mosaic_data.get('total_pairwise_overlap', 0)
            pie_2a2 = raw_area - intersections
            
            # Calcular diferenças
            pie_diff = json_area - pie_2a2
            actual_vs_json = actual_area - json_area
            
            # Exibir métricas como texto
            metrics_text = [
                f"Imagens: {num_images}",
                f"Área Real (approx): {actual_area/1e6:.2f} km²",
                f"Área JSON: {json_area/1e6:.2f} km²",
                f"PIE 2a2: {pie_2a2/1e6:.2f} km²",
                f"Diff JSON-PIE: {pie_diff/1e6:.2f} km²",
                f"Diff Real-JSON: {actual_vs_json/1e6:.2f} km²"
            ]
            
            y_pos = -0.10
            for txt in metrics_text:
                ax.text(0.5, y_pos, txt, ha='center', transform=ax.transAxes, fontsize=9)
                y_pos -= 0.05
            
            # Adicionar legenda
            yellow_patch = mpatches.Patch(color='yellow', label='Bounding Box')
            red_patch = mpatches.Patch(color='red', label='AOI')
            ax.legend(handles=[yellow_patch, red_patch], loc='upper right', fontsize=8)
            
            ax.axis('on')  # Mostrar eixos para visualizar coordenadas
            
            # Limpar arquivos temporários a cada 2 mosaicos
            if i % 2 == 1:
                try:
                    logging.info(f"Limpeza intermediária de arquivos temporários...")
                    for file in os.listdir(TEMP_DIR):
                        file_path = os.path.join(TEMP_DIR, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                except Exception as e:
                    logging.warning(f"Erro na limpeza intermediária: {e}")
    
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Salvar resultado como JPEG
        output_file = f"real_mosaics_{region}.jpg"
        plt.savefig(output_file, dpi=100, bbox_inches='tight', format='jpeg')
        logging.info(f"Visualização salva como {output_file}")
        
    finally:
        # Limpeza final do diretório temporário
        logging.info(f"Limpeza final dos arquivos temporários em {TEMP_DIR}")
        try:
            for file in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            logging.error(f"Erro na limpeza final: {e}")
    
    # Mostrar resultado
    # plt.show()

def main():
    """Função principal simplificada - roda diretamente com paths fixos."""
    visualize_mosaics()

if __name__ == "__main__":
    main()
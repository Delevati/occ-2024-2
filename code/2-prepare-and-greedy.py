"""
Script para processamento de imagens de satélite Sentinel-2 e criação de combinações de mosaicos.

Este script realiza a extração, análise e classificação de imagens de satélite Sentinel-2
contidas em arquivos ZIP. As imagens são avaliadas quanto à sua cobertura geográfica,
percentual de pixels válidos, cobertura de nuvens e eficácia da cobertura.
Após a classificação em imagens centrais e complementares, um algoritmo guloso
é aplicado para encontrar combinações otimizadas de mosaicos.

Autor: Luryand
"""

import os
import re
import time
import json
import zipfile
import logging
import numpy as np
import geopandas as gpd
import rasterio
import pyproj
from datetime import datetime, timedelta
from pathlib import Path
from shapely.geometry import Polygon, box
from shapely.ops import transform as shapely_transform
from collections import defaultdict

# Importar módulos dos greedy_utils
from greedy_utils.configuration import *
from greedy_utils.file_utils import safe_extract, remove_dir_contents
from greedy_utils.metadata_utils import get_date_from_xml, extract_orbit_from_filename, save_classification_metadata
from greedy_utils.image_processing import calculate_coverage_metrics, calculate_cloud_coverage, check_image_suitability, classify_image
from greedy_utils.metadata_utils import get_cloud_cover_in_geom

# --- Codificador JSON personalizado para objetos datetime ---
class DateTimeEncoder(json.JSONEncoder):
    """
    Codificador JSON personalizado para lidar com objetos datetime.
    """
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# --- Lógica de Processamento Principal ---
def process_single_zip_file(zip_path: Path, index: int, total: int, aoi_gdf_wgs84: gpd.GeoDataFrame, aoi_area_wgs84: float) -> dict | None:
    """
    Processa um único arquivo ZIP de imagem Sentinel-2, extraindo os arquivos necessários e calculando métricas.
    """
    logging.info(f"Processando {index+1}/{total}: {zip_path.name}")
    temp_dir = TEMP_EXTRACT_DIR / zip_path.stem
    temp_dir.mkdir(exist_ok=True)
    
    result_data = {
        'status': 'error', 'reason': 'Erro de processamento desconhecido', 'filename': zip_path.name, 'date': None, 'orbit': None,
        'geographic_coverage': 0.0, 'valid_pixels_percentage': 0.0, 'effective_coverage': 0.0, 'cloud_coverage': 1.0,
        'bounds': None, 'crs': None, 'path': None, 'tci_path': None, 'temp_tci_path': None,
        'cloud_mask_path': None, 'temp_cloud_mask_path': None
    }
    
    # Extrai arquivos
    required_patterns = {"MTD_MSIL2A.xml": None, "MSK_CLDPRB_20m.jp2": None, "TCI_10m.jp2": None}
    xml_path = cloud_mask_path_temp = rgb_image_path_temp = None
    
    try:
        with zipfile.ZipFile(zip_path) as zip_ref:
            extracted = safe_extract(zip_ref, required_patterns.keys(), temp_dir)
            if not extracted:
                result_data['reason'] = "Falha ao extrair arquivos necessários do ZIP"
                return result_data
            
            for pattern, paths in extracted.items():
                if not paths:
                    result_data['reason'] = f"Arquivo necessário ausente: {pattern}"
                    return result_data
                
                if "MTD_MSIL2A.xml" in pattern and paths:
                    xml_path = Path(paths[0])
                elif "MSK_CLDPRB_20m.jp2" in pattern and paths:
                    cloud_mask_path_temp = Path(paths[0])
                elif "TCI_10m.jp2" in pattern and paths:
                    rgb_image_path_temp = Path(paths[0])
        
        if not (xml_path and cloud_mask_path_temp and rgb_image_path_temp):
            result_data['reason'] = "Não foi possível localizar todos os arquivos necessários após a extração"
            return result_data
        
        # Extrai metadados
        date_obj = get_date_from_xml(xml_path)
        orbit = extract_orbit_from_filename(zip_path.name)
        result_data['date'] = date_obj
        result_data['orbit'] = orbit
        
        # Transforma AOI para o CRS da imagem
        aoi_gdf_crs_rgb = None
        try:
            with rasterio.open(rgb_image_path_temp) as src:
                img_crs = src.crs
                if img_crs:
                    aoi_gdf_crs_rgb = aoi_gdf_wgs84.to_crs(img_crs)
        except Exception as e:
            result_data['reason'] = f"Falha ao transformar AOI para CRS da imagem: {e}"
            return result_data
        
        # Calcula métricas de cobertura
        coverage_metrics = calculate_coverage_metrics(rgb_image_path_temp, aoi_gdf_crs_rgb, aoi_area_wgs84)
        result_data.update(coverage_metrics)
        
        # Calcula cobertura de nuvens
        aoi_gdf_crs_cloud = aoi_gdf_wgs84.to_crs(coverage_metrics.get('crs'))
        cloud_coverage = calculate_cloud_coverage(cloud_mask_path_temp, aoi_gdf_crs_cloud)
        result_data['cloud_coverage'] = cloud_coverage
        
        # Verifica se a imagem é adequada
        is_suitable, reason = check_image_suitability(
            result_data['geographic_coverage'],
            result_data['valid_pixels_percentage'],
            result_data['effective_coverage'],
            result_data['cloud_coverage']
        )
        
        if not is_suitable:
            result_data['status'] = 'rejected'
            result_data['reason'] = reason
            return result_data
        
        # Classifica e salva imagem
        classification = classify_image(result_data['effective_coverage'])
        output_dir = OUTPUT_BASE_DIR / classification
        output_dir.mkdir(exist_ok=True)
        
        # Salva caminhos temporários
        result_data['temp_tci_path'] = str(rgb_image_path_temp)
        result_data['temp_cloud_mask_path'] = str(cloud_mask_path_temp)
        
        # Atualiza resultado com aceitação
        result_data['status'] = 'accepted'
        result_data['class'] = classification
        
        return result_data

    except Exception as e:
        logging.error(f"Erro ao processar arquivo ZIP {zip_path}: {e}")
        result_data['reason'] = f"Erro de processamento: {str(e)}"
        return result_data

# --- Funções de Busca de Mosaicos ---
def calculate_refined_compatibility(base_img: dict, other_img: dict, max_days: int) -> dict | None:
    """
    Calcula a compatibilidade refinada entre duas imagens para mosaicos.
    """
    # Extrai e valida datas
    base_date_str = base_img.get('date')
    other_date_str = other_img.get('date')
    
    try:
        if isinstance(base_date_str, str):
            base_date = datetime.fromisoformat(base_date_str.replace('Z', '+00:00'))
        else:
            base_date = base_date_str
            
        if isinstance(other_date_str, str):
            other_date = datetime.fromisoformat(other_date_str.replace('Z', '+00:00'))
        else:
            other_date = other_date_str
            
        if not isinstance(base_date, datetime) or not isinstance(other_date, datetime):
            return None
    except (ValueError, TypeError):
        return None

    # Verifica diferença de tempo
    days_diff = abs((base_date - other_date).days)
    if days_diff > max_days:
        return None

    # Verifica correspondência de órbita
    base_orbit = base_img.get('orbit')
    other_orbit = other_img.get('orbit')
    orbit_match = base_orbit is not None and base_orbit == other_orbit
    orbit_bonus = 0.05 if orbit_match else 0

    base_bounds_dict = base_img.get('bounds')
    other_bounds_dict = other_img.get('bounds')
    base_crs_str = base_img.get('crs')
    other_crs_str = other_img.get('crs')
    
    # Verifica se bounds são válidos
    if not isinstance(base_bounds_dict, dict) or not all(k in base_bounds_dict for k in ['left', 'bottom', 'right', 'top']):
        return None
    
    # Mesmo processo para a outra imagem
    if not isinstance(other_bounds_dict, dict) or not all(k in other_bounds_dict for k in ['left', 'bottom', 'right', 'top']):
        return None
            
    # Continua com a validação de CRS
    if not base_crs_str or not other_crs_str:
        return None

    # Calcula geometria de sobreposição
    overlap_geom_wgs84 = None
    overlap_details = {}
    
    try:
        # Converte bounds para polígonos
        base_poly = box(
            base_bounds_dict['left'], 
            base_bounds_dict['bottom'],
            base_bounds_dict['right'], 
            base_bounds_dict['top']
        )
        
        other_poly = box(
            other_bounds_dict['left'],
            other_bounds_dict['bottom'],
            other_bounds_dict['right'],
            other_bounds_dict['top']
        )
        
        # Transforma para WGS84 para cálculo consistente de sobreposição
        base_crs = pyproj.CRS.from_string(base_crs_str)
        other_crs = pyproj.CRS.from_string(other_crs_str)
        wgs84_crs = pyproj.CRS.from_epsg(4326)
        
        # Cria transformadores
        base_to_wgs84 = pyproj.Transformer.from_crs(base_crs, wgs84_crs, always_xy=True)
        other_to_wgs84 = pyproj.Transformer.from_crs(other_crs, wgs84_crs, always_xy=True)
        
        # Transforma geometrias para WGS84
        base_poly_wgs84 = shapely_transform(lambda x, y: base_to_wgs84.transform(x, y), base_poly)
        other_poly_wgs84 = shapely_transform(lambda x, y: other_to_wgs84.transform(x, y), other_poly)
        
        # Calcula sobreposição
        if base_poly_wgs84.intersects(other_poly_wgs84):
            overlap_geom_wgs84 = base_poly_wgs84.intersection(other_poly_wgs84)
            overlap_area = overlap_geom_wgs84.area
            base_area = base_poly_wgs84.area
            other_area = other_poly_wgs84.area
            
            overlap_details = {
                'overlap_area': overlap_area,
                'base_area': base_area,
                'other_area': other_area,
                'overlap_ratio_base': overlap_area / base_area if base_area > 0 else 0,
                'overlap_ratio_other': overlap_area / other_area if other_area > 0 else 0
            }
        else:
            overlap_details = {
                'overlap_area': 0,
                'base_area': base_poly_wgs84.area,
                'other_area': other_poly_wgs84.area,
                'overlap_ratio_base': 0,
                'overlap_ratio_other': 0
            }

    except Exception as e:
        logging.warning(f"Erro ao calcular sobreposição: {e}")
        return None

    # Calcula cobertura de nuvens na área de sobreposição
    cloud_overlap_base = 1.0
    cloud_overlap_other = 1.0
    better_img_in_overlap = None

    if overlap_geom_wgs84 and overlap_geom_wgs84.area > 0:
        # Calcula cobertura de nuvens na área de sobreposição para ambas as imagens
        if base_img.get('cloud_mask_path') and os.path.exists(base_img.get('cloud_mask_path')):
            cloud_overlap_base = get_cloud_cover_in_geom(base_img.get('cloud_mask_path'), overlap_geom_wgs84, wgs84_crs)
            
        if other_img.get('cloud_mask_path') and os.path.exists(other_img.get('cloud_mask_path')):
            cloud_overlap_other = get_cloud_cover_in_geom(other_img.get('cloud_mask_path'), overlap_geom_wgs84, wgs84_crs)
        
        # Determina qual imagem é melhor na área de sobreposição
        if cloud_overlap_base <= cloud_overlap_other:
            better_img_in_overlap = 'base'
        else:
            better_img_in_overlap = 'other'
            
        overlap_details['cloud_overlap_base'] = cloud_overlap_base
        overlap_details['cloud_overlap_other'] = cloud_overlap_other
        overlap_details['better_img_in_overlap'] = better_img_in_overlap
    else:
        overlap_details['cloud_overlap_base'] = cloud_overlap_base
        overlap_details['cloud_overlap_other'] = cloud_overlap_other
        overlap_details['better_img_in_overlap'] = None

    # Calcula fatores de qualidade
    quality_base = (1.0 - base_img.get('cloud_coverage', 1.0)) * base_img.get('valid_pixels_percentage', 0.0)
    quality_other = (1.0 - other_img.get('cloud_coverage', 1.0)) * other_img.get('valid_pixels_percentage', 0.0)

    # Calcula qualidade de sobreposição
    if better_img_in_overlap == 'base':
        quality_factor_overlap = (1.0 - cloud_overlap_base) * base_img.get('valid_pixels_percentage', 0.0)
    elif better_img_in_overlap == 'other':
        quality_factor_overlap = (1.0 - cloud_overlap_other) * other_img.get('valid_pixels_percentage', 0.0)
    else:
        quality_factor_overlap = (quality_base + quality_other) / 2.0

    # Calcula fator de qualidade refinado
    refined_quality_factor = ((1.0 - OVERLAP_QUALITY_WEIGHT) * ((quality_base + quality_other) / 2.0) +
                            OVERLAP_QUALITY_WEIGHT * quality_factor_overlap)

    # Calcula cobertura estimada
    uncovered_area_before = max(0.0, 1.0 - base_img.get('geographic_coverage', 0.0))
    overlap_factor_heuristic = 0.4 if other_img.get('class') == 'central' else 0.2
    contribution_factor = (1.0 - overlap_factor_heuristic)
    added_geo_coverage_est = min(uncovered_area_before, other_img.get('geographic_coverage', 0.0) * contribution_factor)
    estimated_new_geo_coverage = min(1.0, base_img.get('geographic_coverage', 0.0) + added_geo_coverage_est)

    # Calcula pontuação final de eficácia
    effectiveness_score = (added_geo_coverage_est * refined_quality_factor) + orbit_bonus

    return {
        'image': other_img,
        'days_diff': days_diff,
        'estimated_coverage_after_add': estimated_new_geo_coverage,
        'refined_quality_factor': refined_quality_factor,
        'effectiveness_score': effectiveness_score,
        'orbit_match': orbit_match,
        'overlap_details': overlap_details
    }

def heuristica_gulosa(image_metadata: dict, max_days_diff: int) -> list:
    """
    Encontra combinações potenciais de mosaicos usando um algoritmo guloso.
    """
    logging.info("\nAnalisando combinações potenciais de mosaicos...")
    potential_mosaics = []
    centrals = image_metadata.get('central', [])
    complements = image_metadata.get('complement', [])
    all_accepted = centrals + complements
    
    if not all_accepted:
        logging.warning("Nenhuma imagem aceita encontrada para combinações de mosaicos.")
        return []

    processed_centrals = set()
    for central_img in centrals:
        central_key = central_img.get('filename')
        if central_key in processed_centrals:
            continue
            
        mosaic_group = {
            'base_image': central_img,
            'complementary_images': [],
            'estimated_coverage': central_img.get('geographic_coverage', 0.0),
            'avg_quality_factor': (1.0 - central_img.get('cloud_coverage', 1.0)) * central_img.get('valid_pixels_percentage', 0.0),
            'start_date': central_img.get('date'),
            'end_date': central_img.get('date'),
            'overlap_details': []
        }
        
        # Encontra complementos compatíveis
        candidate_imgs = sorted(
            [img for img in all_accepted if img.get('filename') != central_key],
            key=lambda x: x.get('geographic_coverage', 0.0) * (1.0 - x.get('cloud_coverage', 1.0)),
            reverse=True
        )
        
        # Usa calculate_refined_compatibility
        current_base = central_img
        for candidate in candidate_imgs:
            compatibility = calculate_refined_compatibility(current_base, candidate, max_days_diff)
            if not compatibility:
                continue
                
            # Adiciona ao grupo de mosaico
            mosaic_group['complementary_images'].append(compatibility['image'])
            mosaic_group['estimated_coverage'] = compatibility['estimated_coverage_after_add']
            mosaic_group['avg_quality_factor'] = compatibility['refined_quality_factor']
            
            # Atualiza intervalo de datas
            candidate_date = candidate.get('date')
            if candidate_date:
                if isinstance(candidate_date, str):
                    try:
                        candidate_date = datetime.fromisoformat(candidate_date.replace('Z', '+00:00'))
                    except:
                        candidate_date = None
                        
                if candidate_date:
                    if mosaic_group['start_date'] and candidate_date < mosaic_group['start_date']:
                        mosaic_group['start_date'] = candidate_date
                    if mosaic_group['end_date'] and candidate_date > mosaic_group['end_date']:
                        mosaic_group['end_date'] = candidate_date
            
            # Salva detalhes de sobreposição
            mosaic_group['overlap_details'].append(compatibility['overlap_details'])
            
            # Usa a imagem combinada como nova base para o próximo candidato
            current_base = {
                **current_base,
                'geographic_coverage': compatibility['estimated_coverage_after_add']
            }
        
        # Adiciona aos mosaicos potenciais se tiver imagens complementares
        if mosaic_group['complementary_images']:
            potential_mosaics.append(mosaic_group)
            processed_centrals.add(central_key)

    # Lógica para grupos apenas de complementos
    complement_groups = defaultdict(list)
    for comp_img in complements:
        if comp_img.get('date'):
            date_obj = comp_img.get('date')
            if isinstance(date_obj, str):
                try:
                    date_obj = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
                except:
                    continue
            date_key = date_obj.strftime('%Y-%m-%d')
            complement_groups[date_key].append(comp_img)

    for group_key, images_in_group in complement_groups.items():
        if len(images_in_group) < 2:
            continue
            
        # Ordena por cobertura * qualidade
        sorted_comps = sorted(
            images_in_group,
            key=lambda x: x.get('geographic_coverage', 0.0) * (1.0 - x.get('cloud_coverage', 1.0)),
            reverse=True
        )
        
        # Cria um grupo de mosaico começando com o melhor complemento
        base_img = sorted_comps[0]
        mosaic_group = {
            'base_image': base_img,
            'complementary_images': [],
            'estimated_coverage': base_img.get('geographic_coverage', 0.0),
            'avg_quality_factor': (1.0 - base_img.get('cloud_coverage', 1.0)) * base_img.get('valid_pixels_percentage', 0.0),
            'start_date': base_img.get('date'),
            'end_date': base_img.get('date'),
            'overlap_details': []
        }
        
        # Usa calculate_refined_compatibility
        current_base = base_img
        for candidate in sorted_comps[1:]:
            compatibility = calculate_refined_compatibility(current_base, candidate, max_days_diff)
            if not compatibility:
                continue
                
            # Adiciona ao grupo de mosaico
            mosaic_group['complementary_images'].append(candidate)
            mosaic_group['estimated_coverage'] = compatibility['estimated_coverage_after_add']
            mosaic_group['avg_quality_factor'] = compatibility['refined_quality_factor']
            
            # Atualiza intervalo de datas
            candidate_date = candidate.get('date')
            if candidate_date:
                if isinstance(candidate_date, str):
                    try:
                        candidate_date = datetime.fromisoformat(candidate_date.replace('Z', '+00:00'))
                    except:
                        candidate_date = None
                        
                if candidate_date:
                    if mosaic_group['start_date'] and candidate_date < mosaic_group['start_date']:
                        mosaic_group['start_date'] = candidate_date
                    if mosaic_group['end_date'] and candidate_date > mosaic_group['end_date']:
                        mosaic_group['end_date'] = candidate_date
            
            # Salva detalhes de sobreposição
            mosaic_group['overlap_details'].append(compatibility['overlap_details'])
            
            # Usa a imagem combinada como nova base para o próximo candidato
            current_base = {
                **current_base,
                'geographic_coverage': compatibility['estimated_coverage_after_add']
            }
        
        # Adiciona aos mosaicos potenciais se tiver imagens complementares
        if mosaic_group['complementary_images']:
            potential_mosaics.append(mosaic_group)

    logging.info(f"Identificadas {len(potential_mosaics)} combinações potenciais de mosaicos.")
    potential_mosaics.sort(key=lambda x: (x.get('estimated_coverage', 0.0), x.get('avg_quality_factor', 0.0)), reverse=True)
    return potential_mosaics

def run_processing_pipeline():
    """
    Executa o pipeline completo de processamento.
    """
    start_time = time.time()
    logging.info(f"Pipeline iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stats = defaultdict(int)
    image_metadata = {'central': [], 'complement': []}
    all_processed_metadata = []

    try:
        # Carrega e prepara AOI
        aoi_gdf = gpd.read_file(AOI_SHAPEFILE)
        aoi_gdf_wgs84 = aoi_gdf.to_crs(epsg=4326)
        aoi_area_wgs84 = sum(aoi_gdf_wgs84.geometry.area)
        logging.info(f"AOI carregada com área: {aoi_area_wgs84:.6f} graus quadrados")
    except Exception as e:
        logging.error(f"Falha ao carregar arquivo shapefile da AOI: {e}")
        return

    zip_files = sorted(list(ZIP_SOURCE_DIR.glob('S2*.zip')))
    total_zip_files = len(zip_files)
    logging.info(f"Encontrados {total_zip_files} arquivos ZIP correspondentes a 'S2*.zip' em {ZIP_SOURCE_DIR}")
    if total_zip_files == 0:
        logging.warning("Nenhum arquivo ZIP encontrado. Encerrando.")
        return

    image_process_start = time.time()
    for index, zip_path in enumerate(zip_files):
        try:
            result = process_single_zip_file(zip_path, index, total_zip_files, aoi_gdf_wgs84, aoi_area_wgs84)
            if not result:
                stats['error'] += 1
                continue
                
            all_processed_metadata.append(result)
            
            if result['status'] == 'accepted':
                stats['accepted'] += 1
                classification = result['class']
                stats[classification] += 1
                
                # Salva metadados
                output_dir = METADATA_DIR / classification
                output_dir.mkdir(exist_ok=True)
                metadata = save_classification_metadata(
                    output_dir, classification, result, 
                    result.get('date'), result.get('orbit'), 
                    zip_path.name
                )
                
                image_metadata[classification].append(metadata)
            elif result['status'] == 'rejected':
                stats['rejected'] += 1
            else:
                stats['error'] += 1
                
        except Exception as e:
            logging.error(f"Erro ao processar {zip_path}: {e}")
            stats['error'] += 1
    
    image_process_end = time.time()
    image_process_time = image_process_end - image_process_start
    logging.info(f"Processamento de imagens concluído em {image_process_time:.2f} segundos")

    all_metadata_path = METADATA_DIR / 'all_processed_images_log.json'
    try:
        with open(all_metadata_path, 'w') as f:
            json.dump(all_processed_metadata, f, indent=2, cls=DateTimeEncoder)
    except IOError as e:
        logging.error(f"Falha ao escrever log de metadados: {e}")
    except TypeError as e_serial:
        logging.error(f"Erro de serialização JSON: {e_serial}")

    greedy_start = time.time()
    logging.info("\n--- Executando Algoritmo Guloso para Combinações de Mosaicos ---")
    logging.info(f"Imagens centrais disponíveis: {len(image_metadata.get('central', []))}")
    logging.info(f"Imagens complementares disponíveis: {len(image_metadata.get('complement', []))}")
    good_mosaic_combinations = heuristica_gulosa(image_metadata, MOSAIC_TIME_WINDOW_DAYS)
    greedy_end = time.time()
    greedy_time = greedy_end - greedy_start
    logging.info(f"Algoritmo guloso concluído em {greedy_time:.2f} segundos")
    logging.info(f"Encontradas {len(good_mosaic_combinations)} combinações potenciais de mosaicos.")

    optimization_params = {'image_catalog': [], 'mosaic_groups': []}
    
    # Adiciona imagens ao catálogo
    for img_class in ['central', 'complement']:
        for img in image_metadata.get(img_class, []):
            catalog_entry = {
                'id': img.get('filename'),
                'class': img_class,
                'date': img.get('date'),
                'orbit': img.get('orbit'),
                'geographic_coverage': img.get('geographic_coverage', 0.0),
                'valid_pixels_percentage': img.get('valid_pixels_percentage', 0.0),
                'cloud_coverage': img.get('cloud_coverage', 1.0),
                'quality_factor': (1.0 - img.get('cloud_coverage', 1.0)) * img.get('valid_pixels_percentage', 0.0)
            }
            optimization_params['image_catalog'].append(catalog_entry)
    
    # Adiciona grupos de mosaicos
    for idx, mosaic in enumerate(good_mosaic_combinations):
        mosaic_entry = {
            'id': f"mosaic_{idx + 1}",
            'base_image_id': mosaic.get('base_image', {}).get('filename'),
            'complementary_image_ids': [img.get('filename') for img in mosaic.get('complementary_images', [])],
            'estimated_coverage': mosaic.get('estimated_coverage', 0.0),
            'quality_factor': mosaic.get('avg_quality_factor', 0.0),
            'start_date': mosaic.get('start_date'),
            'end_date': mosaic.get('end_date'),
            'overlap_details': mosaic.get('overlap_details', [])
        }
        optimization_params['mosaic_groups'].append(mosaic_entry)
    
    opt_params_path = METADATA_DIR / 'optimization_parameters.json'
    try:
        with open(opt_params_path, 'w') as f:
            json.dump(optimization_params, f, indent=2, cls=DateTimeEncoder)
    except IOError as e:
        logging.error(f"Falha ao escrever parâmetros de otimização: {e}")
    except TypeError as e_serial:
        logging.error(f"Erro de serialização JSON: {e_serial}")

    end_time = time.time()
    total_time = end_time - start_time
    
    total_processed = stats['accepted'] + stats['rejected'] + stats['error']
    total_accepted = stats['accepted']
    total_rejected = stats['rejected']
    total_errors = stats['error']
    
    print("\n" + "="*60 + "\nESTATÍSTICAS FINAIS DE PROCESSAMENTO:\n" + "="*60)
    print(f"Total de arquivos ZIP encontrados:       {total_zip_files}")
    print(f"Total de arquivos ZIP processados:       {total_processed}")
    print("-" * 30)
    print(f"Total de imagens aceitas:                {total_accepted} ({total_accepted/total_processed*100 if total_processed else 0:.1f}%)")
    print(f"  - Aceitas (arquivos copiados OK):      {total_accepted - stats.get('accepted_copy_error', 0)}")
    print(f"  - Aceitas (erro na cópia de arquivo):  {stats.get('accepted_copy_error', 0)}")
    print(f"    - Imagens centrais:                  {stats['central']}")
    print(f"    - Imagens complementares:            {stats['complement']}")
    print("-" * 30)
    print(f"Total de imagens rejeitadas:             {total_rejected} ({total_rejected/total_processed*100 if total_processed else 0:.1f}%)")
    print("-" * 30)
    print(f"Total de erros de processamento:         {total_errors} ({total_errors/total_processed*100 if total_processed else 0:.1f}%)")
    print("="*60)
    print(f"Combinações potenciais de mosaicos identificadas: {len(good_mosaic_combinations)}")
    print(f"Diretório de metadados:                  {METADATA_DIR}")
    print(f"Arquivo de parâmetros de otimização:     {opt_params_path}")
    print(f"Este arquivo pode agora ser usado com CPLEX para a etapa final de otimização.")
    print("="*60)
    print(f"TEMPOS DE EXECUÇÃO:")
    print(f"Tempo de processamento de imagens:       {image_process_time:.2f} segundos")
    print(f"Tempo do algoritmo guloso:               {greedy_time:.2f} segundos")
    print(f"Tempo total de execução:                 {total_time:.2f} segundos")
    print("="*60)
    logging.info(f"Pipeline finalizado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Tempo total de execução: {total_time:.2f} segundos")

if __name__ == "__main__":
    run_processing_pipeline()
"""
Utilitários para processamento de imagens de satélite.

Este módulo contém funções para calcular métricas de cobertura,
analisar cobertura de nuvens e classificar imagens quanto à sua
adequação para uso em mosaicos.
"""

import rasterio
import numpy as np
import geopandas as gpd
import logging
import traceback
from shapely.geometry import Polygon, box
from .metadata_utils import get_cloud_cover_in_geom
import pyproj
from shapely.ops import transform as shapely_transform

def calculate_coverage_metrics(raster_path, aoi_gdf_crs_raster, aoi_area_wgs84):
    """
    Calcula métricas de cobertura para uma imagem raster em relação a uma área de interesse.
    
    Parâmetros:
        raster_path: Caminho para o arquivo raster
        aoi_gdf_crs_raster: GeoDataFrame da AOI no mesmo CRS do raster
        aoi_area_wgs84: Área da AOI em WGS84
        
    Retorno:
        Dicionário com métricas calculadas: cobertura geográfica, percentual de pixels válidos,
        cobertura efetiva, limites (bounds) e CRS
    """
    metrics = {
        'geographic_coverage': 0.0, 
        'valid_pixels_percentage': 0.0, 
        'effective_coverage': 0.0,
        'bounds': None, 
        'crs': None
    }
    
    try:
        with rasterio.open(raster_path) as src:
            metrics['bounds'] = src.bounds
            metrics['crs'] = src.crs if src.crs else None
            if not metrics['crs']:
                logging.error(f"CRS ausente para o raster {raster_path}. Não é possível calcular cobertura.")
                return metrics

            aoi_geometry_transformed = aoi_gdf_crs_raster.geometry.iloc[0]
            img_poly_raster_crs = box(*src.bounds)
            img_gdf_raster_crs = gpd.GeoDataFrame(geometry=[img_poly_raster_crs], crs=src.crs)

            try:
                # Calcula interseção com AOI
                intersection = aoi_geometry_transformed.intersection(img_poly_raster_crs)
                intersection_area = intersection.area if not intersection.is_empty else 0
                
                # Calcula cobertura geográfica
                aoi_geometry_area = aoi_gdf_crs_raster.geometry.area.sum()
                metrics['geographic_coverage'] = min(1.0, intersection_area / aoi_area_wgs84)
                
                # Calcula percentual de pixels válidos
                raster_data = src.read(1, masked=True)
                total_pixels = raster_data.size
                valid_pixels = np.sum(~raster_data.mask)
                metrics['valid_pixels_percentage'] = valid_pixels / total_pixels if total_pixels > 0 else 0.0
                
                # Calcula cobertura efetiva
                metrics['effective_coverage'] = metrics['geographic_coverage'] * metrics['valid_pixels_percentage']
                
                return metrics
            except Exception as e_geo:
                logging.error(f"Erro ao calcular geometria para {raster_path}: {e_geo}\n{traceback.format_exc()}")
                return metrics
    except rasterio.RasterioIOError as e_rio:
        logging.error(f"Erro de IO do Rasterio: {e_rio}")
    except Exception as e:
        logging.error(f"Erro ao calcular métricas de cobertura: {e}\n{traceback.format_exc()}")
    
    return metrics

def calculate_cloud_coverage(cloud_mask_path, aoi_gdf_crs_mask):
    """
    Calcula a cobertura de nuvens para uma área de interesse usando uma máscara de nuvens.
    
    Parâmetros:
        cloud_mask_path: Caminho para o arquivo de máscara de nuvens
        aoi_gdf_crs_mask: GeoDataFrame da AOI no mesmo CRS da máscara de nuvens
        
    Retorno:
        Percentual (0-1) de cobertura de nuvens na AOI
    """
    try:
        with rasterio.open(cloud_mask_path) as src:
            try:
                # Cria máscara para AOI
                geom = aoi_gdf_crs_mask.geometry.iloc[0]
                aoi_mask = rasterio.features.geometry_mask(
                    [geom],
                    out_shape=(src.height, src.width),
                    transform=src.transform,
                    invert=True
                )
                
                # Lê dados da máscara de nuvens
                cloud_data = src.read(1, masked=False)
                
                # Aplica máscara AOI
                cloud_data_aoi = cloud_data[aoi_mask]
                
                if cloud_data_aoi.size == 0:
                    logging.warning(f"AOI não tem interseção com a máscara de nuvens {cloud_mask_path}.")
                    return 1.0
                
                # Calcula percentual de nuvens (valores > 0 são considerados nuvens)
                cloudy_pixels = np.sum(cloud_data_aoi > 0)
                return cloudy_pixels / cloud_data_aoi.size
            except ValueError:
                logging.warning(f"Geometria da AOI provavelmente fora dos limites da máscara de nuvens {cloud_mask_path}. "
                                f"Assumindo 100% de nuvens por segurança.")
                return 1.0
            except Exception as e_cloud_mask:
                logging.error(f"Erro durante processamento da máscara de nuvens para {cloud_mask_path}: {e_cloud_mask}\n{traceback.format_exc()}")
                return 1.0
    except rasterio.RasterioIOError as e_rio:
        logging.error(f"Erro do Rasterio ao abrir máscara de nuvens {cloud_mask_path}: {e_rio}")
    except Exception as e:
        logging.error(f"Erro ao calcular cobertura de nuvens para {cloud_mask_path}: {e}")
    
    return 1.0  # Retorna 100% de cobertura de nuvens em caso de erro

def check_image_suitability(geo_coverage, valid_pix_perc, eff_coverage, cloud_perc):
    """
    Verifica se uma imagem é adequada para processamento baseado em suas métricas.
    
    Parâmetros:
        geo_coverage: Cobertura geográfica (0-1)
        valid_pix_perc: Percentual de pixels válidos (0-1)
        eff_coverage: Cobertura efetiva (0-1)
        cloud_perc: Percentual de cobertura de nuvens (0-1)
        
    Retorno:
        Tupla (adequada, razão) onde adequada é um booleano e razão é uma string explicativa
    """
    from .configuration import COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD, MAX_CLOUD_COVERAGE_THRESHOLD
    
    if valid_pix_perc <= 1e-6:
        return False, f"IMAGEM SEM PIXELS VÁLIDOS NA AOI ({valid_pix_perc:.2%})"
        
    if geo_coverage < COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD: 
        return False, f"IMAGEM COM COBERTURA GEOGRÁFICA INSUFICIENTE ({geo_coverage:.2%} < " \
                      f"{COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD:.0%})"
                      
    min_effective_coverage_required = COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD * 0.5
    if eff_coverage < min_effective_coverage_required: 
        return False, f"IMAGEM COM COBERTURA EFETIVA INSUFICIENTE ({eff_coverage:.2%} < " \
                      f"{min_effective_coverage_required:.0%})"
                      
    if cloud_perc > MAX_CLOUD_COVERAGE_THRESHOLD: 
        return False, f"IMAGEM REJEITADA: Muitas nuvens ({cloud_perc:.1%} > {MAX_CLOUD_COVERAGE_THRESHOLD:.0%})"
        
    return True, "OK"

def classify_image(effective_coverage):
    """
    Classifica uma imagem como 'central' ou 'complement' baseado na cobertura efetiva.
    
    Parâmetros:
        effective_coverage: Cobertura efetiva da imagem (0-1)
        
    Retorno:
        String 'central' ou 'complement'
    """
    from .configuration import CENTRAL_IMAGE_EFFECTIVE_COVERAGE_THRESHOLD
    return "central" if effective_coverage >= CENTRAL_IMAGE_EFFECTIVE_COVERAGE_THRESHOLD else "complement"
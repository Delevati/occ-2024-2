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
import rasterio
from rasterio.mask import geometry_mask
from shapely.geometry import Polygon, box
from .metadata_utils import get_cloud_cover_in_geom
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
                geom = aoi_gdf_crs_raster.geometry.iloc[0]
                aoi_mask = geometry_mask(
                    [geom],
                    out_shape=(src.height, src.width),
                    transform=src.transform,
                    invert=True
                )

                raster_data = src.read(1, masked=False)
                raster_data_aoi = raster_data[aoi_mask]  # Apenas pixels na AOI

                # Calcula percentual de pixels válidos apenas na AOI
                total_pixels_in_aoi = raster_data_aoi.size
                valid_pixels_in_aoi = np.sum(raster_data_aoi > 0)
                    
                logging.debug(f"Total de pixels na AOI: {total_pixels_in_aoi}")
                logging.debug(f"Pixels válidos na AOI: {valid_pixels_in_aoi}")

                metrics['valid_pixels_percentage'] = valid_pixels_in_aoi / total_pixels_in_aoi if total_pixels_in_aoi > 0 else 0.0
                
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
    """
    try:
        geom = aoi_gdf_crs_mask.geometry.iloc[0]
        if aoi_gdf_crs_mask.crs and aoi_gdf_crs_mask.crs.to_epsg() != 4326:
            geom_wgs84 = aoi_gdf_crs_mask.to_crs(epsg=4326).geometry.iloc[0]
        else:
            geom_wgs84 = geom

        return get_cloud_cover_in_geom(cloud_mask_path, geom_wgs84, source_crs=aoi_gdf_crs_mask.crs)
    except Exception as e:
        logging.error(f"Erro ao calcular cobertura de nuvens: {e}")
        return 1.0

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
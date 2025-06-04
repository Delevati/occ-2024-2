import logging
import zipfile
import rasterio
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from greedy_utils.configuration import *
from greedy_utils.file_utils import safe_extract
from greedy_utils.metadata_utils import get_date_from_xml, extract_orbit_from_filename
from greedy_utils.image_processing import calculate_coverage_metrics, calculate_cloud_coverage, check_image_suitability, classify_image

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

        logging.info(f"  Geo Cov: {result_data['geographic_coverage']:.2%}, Valid Pix: {result_data['valid_pixels_percentage']:.2%}, Eff Cov: {result_data['effective_coverage']:.2%}, Cloud Cov: {result_data['cloud_coverage']:.2%}")
        
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
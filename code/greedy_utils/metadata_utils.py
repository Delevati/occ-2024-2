"""
Utilitários para processamento de metadados de imagens Sentinel-2.

Este módulo contém funções para extração, análise e manipulação
de metadados de imagens de satélite Sentinel-2, incluindo extração
de datas, órbitas, cálculo de cobertura de nuvens e outros atributos.
"""

import os
import re
import json
import logging
import xml.etree.ElementTree as ET
import traceback
from datetime import datetime
from pathlib import Path
import pyproj
import rasterio
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform

def get_date_from_xml(xml_path):
    """
    Extrai a data de um arquivo XML de metadados do Sentinel-2.
    
    Tenta encontrar a data em diversas tags do XML e formatos conhecidos.
    Se falhar, tenta extrair a data do nome do arquivo.
    
    Parâmetros:
        xml_path: Caminho para o arquivo XML de metadados
        
    Retorno:
        Objeto datetime com a data extraída ou None se falhar
    """
    date_formats = ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S']
    date_tags = ['DATATAKE_SENSING_START', 'SENSING_TIME', 'PRODUCT_START_TIME', 'GENERATION_TIME']
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for tag_name in date_tags:
            elements = root.findall(f".//*[contains(local-name(), '{tag_name}')]")
            if elements:
                for elem in elements:
                    for date_format in date_formats:
                        try:
                            return datetime.strptime(elem.text.strip(), date_format)
                        except (ValueError, TypeError):
                            continue
    except ET.ParseError as e_xml:
        logging.warning(f"Não foi possível analisar o arquivo XML: {xml_path} - {e_xml}")
    except Exception as e:
        logging.warning(f"Erro ao ler data do XML {xml_path}: {e}")
    
    # Tenta extrair a data do nome do arquivo caso a análise do XML falhe
    try:
        filename = os.path.basename(xml_path)
        date_match = re.search(r'_(\d{8}T\d{6})_', filename)
        if date_match:
            date_str = date_match.group(1)
            return datetime.strptime(date_str, '%Y%m%dT%H%M%S')
    except Exception as e:
        logging.warning(f"Erro ao extrair data do nome do arquivo {xml_path}: {e}")
    
    return None

def extract_orbit_from_filename(filename):
    """
    Extrai o número da órbita do nome de arquivo Sentinel-2.
    
    Parâmetros:
        filename: Nome do arquivo Sentinel-2
        
    Retorno:
        Número da órbita (int) ou None se não encontrado
    """
    orbit_match = re.search(r'_R(\d{3})_', filename)
    return int(orbit_match.group(1)) if orbit_match else None

def save_classification_metadata(output_dir, classification, metrics, date_obj, orbit, zip_filename):
    """
    Salva os metadados de classificação de uma imagem em arquivo JSON.
    
    Parâmetros:
        output_dir: Diretório onde o arquivo JSON será salvo
        classification: Classificação da imagem ('central' ou 'complement')
        metrics: Dicionário com métricas calculadas para a imagem
        date_obj: Objeto datetime com a data da imagem
        orbit: Número da órbita da imagem
        zip_filename: Nome do arquivo ZIP original
        
    Retorno:
        Dicionário com os metadados salvos
    """
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
    
    bounds_data = None
    if metrics.get('bounds'):
        bounds = metrics.get('bounds')
        if hasattr(bounds, '_asdict'):  # Se for namedtuple (BoundingBox)
            bounds_data = bounds._asdict()
        else:
            bounds_data = bounds  # Já é um dicionário
    
    metadata = {
        'source_zip': zip_filename,
        'filename': Path(zip_filename).stem,
        'status': metrics.get('status', 'unknown'),
        'reason': metrics.get('reason', ''),
        'class': classification if metrics.get('status', 'error').startswith('accepted') else None,
        'date': date_obj.isoformat() if date_obj else None,
        'orbit': orbit,
        'geographic_coverage': metrics.get('geographic_coverage', 0.0),
        'valid_pixels_percentage': metrics.get('valid_pixels_percentage', 0.0),
        'effective_coverage': metrics.get('effective_coverage', 0.0),
        'cloud_coverage': metrics.get('cloud_coverage', 1.0),
        'bounds': bounds_data,
        'crs': str(metrics.get('crs')) if metrics.get('crs') else None,
        'tci_path': metrics.get('tci_path'),
        'cloud_mask_path': metrics.get('cloud_mask_path'),
        'temp_tci_path': metrics.get('temp_tci_path'),
        'temp_cloud_mask_path': metrics.get('temp_cloud_mask_path')
    }
    
    try:
        output_path = output_dir / f"{Path(zip_filename).stem}.json"
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=DateTimeEncoder)
    except IOError as e:
        logging.error(f"Falha ao escrever metadados em {output_path}: {e}")
    except TypeError as e_serial:
        logging.error(f"Erro de serialização JSON: {e_serial}")
    
    return metadata

def get_cloud_cover_in_geom(cloud_mask_path, geometry_wgs84, source_crs=None):
    """
    Calcula a porcentagem de cobertura de nuvens dentro de uma geometria específica.
    
    Esta função pode ser chamada de duas formas:
    1. Com caminho direto para a máscara: get_cloud_cover_in_geom(path_str, geometry, crs)
    2. Com dicionário de metadados: get_cloud_cover_in_geom(img_meta, geometry)
    
    Parâmetros:
        cloud_mask_path: Caminho para o arquivo de máscara de nuvens ou dicionário de metadados
        geometry_wgs84: Geometria em WGS84 para a qual calcular a cobertura de nuvens
        source_crs: Sistema de referência de coordenadas da geometria (opcional)
        
    Retorno:
        Percentual (0-1) de cobertura de nuvens na geometria fornecida
    """
    # Compatibilidade com ambas formas de chamada
    if isinstance(cloud_mask_path, dict):
        # Caso 1: Recebeu dicionário de metadados
        img_meta = cloud_mask_path
        cloud_mask_path_str = img_meta.get('cloud_mask_path') or img_meta.get('temp_cloud_mask_path')
        if not cloud_mask_path_str:
            logging.warning(f"Caminho da máscara de nuvens não encontrado para {img_meta.get('filename')} em get_cloud_cover_in_geom.")
            return 1.0  # Assume 100% de nuvens se não encontrar o caminho
        
        # Verifica se o caminho existe
        cloud_mask_path = Path(cloud_mask_path_str)
        if not cloud_mask_path.is_absolute():
            # Tenta resolver caminho relativo ao BASE_VOLUME
            try:
                from .configuration import BASE_VOLUME
                cloud_mask_path = BASE_VOLUME / cloud_mask_path_str.lstrip('/')
            except ImportError:
                logging.warning(f"BASE_VOLUME não disponível na configuração. Tentando caminho como está: {cloud_mask_path}")
            except Exception as e_vol:
                logging.warning(f"Erro ao resolver caminho com BASE_VOLUME: {e_vol}. Tentando caminho como está: {cloud_mask_path}")
    else:
        # Caso 2: Recebeu caminho direto (string ou Path)
        cloud_mask_path = Path(cloud_mask_path)
    
    if not cloud_mask_path.exists():
        logging.warning(f"Arquivo de máscara de nuvens não encontrado em {cloud_mask_path}.")
        return 1.0  # Assume 100% de nuvens se o arquivo não existir
    
    try:
        with rasterio.open(cloud_mask_path) as cloud_src:
            mask_crs = cloud_src.crs
            if not mask_crs:
                logging.warning(f"CRS ausente para máscara de nuvens {cloud_mask_path}. Impossível calcular cobertura de nuvens na sobreposição.")
                return 1.0  # Assume 100% se CRS estiver ausente
            
            try:
                # Transforma a geometria de WGS84 para o CRS da máscara
                wgs84_crs = pyproj.CRS.from_epsg(4326)
                transformer = pyproj.Transformer.from_crs(wgs84_crs, mask_crs, always_xy=True)
                geometry_mask_crs = shapely_transform(transformer.transform, geometry_wgs84)
            except Exception as e_transform:
                logging.error(f"Falha ao transformar geometria para CRS da máscara {mask_crs}: {e_transform}")
                return 1.0  # Assume 100% em caso de erro de transformação
            
            try:
                # Cria máscara para a geometria
                geom_mask = rasterio.features.geometry_mask(
                    [geometry_mask_crs],
                    out_shape=(cloud_src.height, cloud_src.width),
                    transform=cloud_src.transform,
                    invert=True
                )
                
                # Lê os dados da máscara de nuvens dentro da geometria
                cloud_data = cloud_src.read(1, masked=True)
                cloud_data_in_geom = np.ma.masked_array(
                    cloud_data,
                    mask=~geom_mask
                )
                
                # Calcula percentual de nuvens (valores > 0 são considerados nuvens)
                valid_pixels = np.sum(~cloud_data_in_geom.mask)
                if valid_pixels > 0:
                    cloud_pixels = np.sum((cloud_data_in_geom > 0) & ~cloud_data_in_geom.mask)
                    return cloud_pixels / valid_pixels
                
                return 1.0  # Se não houver pixels válidos, assume 100% de nuvens
                
            except Exception as e_process:
                logging.error(f"Erro ao processar máscara de nuvens: {e_process}\n{traceback.format_exc()}")
                return 1.0
    
    except rasterio.RasterioIOError as e_rio:
        logging.error(f"Erro do Rasterio ao abrir máscara de nuvens {cloud_mask_path}: {e_rio}")
    except Exception as e:
        logging.error(f"Erro ao calcular cobertura de nuvens: {e}\n{traceback.format_exc()}")
    
    return 1.0  # Assume 100% de nuvens em caso de erro
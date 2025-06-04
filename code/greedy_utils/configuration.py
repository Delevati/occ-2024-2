"""
Configurações globais para o sistema de processamento de imagens Sentinel-2.

Este módulo define constantes, caminhos de diretórios e parâmetros 
que são usados por diversos componentes do sistema.
"""

from pathlib import Path
import logging
import matplotlib

# --- Configuração de logging e matplotlib ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuração de diretórios ---
BASE_VOLUME = Path("/Volumes/luryand")
ZIP_SOURCE_DIR = BASE_VOLUME / "nova_busca/RS"
OUTPUT_BASE_DIR = BASE_VOLUME / "coverage_otimization_rs"
TEMP_EXTRACT_DIR = BASE_VOLUME / "temp"
AOI_SHAPEFILE = Path("/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/RS/ibirapuita_31982.shp")
METADATA_DIR = OUTPUT_BASE_DIR / "metadata"
PLOTS_DIR = OUTPUT_BASE_DIR / 'publication_plots'
VALIDATION_DIR = OUTPUT_BASE_DIR / 'validation'
VALIDATION_TCIS_DIR = VALIDATION_DIR / 'rejected_tcis'
TRASH_DIR = OUTPUT_BASE_DIR / 'trash'

# --- Parâmetros de algoritmo ---
MOSAIC_MIN_CONTRIBUTION_THRESHOLD = 0.05
CENTRAL_IMAGE_EFFECTIVE_COVERAGE_THRESHOLD = 0.3
COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD = 0.02
MOSAIC_TIME_WINDOW_DAYS = 5
MAX_CLOUD_COVERAGE_THRESHOLD = 0.4
OVERLAP_QUALITY_WEIGHT = 0.3
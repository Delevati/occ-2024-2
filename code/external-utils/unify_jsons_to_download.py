import os
import json
import logging
from pathlib import Path
import copy

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])

# Diretório base onde estão as pastas de estados
BASE_DIR = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture"

# Estados disponíveis
STATES = ["AL", "BA", "MG", "MG-SP-RJ", "PI-PE-CE", "RS"]

# Nome do arquivo a procurar em cada pasta de estado
TARGET_FILENAME = "optimization_parameters.json"

# Arquivo de saída unificado
OUTPUT_PATH = "/Users/luryand/Documents/encode-image/coverage_otimization/unified_parameters.json"

def get_state_files():
    """Encontrar os arquivos optimization_parameters.json de cada estado"""
    files_by_state = {}
    
    for state in STATES:
        state_dir = os.path.join(BASE_DIR, state)
        if not os.path.exists(state_dir):
            logging.warning(f"Diretório não encontrado para estado {state}: {state_dir}")
            continue
            
        target_file = os.path.join(state_dir, TARGET_FILENAME)
        if os.path.exists(target_file):
            files_by_state[state] = target_file
            logging.info(f"Encontrado arquivo para {state}: {target_file}")
        else:
            logging.warning(f"Arquivo {TARGET_FILENAME} não encontrado para estado {state}")
    
    return files_by_state

def unify_jsons(state_files):
    """Unificar os JSONs de todos os estados em um único arquivo"""
    if not state_files:
        logging.error("Nenhum arquivo encontrado para processar")
        return None
    
    # Iniciar com estrutura vazia
    unified_data = {
        "image_catalog": [],
        "mosaic_groups": []
    }
    
    # Controle de duplicatas
    image_catalog_dict = {}
    group_ids = set()
    
    # Processar cada arquivo de estado
    for state, file_path in state_files.items():
        logging.info(f"Processando estado: {state}")
        
        try:
            with open(file_path, 'r') as f:
                state_data = json.load(f)
            
            # Processar catálogo de imagens
            for img in state_data.get("image_catalog", []):
                filename = img.get("filename")
                if filename and filename not in image_catalog_dict:
                    image_catalog_dict[filename] = img
            
            # Processar grupos de mosaico
            for group in state_data.get("mosaic_groups", []):
                # Adicionar código do estado ao ID para garantir unicidade
                original_id = group.get("group_id", "unknown")
                new_id = f"{original_id}_{state}"
                
                # Evitar duplicatas
                if new_id in group_ids:
                    logging.warning(f"Ignorando ID de grupo duplicado: {new_id}")
                    continue
                
                # Copiar o grupo e atualizar ID
                new_group = copy.deepcopy(group)
                new_group["group_id"] = new_id
                new_group["state"] = state
                
                # Registrar ID
                group_ids.add(new_id)
                
                # Adicionar ao resultado unificado
                unified_data["mosaic_groups"].append(new_group)
                
        except Exception as e:
            logging.error(f"Erro ao processar {state} ({file_path}): {e}")
    
    # Adicionar todas as imagens únicas ao catálogo unificado
    unified_data["image_catalog"] = list(image_catalog_dict.values())
    
    # Estatísticas
    logging.info(f"Dados unificados contêm:")
    logging.info(f"- {len(unified_data['image_catalog'])} imagens únicas")
    logging.info(f"- {len(unified_data['mosaic_groups'])} grupos de mosaico")
    
    return unified_data

def main():
    """Função principal para unificar JSONs dos estados"""
    logging.info("Iniciando unificação de JSONs dos estados")
    
    # Encontrar arquivos por estado
    state_files = get_state_files()
    
    # Unificar JSONs
    unified_data = unify_jsons(state_files)
    
    if unified_data:
        # Salvar arquivo unificado
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(unified_data, f, indent=2)
        
        logging.info(f"JSON unificado salvo em: {OUTPUT_PATH}")
        
        # Criar também arquivo de imagens para recapture
        images_list = {"images": [img["filename"] for img in unified_data["image_catalog"]]}
        recapture_file = OUTPUT_PATH.replace(".json", "_recapture.json")
        
        with open(recapture_file, 'w') as f:
            json.dump(images_list, f, indent=2)
        
        logging.info(f"Lista de imagens para recapture salva em: {recapture_file}")
        logging.info(f"Total de {len(images_list['images'])} imagens para download")
    else:
        logging.error("Falha ao gerar JSON unificado")

if __name__ == "__main__":
    main()
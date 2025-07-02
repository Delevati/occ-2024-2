import os
import datetime
import re
import json

def save_selected_mosaics_log(selected_mosaics, input_file_path=None, log_file_path="selected_mosaics_log.txt"):
    """
    Salva os mosaicos selecionados em um arquivo de log cumulativo e também gera um JSON
    compatível com o script 1.2-cdse-recapture.py para download das imagens.
    
    Args:
        selected_mosaics: Lista de dicionários com os grupos de mosaicos selecionados
        input_file_path: Caminho do arquivo de input para extrair o nome da área
        log_file_path: Caminho para o arquivo de log
    """
    area_name = "unknown"
    if input_file_path:
        filename = os.path.basename(input_file_path)
        matches = re.findall(r'[A-Z]{2,3}(?:-[A-Z]{2,3})*', filename)
        if matches:
            area_name = matches[0]
        else:
            dirname = os.path.basename(os.path.dirname(input_file_path))
            matches = re.findall(r'[A-Z]{2,3}(?:-[A-Z]{2,3})*', dirname)
            if matches:
                area_name = matches[0]
    
    if not input_file_path:
        import sys
        main_module = sys.modules['__main__']
        if hasattr(main_module, 'OPTIMIZATION_PARAMS_FILE'):
            file_path = getattr(main_module, 'OPTIMIZATION_PARAMS_FILE')
            if file_path:
                filename = os.path.basename(file_path)
                matches = re.findall(r'[A-Z]{2,3}(?:-[A-Z]{2,3})*', filename)
                if matches:
                    area_name = matches[0]
    
    if not selected_mosaics:
        print("Nenhum mosaico selecionado para registrar no log.")
        return
    
    # Criar o arquivo de log TXT (cumulativo)
    with open(log_file_path, 'a') as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"\n{'='*50}\n")
        log_file.write(f"EXECUÇÃO: {timestamp}\n")
        log_file.write(f"ÁREA: {area_name}\n")
        
        if input_file_path:
            log_file.write(f"ARQUIVO: {input_file_path}\n")
        
        log_file.write(f"MOSAICOS SELECIONADOS: {len(selected_mosaics)}\n")
        log_file.write(f"{'-'*50}\n")
        
        all_images = []
        
        for mosaic in selected_mosaics:
            group_id = mosaic.get('group_id', 'unknown')
            images = mosaic.get('images', [])
            
            log_file.write(f"Mosaico: {group_id}\n")
            log_file.write(f"  Imagens ({len(images)}):\n")
            
            for img in images:
                log_file.write(f"    - {img}\n")
                all_images.append(img)
            
            log_file.write(f"{'-'*30}\n")
        
        log_file.write(f"{'='*50}\n")

    download_json_structure = {
        "mosaic_groups": [
            {
                "group_id": f"selected_mosaics_{area_name}",
                "images": all_images
            }
        ]
    }
    
    timestamp_short = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{area_name}_{timestamp_short}.json"

    json_path = os.path.join(os.path.dirname(log_file_path), json_filename)
    
    with open(json_path, 'w') as json_file:
        json.dump(download_json_structure, json_file, indent=2)
    
    print(f"\nGerado arquivo JSON para download: {json_path}")
    print(f"Total de {len(all_images)} imagens incluídas no arquivo JSON")
    print(f'json_path = "{json_path}"')
import os
import datetime
import re
import json

def save_selected_mosaics_log(selected_mosaics, input_file_path=None, log_file_path="selected_mosaics_log.txt", 
                             metrics=None, group_intersections=None, group_shared_ratios=None):
    """
    Salva os mosaicos selecionados em um arquivo de log cumulativo e também gera um JSON
    com informações detalhadas sobre cada mosaico e suas interseções.
    
    Args:
        selected_mosaics: Lista de dicionários com os grupos de mosaicos selecionados
        input_file_path: Caminho do arquivo de input para extrair o nome da área
        log_file_path: Caminho para o arquivo de log
        metrics: Dicionário com as métricas calculadas para cada grupo
        group_intersections: Dicionário com as interseções calculadas entre grupos
        group_shared_ratios: Dicionário com as razões de sobreposição entre grupos
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
            
            # Extrair métricas do mosaico, se disponíveis
            coverage = metrics["group_coverages"].get(group_id, 0) if metrics and "group_coverages" in metrics else 0
            cloud_coverage = metrics["group_cloud_coverages"].get(group_id, 0) if metrics and "group_cloud_coverages" in metrics else 0
            quality = metrics["group_qualities"].get(group_id, 0) if metrics and "group_qualities" in metrics else 0
            
            log_file.write(f"Mosaico: {group_id}\n")
            log_file.write(f"  Cobertura: {coverage:.4f}, Nuvens: {cloud_coverage:.4f}, Qualidade: {quality:.4f}\n")
            log_file.write(f"  Imagens ({len(images)}):\n")
            
            for img in images:
                log_file.write(f"    - {img}\n")
                all_images.append(img)
            
            log_file.write(f"{'-'*30}\n")
        
        log_file.write(f"{'='*50}\n")

    # Criar um único JSON que contenha tanto as informações básicas quanto as detalhadas
    output_json = {
        "metadata": {
            "area_name": area_name,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_mosaics": len(selected_mosaics)
        },
        "mosaic_groups": []
    }
    
    # Adicionar informações detalhadas por mosaico
    for mosaic in selected_mosaics:
        group_id = mosaic.get('group_id', 'unknown')
        images = mosaic.get('images', [])
        
        # Criar objeto de mosaico com métricas e informações
        mosaic_data = {
            "group_id": group_id,
            "images": images,
            "metrics": {
                "coverage": metrics["group_coverages"].get(group_id, 0) if metrics and "group_coverages" in metrics else 0,
                "cloud_coverage": metrics["group_cloud_coverages"].get(group_id, 0) if metrics and "group_cloud_coverages" in metrics else 0,
                "quality": metrics["group_qualities"].get(group_id, 0) if metrics and "group_qualities" in metrics else 0
            },
            "intersections": {}
        }
        
        # Adicionar informações de interseções com outros mosaicos selecionados
        if group_intersections:
            selected_ids = [m.get('group_id') for m in selected_mosaics]
            for other_id in selected_ids:
                if group_id != other_id:
                    # Verificar se existe interseção para este par de mosaicos
                    intersection_value = 0
                    if (group_id, other_id) in group_intersections:
                        intersection_value = group_intersections[(group_id, other_id)]
                    elif (other_id, group_id) in group_intersections:
                        intersection_value = group_intersections[(other_id, group_id)]
                    
                    # Verificar se existe ratio de sobreposição para este par
                    shared_ratio = 0
                    if group_shared_ratios:
                        if (group_id, other_id) in group_shared_ratios:
                            shared_ratio = group_shared_ratios[(group_id, other_id)]
                        elif (other_id, group_id) in group_shared_ratios:
                            shared_ratio = group_shared_ratios[(other_id, group_id)]
                    
                    mosaic_data["intersections"][other_id] = {
                        "intersection_area": intersection_value,
                        "shared_ratio": shared_ratio
                    }
        
        output_json["mosaic_groups"].append(mosaic_data)
    
    # Salvar JSON único com todas as informações
    timestamp_short = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{area_name}_{timestamp_short}.json"
    
    with open(json_filename, 'w') as json_file:
        json.dump(output_json, json_file, indent=2)
    
    print(f"\nGerado arquivo JSON com métricas detalhadas: {json_filename}")
    print(f"Total de {len(all_images)} imagens incluídas no arquivo JSON")
    print(f'json_path = "{json_filename}"')
    
    return json_filename
import os
import json
import traceback
from pprint import pprint

def count_images_in_cplex_outputs(cplex_dir):
    """
    Conta o número de imagens por mosaico nos arquivos de saída do CPLEX em uma pasta.
    
    Args:
        cplex_dir (str): Caminho para a pasta contendo os arquivos de saída do CPLEX.
    """
    if not os.path.exists(cplex_dir):
        print(f"Erro: O diretório {cplex_dir} não existe.")
        return

    # Listar todos os arquivos JSON na pasta
    cplex_files = [f for f in os.listdir(cplex_dir) if f.endswith('.json')]
    if not cplex_files:
        print(f"Nenhum arquivo JSON encontrado no diretório {cplex_dir}.")
        return

    print(f"Processando {len(cplex_files)} arquivos no diretório {cplex_dir}...\n")

    # Contar imagens por mosaico em cada arquivo
    for cplex_file in cplex_files:
        file_path = os.path.join(cplex_dir, cplex_file)
        try:
            with open(file_path, 'r') as f:
                cplex_data = json.load(f)
            
            # Verificar o tipo de dados carregados
            print(f"Arquivo: {cplex_file}")
            if isinstance(cplex_data, list):
                # Processar lista de grupos
                mosaic_image_counts = {}
                total_images = 0
                
                for group in cplex_data:
                    if isinstance(group, dict):
                        group_id = group.get('group_id', group.get('id', 'unknown'))
                        images = group.get('images', [])
                        num_images = len(images)
                        mosaic_image_counts[group_id] = num_images
                        total_images += num_images
                
                # Exibir os resultados para o arquivo atual
                for mosaic, count in mosaic_image_counts.items():
                    print(f"  Mosaico {mosaic}: {count} imagens")
                
                # Calcular estatísticas
                if mosaic_image_counts:
                    print(f"\n  Total de mosaicos: {len(mosaic_image_counts)}")
                    print(f"  Total de imagens: {total_images}")
                    print(f"  Média de imagens por mosaico: {total_images / len(mosaic_image_counts):.2f}")
            
            elif isinstance(cplex_data, dict):
                # Processar dicionário - formato alternativo
                print("  Formato: Dicionário")
                if 'groups' in cplex_data:
                    # Estrutura com 'groups'
                    total_images = 0
                    for group_id, group_info in cplex_data['groups'].items():
                        images = group_info.get('images', [])
                        num_images = len(images)
                        total_images += num_images
                        print(f"  Mosaico {group_id}: {num_images} imagens")
                    
                    # Calcular estatísticas
                    groups = cplex_data['groups']
                    if groups:
                        print(f"\n  Total de mosaicos: {len(groups)}")
                        print(f"  Total de imagens: {total_images}")
                        print(f"  Média de imagens por mosaico: {total_images / len(groups):.2f}")
                
                elif 'mosaic_groups' in cplex_data:
                    # Estrutura com 'mosaic_groups'
                    print("  Processando 'mosaic_groups'...")
                    mosaic_groups = cplex_data['mosaic_groups']
                    
                    total_images = 0
                    
                    # Verificar se mosaic_groups é uma lista ou dicionário
                    if isinstance(mosaic_groups, list):
                        for i, group in enumerate(mosaic_groups):
                            group_id = group.get('id', f"grupo_{i}")
                            images = group.get('images', [])
                            num_images = len(images)
                            total_images += num_images
                            print(f"  Mosaico {group_id}: {num_images} imagens")
                    elif isinstance(mosaic_groups, dict):
                        for group_id, group_info in mosaic_groups.items():
                            images = group_info.get('images', [])
                            num_images = len(images)
                            total_images += num_images
                            print(f"  Mosaico {group_id}: {num_images} imagens")
                    else:
                        print(f"  Formato de mosaic_groups desconhecido: {type(mosaic_groups)}")
                        
                    # Calcular estatísticas
                    total_mosaicos = len(mosaic_groups) if isinstance(mosaic_groups, list) else len(mosaic_groups.keys())
                    print(f"\n  Total de mosaicos: {total_mosaicos}")
                    print(f"  Total de imagens: {total_images}")
                    print(f"  Média de imagens por mosaico: {total_images / total_mosaicos:.2f}")
                    
                else:
                    # Outro formato de dicionário
                    print("  Estrutura do dicionário:")
                    print("  " + str(list(cplex_data.keys())[:5]) + "...")
                    print("  Por favor, verifique o formato do arquivo manualmente.")
            
            else:
                # Outro tipo de dados
                print(f"  Tipo de dados não suportado: {type(cplex_data)}")
                print(f"  Primeiros 100 caracteres: {str(cplex_data)[:100]}...")
            
            print("\n")
        except Exception as e:
            print(f"Erro ao processar o arquivo {cplex_file}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    # Caminho para a pasta com as saídas do CPLEX
    cplex_dir = "/Users/luryand/Documents/encode-image/coverage_otimization/Gulosa-opt"
    
    # Tentar pasta alternativa se a primeira não existir
    if not os.path.exists(cplex_dir):
        cplex_dir = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/PI-PE-CE/cplex"
    
    # Contar imagens por mosaico
    count_images_in_cplex_outputs(cplex_dir)
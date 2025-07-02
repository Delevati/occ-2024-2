"""
Otimização de Seleção de Mosaicos Utilizando CPLEX com Princípio da Inclusão-Exclusão
===================================================================

Este script implementa um modelo de Programação Linear Inteira Mista (PLIM) para
otimizar a seleção de grupos de mosaicos, utilizando a biblioteca docplex do IBM CPLEX.

FLUXO DE PROCESSAMENTO:
-----------------------
1. Execução do script greedy-plots para identificação de imagens e grupos potenciais
2. Pré-processamento com area.py para cálculo de cobertura usando PIE simplificado
3. Execução deste script (3.1-CPLEX.py) para otimização da seleção final de grupos

METODOLOGIA:
-----------
O modelo CPLEX utiliza valores de cobertura geométrica pré-calculados pelo método do 
Princípio da Inclusão-Exclusão (PIE) simplificado, que:

1. Computa precisamente as áreas individuais de cada imagem
2. Calcula as interseções 2a2 entre imagens do mesmo grupo
3. Aplica a fórmula PIE simplificada: A(união) = soma(A_i) - soma(A_i∩A_j)

Esta abordagem é cientificamente precisa e computacionalmente eficiente para o 
cálculo de cobertura em grupos de imagens.

PARÂMETROS DO MODELO:
--------------------
- Função objetivo: Maximiza cobertura e qualidade, penalizando número de grupos e nuvens
- Restrições: Exclusividade de imagens, cobertura mínima (90%), limite de nuvens (50%)
- Pesos: α=0.05 (grupos), γ=0.8 (nuvens)
"""

import os
import json
from docplex.mp.model import Model
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])

METADATA_DIR = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/PI-PE-CE"
OUTPUT_DIR = "/Users/luryand/Documents/encode-image/coverage_otimization/results"
OPTIMIZATION_PARAMS_FILE = os.path.join(METADATA_DIR, 'optimization_parameters-PI-PE-CE-precalc.json')
CPLEX_RESULTS_FILE = os.path.join(OUTPUT_DIR, 'cplex_selected_mosaic_groups-PI-PE-CE-og1g2.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_group_coverage(group):
    """Retorna o valor de cobertura geométrica pré-calculado."""
    if 'geometric_coverage' in group:
        return group['geometric_coverage']
    logging.error(f"Nenhum valor de cobertura encontrado para o grupo {group['group_id']}")
    return 0.0

def solve_mosaic_selection_milp(optimization_params):
    """
    Implementa e resolve o problema de Programação Linear Inteira Mista para seleção de mosaicos.
    
    O modelo maximiza a cobertura total ponderada pela qualidade, penalizando:
    1. O número total de grupos (para minimizar recursos)
    2. A cobertura de nuvens (para maximizar qualidade visual)
    
    Args:
        optimization_params (dict): Parâmetros de otimização incluindo grupos de mosaicos
        
    Returns:
        list: Lista de grupos selecionados pelo modelo CPLEX
    """
    mosaic_groups = optimization_params.get('mosaic_groups', [])
    if not mosaic_groups:
        logging.error("Nenhum grupo de mosaico encontrado nos parâmetros")
        return []

    image_catalog = optimization_params.get('image_catalog', [])
    
    # Criar o modelo CPLEX
    mdl = Model(name='mosaic_selection_optimization')
    
    # Calcular métricas para cada grupo de mosaico
    group_cloud_coverages = {}
    group_coverages = {}
    group_qualities = {}

    logging.info("Analisando métricas dos grupos de mosaico (PIE simplificado)...")
    for group in mosaic_groups:
        group_id = group['group_id']
        images = group['images']
        num_images = len(images)

        # Calcular cobertura de nuvens máxima do grupo (pior caso)
        cloud_coverages = [
            image['cloud_coverage']
            for image_filename in images
            for image in image_catalog
            if image['filename'] == image_filename
        ]
        max_cloud_coverage = max(cloud_coverages) if cloud_coverages else 0
        group_cloud_coverages[group_id] = max_cloud_coverage
        
        # Fator de qualidade do grupo
        group_qualities[group_id] = group.get('quality_factor', 1.0)

        # Cobertura geométrica usando PIE simplificado
        group_coverages[group_id] = calculate_group_coverage(group)
        
        coverage_source = "geométrica"
        
        logging.info(
            f"Grupo {group_id}: {num_images} imagens, "
            f"cobertura {coverage_source}: {group_coverages[group_id]:.4f}, "
            f"nuvens: {max_cloud_coverage:.4f}, qualidade: {group_qualities[group_id]:.4f}"
        )

    # Variáveis de decisão: y[group_id] = 1 se o grupo for selecionado, 0 caso contrário
    y = {
        group['group_id']: mdl.binary_var(name=f'y_{group["group_id"]}')
        for group in mosaic_groups
    }

    # Pesos para a função objetivo
    # alpha = 0.05  # Penalidade por número de grupos
    alpha = 0.4  # Penalidade por número de grupos
    gamma = 0.8   # Penalidade por cobertura de nuvens

    # Componentes da função objetivo
    total_coverage_quality = mdl.sum(
        group_coverages[group['group_id']] * group_qualities[group_id] * y[group['group_id']]
        for group in mosaic_groups if group['group_id'] in y
    )

    penalty_num_groups = alpha * mdl.sum(y[group_id] for group_id in y)
    
    penalty_cloud_coverage = gamma * mdl.sum(
        group_cloud_coverages[group['group_id']] * y[group['group_id']]
        for group in mosaic_groups
        if group['group_id'] in y
    )

    # Definir função objetivo
    mdl.maximize(total_coverage_quality - penalty_num_groups - penalty_cloud_coverage)
    logging.info(f"Função objetivo: Maximizar (Cobertura * Qualidade) - {alpha}(Num Grupos) - {gamma}(Cobertura de Nuvens)")

    # RESTRIÇÕES DO MODELO
    logging.info("Aplicando restrições do modelo...")

    # RESTRIÇÃO 1: Limite de cobertura de nuvens
    cloud_threshold = 0.50
    for group_id, cloud_coverage in group_cloud_coverages.items():
        if cloud_coverage > cloud_threshold and group_id in y:
            mdl.add_constraint(y[group_id] == 0, ctname=f"exclude_high_cloud_{group_id}")

    # RESTRIÇÃO 2: Exclusividade - cada imagem pode aparecer em no máximo um grupo selecionado
    image_to_groups = defaultdict(list)
    for group in mosaic_groups:
        group_id = group['group_id']
        for image_filename in group.get('images', []):
            image_to_groups[image_filename].append(group_id)

    for image_filename, groups_with_image in image_to_groups.items():
        if len(groups_with_image) > 1:
            clean_name = image_filename.replace('.', '_').replace('-', '_')[:30]
            mdl.add_constraint(
                mdl.sum(y[group_id] for group_id in groups_with_image) <= 1,
                ctname=f"exclusivity_{clean_name}"
            )

    # RESTRIÇÃO 3: Cobertura mínima total da AOI considerando interseções entre grupos
    min_total_coverage = 0.85  # 85% de cobertura mínima
    
    # 1. Criar variáveis binárias para rastrear pares de grupos selecionados (o_g1,g2)
    logging.info("Criando variáveis para controlar interseções entre grupos...")
    group_pairs = {}
    for i, group1 in enumerate(mosaic_groups):
        g1_id = group1['group_id']
        for j, group2 in enumerate(mosaic_groups[i+1:], i+1):
            g2_id = group2['group_id']
            pair_name = f"o_{g1_id}_{g2_id}"
            group_pairs[(g1_id, g2_id)] = mdl.binary_var(name=pair_name)
    
    # 2. Adicionar restrições para vincular variáveis de pares às variáveis de grupos
    logging.info("Adicionando restrições lógicas para variáveis de interseção...")
    for (g1_id, g2_id), pair_var in group_pairs.items():
        # o_g1,g2 = 1 somente se ambos y[g1_id] = 1 E y[g2_id] = 1
        mdl.add_constraint(y[g1_id] + y[g2_id] - 1 <= pair_var, 
                          ctname=f"pair_linkage_1_{g1_id}_{g2_id}")
        mdl.add_constraint(pair_var <= y[g1_id], 
                          ctname=f"pair_linkage_2_{g1_id}_{g2_id}")
        mdl.add_constraint(pair_var <= y[g2_id], 
                          ctname=f"pair_linkage_3_{g1_id}_{g2_id}")
    
    # 3. Calcular interseções entre pares de grupos
    group_intersections = {}
    logging.info("Calculando interseções entre grupos de mosaicos...")
    
    # Obter a área total da AOI (ou usar normalizado = 1.0)
    aoi_area = 1.0
    for group in mosaic_groups:
        if 'geometric_coverage_m2' in group and 'geometric_coverage' in group:
            aoi_area = group['geometric_coverage_m2'] / group['geometric_coverage']
            break
    
    logging.info(f"Área da AOI estimada: {aoi_area:.2f} m²")
    
    for i, group1 in enumerate(mosaic_groups):
        g1_id = group1['group_id']
        g1_images = set(group1.get('images', []))
        
        for j, group2 in enumerate(mosaic_groups[i+1:], i+1):
            g2_id = group2['group_id']
            g2_images = set(group2.get('images', []))
            
            # Se não há imagens compartilhadas, a interseção é zero
            shared_images = g1_images.intersection(g2_images)
            if not shared_images:
                group_intersections[(g1_id, g2_id)] = 0
                continue
                
            # Calcular a interseção entre os grupos
            if ('geometric_coverage_m2' in group1 and 'geometric_coverage_m2' in group2):
                # Estimativa da interseção baseada em imagens compartilhadas
                shared_ratio = len(shared_images) / min(len(g1_images), len(g2_images))
                
                # Cálculo conservador da interseção
                smaller_coverage = min(group1['geometric_coverage_m2'], group2['geometric_coverage_m2'])
                intersection_area = smaller_coverage * shared_ratio * 0.7  # Fator de ajuste
                
                # Normalizar pela área da AOI
                group_intersections[(g1_id, g2_id)] = intersection_area / aoi_area
                
                logging.info(f"Interseção entre {g1_id} e {g2_id}: {len(shared_images)} imagens, " +
                           f"área estimada: {intersection_area:.2f} m² ({group_intersections[(g1_id, g2_id)]:.4f} da AOI)")
            else:
                # Estimativa simplificada se não houver dados geométricos disponíveis
                shared_ratio = len(shared_images) / len(g1_images.union(g2_images))
                group_intersections[(g1_id, g2_id)] = min(
                    group_coverages[g1_id], group_coverages[g2_id]
                ) * shared_ratio * 0.5  # Fator conservador
                
                logging.info(f"Interseção estimada entre {g1_id} e {g2_id}: " +
                           f"{group_intersections[(g1_id, g2_id)]:.4f} (baseado em {len(shared_images)} imagens compartilhadas)")
    
    # 4. Aplicar a restrição de cobertura corrigida usando PIE
    logging.info("Aplicando restrição de cobertura com Princípio da Inclusão-Exclusão...")
    
    coverage_expr = mdl.sum(
        group_coverages[group['group_id']] * y[group['group_id']]
        for group in mosaic_groups if group['group_id'] in y
    )
    
    # Subtrair as interseções entre pares de grupos quando ambos são selecionados
    for (g1_id, g2_id), intersection in group_intersections.items():
        if intersection > 0:
            coverage_expr -= intersection * group_pairs[(g1_id, g2_id)]
    
    mdl.add_constraint(coverage_expr >= min_total_coverage, ctname="minimum_total_coverage")

    # Resolver o modelo
    logging.info("Iniciando resolução do modelo CPLEX...")
    solution = mdl.solve()

    # Processar resultados
    selected_groups_details = []
    total_coverage = 0
    total_cloud = 0
    total_quality = 0
    selected_group_ids = []

    if solution:
        logging.info("\n--- SOLUÇÃO ENCONTRADA ---")
        
        for group in mosaic_groups:
            group_id = group['group_id']
            if group_id in y and solution.get_value(y[group_id]) > 0.9:
                selected_groups_details.append(group)
                selected_group_ids.append(group_id)
                coverage = group_coverages[group_id]
                cloud_coverage = group_cloud_coverages.get(group_id, 0)
                quality = group_qualities.get(group_id, 1.0)
                
                total_coverage += coverage
                total_cloud += cloud_coverage
                total_quality += quality
                
                logging.info(
                    f"Grupo {group_id} selecionado: "
                    f"Coverage = {coverage:.4f}, "
                    f"Nuvens = {cloud_coverage:.4f}, "
                    f"Qualidade = {quality:.4f}"
                )
        
        # Calcular cobertura real considerando interseções
        true_coverage = total_coverage
        for (g1_id, g2_id), intersection in group_intersections.items():
            if g1_id in selected_group_ids and g2_id in selected_group_ids:
                true_coverage -= intersection
                logging.info(f"Ajustando cobertura: -{intersection:.4f} (interseção entre {g1_id} e {g2_id})")

        n = len(selected_groups_details)
        logging.info("\n--- MÉTRICAS DA SOLUÇÃO ---")
        logging.info(f"Número de grupos selecionados: {n}")
        logging.info(f"Cobertura total bruta (soma simples): {total_coverage:.4f}")
        logging.info(f"Cobertura total real (PIE): {true_coverage:.4f}")
        logging.info(f"Cobertura média de nuvens: {total_cloud/n if n else 0:.4f}")
        logging.info(f"Qualidade média: {total_quality/n if n else 0:.4f}")
        logging.info(f"IDs dos grupos selecionados: {selected_group_ids}")
    else:
        logging.error("Nenhuma solução viável encontrada pelo CPLEX")

    return selected_groups_details

def save_cplex_results(selected_groups, output_filepath):
    """
    Salva os resultados da otimização CPLEX em um arquivo JSON.
    
    Args:
        selected_groups (list): Lista de grupos selecionados pelo CPLEX
        output_filepath (str): Caminho para o arquivo de saída
    
    Returns:
        bool: True se os resultados foram salvos com sucesso, False caso contrário
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w') as f:
        json.dump(selected_groups, f, indent=2)
    logging.info(f"Resultados da otimização CPLEX salvos em: {output_filepath}")
    return True

def main():
    """
    Função principal que executa o fluxo completo de otimização:
    1. Carrega parâmetros de otimização com valores de cobertura pré-calculados pelo PIE
    2. Executa o modelo CPLEX para selecionar o conjunto ideal de grupos
    3. Salva os resultados no arquivo de saída
    """
    logging.info("=== INICIANDO OTIMIZAÇÃO CPLEX COM MÉTODO PIE SIMPLIFICADO ===")
    
    # Carregar parâmetros de otimização
    logging.info(f"Carregando parâmetros pré-calculados: {OPTIMIZATION_PARAMS_FILE}")
    with open(OPTIMIZATION_PARAMS_FILE, 'r') as f:
        optimization_params = json.load(f)
    
    # Verificar cobertura geométrica
    has_precalc = any('geometric_coverage' in group for group in optimization_params.get('mosaic_groups', []))
    if has_precalc:
        logging.info("✓ Valores de cobertura geométrica encontrados")
    else:
        logging.warning("⚠ Valores de cobertura geométrica não encontrados. Execute area.py primeiro.")
    
    # Resolver o problema de otimização
    selected_mosaic_groups = solve_mosaic_selection_milp(optimization_params)

    # Salvar resultados
    if selected_mosaic_groups:
        save_cplex_results(selected_mosaic_groups, CPLEX_RESULTS_FILE)
    else:
        logging.warning("Nenhum grupo de mosaico selecionado pela otimização")

    logging.info("=== OTIMIZAÇÃO CPLEX CONCLUÍDA ===")

if __name__ == "__main__":
    main()
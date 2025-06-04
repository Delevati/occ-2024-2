"""
Modelo de Otimização para Seleção de Mosaicos com CPLEX
=======================================================

Este script implementa um modelo de otimização matemática usando programação
linear inteira mista (MILP) para selecionar o conjunto ótimo de grupos de mosaicos
que maximizam a cobertura efetiva da área de interesse (AOI).

O modelo utiliza o Princípio da Inclusão-Exclusão (PIE) para calcular corretamente
a cobertura sem duplicação de áreas sobrepostas, considerando a geometria espacial
das imagens na representação da AOI.

Autor: Luryand Costa
Instituição: Universidade Federal
Data: Maio/2025
"""

import os
import json
from docplex.mp.model import Model
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])

METADATA_DIR = "/home/ubuntu/cplex"
OUTPUT_DIR = "/home/ubuntu/cplex/results"
OPTIMIZATION_PARAMS_FILE = os.path.join(METADATA_DIR, 'optimization_parameters-PI-PE-CE-precalc.json')
CPLEX_RESULTS_FILE = os.path.join(OUTPUT_DIR, 'cplex_selected_mosaic_groups-PI-PE-CE-og1g2.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_group_coverage(group):
    """
    Extrai o valor de cobertura geométrica pré-calculada de um grupo de mosaico.
    
    Args:
        group (dict): Dicionário contendo metadados do grupo de mosaico
        
    Returns:
        float: Valor de cobertura geométrica do grupo
    """
    if 'geometric_coverage' in group:
        return group['geometric_coverage']
    logging.error(f"Nenhum valor de cobertura encontrado para o grupo {group['group_id']}")
    return 0.0

def solve_mosaic_selection_milp(optimization_params):
    """
    Implementa e resolve o modelo de programação linear inteira mista (MILP)
    para seleção ótima de grupos de mosaicos.
    
    O modelo considera:
    1. Maximizar a cobertura efetiva considerando sobreposições (PIE)
    2. Minimizar o número de grupos selecionados
    3. Minimizar a cobertura de nuvens
    4. Garantir restrições de exclusividade entre imagens
    
    Args:
        optimization_params (dict): Parâmetros contendo grupos de mosaicos e metadados
        
    Returns:
        tuple: (lista de grupos selecionados, variáveis do modelo para validação)
    """
    global group_intersections
    group_intersections = {}
    
    mosaic_groups = optimization_params.get('mosaic_groups', [])
    if not mosaic_groups:
        logging.error("Nenhum grupo de mosaico encontrado nos parâmetros")
        return [], None

    image_catalog = optimization_params.get('image_catalog', [])
    
    mdl = Model(name='mosaic_selection_optimization')
    
    group_cloud_coverages = {}
    group_coverages = {}
    group_qualities = {}

    logging.info("Analisando métricas dos grupos de mosaico (PIE simplificado)...")
    for group in mosaic_groups:
        group_id = group['group_id']
        images = group['images']
        num_images = len(images)

        cloud_coverages = [
            image['cloud_coverage']
            for image_filename in images
            for image in image_catalog
            if image['filename'] == image_filename
        ]
        max_cloud_coverage = max(cloud_coverages) if cloud_coverages else 0
        group_cloud_coverages[group_id] = max_cloud_coverage
        
        group_qualities[group_id] = group.get('quality_factor', 1.0)
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
    alpha = 0.4  # Penalidade por número de grupos
    gamma = 0.8   # Penalidade por cobertura de nuvens

    # Função objetivo: Maximiza cobertura e qualidade, penalizando número de grupos e nuvens
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

    mdl.maximize(total_coverage_quality - penalty_num_groups - penalty_cloud_coverage)
    logging.info(f"Função objetivo: Maximizar (Cobertura * Qualidade) - {alpha}(Num Grupos) - {gamma}(Cobertura de Nuvens)")

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
    group_pairs = {}
    for i, group1 in enumerate(mosaic_groups):
        g1_id = group1['group_id']
        for j, group2 in enumerate(mosaic_groups[i+1:], i+1):
            g2_id = group2['group_id']
            pair_name = f"o_{g1_id}_{g2_id}"
            group_pairs[(g1_id, g2_id)] = mdl.binary_var(name=pair_name)
    
    # 2. Adicionar restrições para vincular variáveis de pares às variáveis de grupos
    for (g1_id, g2_id), pair_var in group_pairs.items():
        # o_g1,g2 = 1 somente se ambos y[g1_id] = 1 E y[g2_id] = 1
        mdl.add_constraint(y[g1_id] + y[g2_id] - 1 <= pair_var, 
                          ctname=f"pair_linkage_1_{g1_id}_{g2_id}")
        mdl.add_constraint(pair_var <= y[g1_id], 
                          ctname=f"pair_linkage_2_{g1_id}_{g2_id}")
        mdl.add_constraint(pair_var <= y[g2_id], 
                          ctname=f"pair_linkage_3_{g1_id}_{g2_id}")
    
    # 3. Calcular interseções entre pares de grupos
    group_intersections.clear()
    
    # Obter a área total da AOI
    aoi_area = 1.0
    for group in mosaic_groups:
        if 'geometric_coverage_m2' in group and 'geometric_coverage' in group:
            aoi_area = group['geometric_coverage_m2'] / group['geometric_coverage']
            break
    
    logging.info(f"Área da AOI estimada: {aoi_area:.2f} m²")
    
    significant_intersections = []
    
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
                intersection_area = smaller_coverage * shared_ratio

                # Logs de diagnóstico para valores intermediários
                logging.info(f"Diagnóstico interseção {g1_id}-{g2_id}: shared_ratio={shared_ratio:.6f}")
                logging.info(f"Diagnóstico interseção {g1_id}-{g2_id}: smaller_coverage={smaller_coverage:.6f}")
                logging.info(f"Diagnóstico interseção {g1_id}-{g2_id}: intersection_area={intersection_area:.6f}")
                logging.info(f"Diagnóstico interseção {g1_id}-{g2_id}: aoi_area={aoi_area:.6f}")
                
                # Normalizar pela área da AOI
                if aoi_area < 0.001:
                    logging.warning(f"Área AOI muito pequena: {aoi_area}. Usando valor padrão.")
                    aoi_area = 1.0

                intersection_value = intersection_area / aoi_area
                group_intersections[(g1_id, g2_id)] = intersection_value
                
                if intersection_value > 0.01:  # Interseção maior que 1%
                    significant_intersections.append((g1_id, g2_id, intersection_value))

                logging.info(f"Interseção entre {g1_id} e {g2_id}: {len(shared_images)} imagens, " +
                        f"área estimada: {intersection_area:.6f} m² ({intersection_value:.8f} da AOI)")
            else:
                # Estimativa simplificada se não houver dados geométricos disponíveis
                shared_ratio = len(shared_images) / len(g1_images.union(g2_images))
                intersection_value = min(
                    group_coverages[g1_id], group_coverages[g2_id]
                ) * shared_ratio * 0.5  # Fator conservador
                
                group_intersections[(g1_id, g2_id)] = intersection_value
                
                logging.info(f"Interseção estimada entre {g1_id} e {g2_id}: " +
                           f"{intersection_value:.4f} (baseado em {len(shared_images)} imagens compartilhadas)")
    
    # 4. Aplicar a restrição de cobertura
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
        
        # Aplicar o Princípio da Inclusão-Exclusão para subtrair interseções
        for (g1_id, g2_id), intersection in group_intersections.items():
            if g1_id in selected_group_ids and g2_id in selected_group_ids:
                true_coverage -= intersection
                logging.info(f"Ajustando cobertura: -{intersection:.8f} (interseção entre {g1_id} e {g2_id})")
        
        n = len(selected_groups_details)
        logging.info("\n--- MÉTRICAS DA SOLUÇÃO ---")
        logging.info(f"Número de grupos selecionados: {n}")
        logging.info(f"Cobertura total bruta (soma simples): {total_coverage:.4f}")
        logging.info(f"Cobertura total MILP: {true_coverage:.4f}")
        logging.info(f"Cobertura média de nuvens: {total_cloud/n if n else 0:.4f}")
        logging.info(f"Qualidade média: {total_quality/n if n else 0:.4f}")
        logging.info(f"IDs dos grupos selecionados: {selected_group_ids}")
        
        model_vars = {
            "mdl": mdl,
            "solution": solution,
            "y": y,
            "group_pairs": group_pairs,
            "group_coverages": group_coverages,
            "group_cloud_coverages": group_cloud_coverages,
            "group_intersections": group_intersections,
            "selected_group_ids": selected_group_ids,
            "min_total_coverage": min_total_coverage,
            "mosaic_groups": mosaic_groups
        }
        
        return selected_groups_details, model_vars
    else:
        logging.error("Nenhuma solução viável encontrada pelo CPLEX")
        return [], None

def save_cplex_results(selected_groups, output_filepath):
    """
    Salva os resultados da otimização CPLEX em um arquivo JSON.
    
    Args:
        selected_groups (list): Lista de grupos de mosaicos selecionados
        output_filepath (str): Caminho para o arquivo de saída
        
    Returns:
        bool: True se o salvamento foi bem-sucedido
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w') as f:
        json.dump(selected_groups, f, indent=2)
    logging.info(f"Resultados da otimização CPLEX salvos em: {output_filepath}")
    return True

def validate_cplex_decisions(mdl, solution, y, group_pairs, group_coverages, 
                          group_cloud_coverages, group_intersections, 
                          selected_group_ids, min_total_coverage, mosaic_groups):
    """
    Realiza uma validação detalhada das decisões tomadas pelo modelo CPLEX,
    analisando restrições, função objetivo e calculando a cobertura usando
    o método PIE incremental.
    
    Args:
        mdl: Modelo CPLEX
        solution: Solução do modelo
        y: Variáveis de decisão para grupos
        group_pairs: Variáveis de decisão para pares de grupos
        group_coverages: Coberturas dos grupos
        group_cloud_coverages: Cobertura de nuvens dos grupos
        group_intersections: Interseções entre pares de grupos
        selected_group_ids: IDs dos grupos selecionados
        min_total_coverage: Cobertura mínima requerida
        mosaic_groups: Lista de grupos de mosaicos
        
    Returns:
        dict: Relatório detalhado da validação
    """
    logging.info("\n====== VALIDAÇÃO DETALHADA DAS DECISÕES DO CPLEX ======")
    
    validation_report = {
        "objective": {
            "total_value": solution.get_objective_value(),
            "components": {}
        },
        "constraints": {
            "cloud_threshold": [],
            "exclusivity": [],
            "coverage_pie": {},
            "logical_pairs": []
        },
        "group_analysis": {}
    }
    
    # 1. Análise da Função Objetivo
    logging.info("\n1. ANÁLISE DA FUNÇÃO OBJETIVO")
    
    alpha = 0.4
    gamma = 0.8
    obj_coverage_quality = 0
    obj_num_groups_penalty = 0
    obj_cloud_penalty = 0
    
    for group_id in selected_group_ids:
        coverage = group_coverages.get(group_id, 0)
        quality = 1.0
        cloud = group_cloud_coverages.get(group_id, 0)
        
        obj_coverage_quality += coverage * quality
        obj_num_groups_penalty += alpha
        obj_cloud_penalty += gamma * cloud
    
    validation_report["objective"]["components"] = {
        "coverage_quality": obj_coverage_quality,
        "num_groups_penalty": obj_num_groups_penalty,
        "cloud_penalty": obj_cloud_penalty
    }
    
    logging.info(f"Valor total da função objetivo: {solution.get_objective_value():.4f}")
    logging.info(f"+ Cobertura * Qualidade: {obj_coverage_quality:.4f}")
    logging.info(f"- Penalidade por número de grupos: {obj_num_groups_penalty:.4f}")
    logging.info(f"- Penalidade por cobertura de nuvens: {obj_cloud_penalty:.4f}")
    logging.info(f"= Total calculado: {(obj_coverage_quality - obj_num_groups_penalty - obj_cloud_penalty):.4f}")
    
    # 2. Verificação da Restrição de Nuvens
    logging.info("\n2. VERIFICAÇÃO DA RESTRIÇÃO DE NUVENS (LIMITE: 50%)")
    cloud_threshold = 0.5
    
    for group_id, cloud_coverage in group_cloud_coverages.items():
        is_selected = group_id in selected_group_ids
        is_cloud_violated = cloud_coverage > cloud_threshold
        
        if is_cloud_violated:
            status = "REJEITADO (RESTRIÇÃO DE NUVENS)" if not is_selected else "SELECIONADO (VIOLAÇÃO)"
            logging.info(f"Grupo {group_id}: {status} - Cobertura de nuvens: {cloud_coverage:.2%} > {cloud_threshold:.0%}")
            validation_report["constraints"]["cloud_threshold"].append({
                "group_id": group_id,
                "cloud_coverage": cloud_coverage,
                "is_selected": is_selected,
                "violated": is_cloud_violated and is_selected
            })
    
    # 3. Verificação das Restrições de Exclusividade
    logging.info("\n3. VERIFICAÇÃO DAS RESTRIÇÕES DE EXCLUSIVIDADE")
    
    image_to_selected_groups = defaultdict(list)
    for group_id in selected_group_ids:
        group = next((g for g in mosaic_groups if g['group_id'] == group_id), None)
        if group:
            for image_filename in group.get('images', []):
                image_to_selected_groups[image_filename].append(group_id)
    
    exclusivity_violations = 0
    for image_filename, selected_groups in image_to_selected_groups.items():
        if len(selected_groups) > 1:
            exclusivity_violations += 1
            logging.info(f"VIOLAÇÃO DE EXCLUSIVIDADE: Imagem {image_filename} presente em múltiplos grupos: {selected_groups}")
            validation_report["constraints"]["exclusivity"].append({
                "image": image_filename,
                "selected_groups": selected_groups,
                "violated": True
            })
    
    logging.info(f"Violações de exclusividade: {exclusivity_violations}")
    if exclusivity_violations == 0:
        logging.info("✓ Todas as restrições de exclusividade satisfeitas")
    
    # 4. VERIFICAÇÃO DA RESTRIÇÃO DE COBERTURA (PIE INCREMENTAL)
    logging.info("\n4. VERIFICAÇÃO DA RESTRIÇÃO DE COBERTURA (PIE INCREMENTAL)")
    
    # Estimar área da AOI
    aoi_area = 1.0  # Valor padrão
    for group in mosaic_groups:
        if 'geometric_coverage_m2' in group and 'geometric_coverage' in group:
            aoi_area = group['geometric_coverage_m2'] / group['geometric_coverage']
            break
            
    # Demonstrativo de soma subtração de sobreposição de cobertura
    sorted_groups = sorted(
        [(group_id, group_coverages[group_id]) for group_id in selected_group_ids],
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Cálculo incremental - cada grupo adiciona apenas sua área única
    pie_steps = []
    current_covered_groups = set()
    current_pie_area = 0
    
    # Preparar tabela para exibição
    pie_incremental_table = []
    
    for i, (group_id, coverage) in enumerate(sorted_groups):
        # Encontrar o grupo completo
        group = next((g for g in mosaic_groups if g['group_id'] == group_id), None)
        individual_area = coverage * aoi_area
        
        # Para o primeiro grupo, todo o valor é incremento
        if i == 0:
            increment = coverage
            increment_area = individual_area
        else:
            # Para os grupos seguintes, calcular sobreposições com grupos já incluídos
            overlaps_with_previous = 0
            for prev_id in current_covered_groups:
                if (prev_id, group_id) in group_intersections:
                    overlaps_with_previous += group_intersections[(prev_id, group_id)]
                elif (group_id, prev_id) in group_intersections:
                    overlaps_with_previous += group_intersections[(group_id, prev_id)]
            
            # O incremento real é a cobertura menos o que já foi contado
            increment = max(0, coverage - overlaps_with_previous)
            increment_area = increment * aoi_area
        
        # Atualizar o conjunto de grupos cobertos
        current_covered_groups.add(group_id)
        
        # Atualizar a cobertura acumulada
        current_pie_area += increment
        
        # Adicionar à tabela com colunas simplificadas conforme solicitado
        pie_incremental_table.append([
            group_id, 
            f"{coverage*100:.2f}%", 
            f"{increment*100:.2f}%",
            f"{increment_area/1e6:.2f} km²"
        ])
        
        # Registrar este passo para o relatório
        pie_steps.append({
            'group_id': group_id,
            'pie_individual_pct': coverage*100,
            'pie_incremento_pct': increment*100,
            'pie_incremento_km2': increment_area/1e6
        })
    
    # Exibir a tabela no formato solicitado
    logging.info("\n== COBERTURA PIE INCREMENTAL (PAR A PAR) ==")
    logging.info("Mosaicos ordenados por área PIE (maior primeiro):")
    logging.info("{:<10} {:<20} {:<20} {:<20}".format(
        "Mosaico", "PIE Individual %", "Incremento PIE %", "Incremento PIE km²"
    ))
    
    for row in pie_incremental_table:
        logging.info("{:<10} {:<20} {:<20} {:<20}".format(
            row[0], row[1], row[2], row[3]
        ))
    
    # Este valor agora usa o método PIE incremental como o valor correto
    true_coverage = current_pie_area
    coverage_slack = true_coverage - min_total_coverage
    
    # Informações adicionais
    logging.info("\n-- RESUMO DE COBERTURA --")
    logging.info(f"Cobertura final pelo método PIE incremental: {true_coverage*100:.2f}%")
    logging.info(f"Cobertura mínima exigida: {min_total_coverage*100:.2f}%")
    logging.info(f"Folga na restrição: {coverage_slack*100:.2f}%")
    
    status = "✓ SATISFEITA" if true_coverage >= min_total_coverage else "❌ VIOLADA"
    logging.info(f"Restrição de cobertura: {status}")
    
    # Atualizar o relatório de validação
    validation_report["constraints"]["coverage_pie"] = {
        "method": "incremental_pie",
        "true_coverage": true_coverage, 
        "min_required": min_total_coverage,
        "slack": coverage_slack,
        "satisfied": coverage_slack >= 0,
        "pie_steps": pie_steps
    }
    
    # 5. Verificação das Restrições Lógicas de Pares
    logging.info("\n5. VERIFICAÇÃO DAS RESTRIÇÕES LÓGICAS DE PARES")
    
    logical_violations = 0
    for (g1_id, g2_id), pair_var in group_pairs.items():
        is_g1_selected = g1_id in selected_group_ids
        is_g2_selected = g2_id in selected_group_ids
        pair_value = solution.get_value(pair_var)
        
        logical_constraint1 = (is_g1_selected + is_g2_selected - 1) <= pair_value + 0.001
        logical_constraint2 = pair_value <= is_g1_selected + 0.001
        logical_constraint3 = pair_value <= is_g2_selected + 0.001
        
        satisfied = logical_constraint1 and logical_constraint2 and logical_constraint3
        
        if not satisfied:
            logical_violations += 1
            logging.info(f"VIOLAÇÃO LÓGICA: Par ({g1_id}, {g2_id}): y1={is_g1_selected}, y2={is_g2_selected}, o={pair_value:.1f}")
            
        if is_g1_selected or is_g2_selected:
            validation_report["constraints"]["logical_pairs"].append({
                "groups": [g1_id, g2_id],
                "y1": is_g1_selected,
                "y2": is_g2_selected,
                "pair_value": pair_value,
                "satisfied": satisfied
            })
    
    logging.info(f"Violações de restrições lógicas: {logical_violations}")
    if logical_violations == 0:
        logging.info("✓ Todas as restrições lógicas de pares satisfeitas")
    
    # 6. Análise de Grupos
    logging.info("\n6. ANÁLISE DE SELEÇÃO DE GRUPOS")
    
    all_groups = [g['group_id'] for g in mosaic_groups]
    rejected_groups = [g_id for g_id in all_groups if g_id not in selected_group_ids]
    
    logging.info("GRUPOS SELECIONADOS (amostra):")
    for group_id in selected_group_ids[:min(5, len(selected_group_ids))]:
        coverage = group_coverages.get(group_id, 0)
        cloud = group_cloud_coverages.get(group_id, 0)
        quality = 1.0
        
        contribution = coverage * quality - alpha - gamma * cloud
        
        logging.info(f"Grupo {group_id}: Cobertura={coverage:.4f}, Nuvens={cloud:.4f}, Contribuição={contribution:.4f}")
        
        validation_report["group_analysis"][group_id] = {
            "selected": True,
            "coverage": coverage,
            "cloud": cloud,
            "quality": quality,
            "objective_contribution": contribution
        }
    
    logging.info("\nGRUPOS REJEITADOS (amostra):")
    for group_id in rejected_groups[:min(5, len(rejected_groups))]:
        coverage = group_coverages.get(group_id, 0)
        cloud = group_cloud_coverages.get(group_id, 0)
        quality = 1.0
        
        potential_contribution = coverage * quality - alpha - gamma * cloud
        
        reason = "Contribuição negativa para o objetivo" if potential_contribution < 0 else \
                 "Cobertura de nuvens acima do limite" if cloud > 0.5 else \
                 "Provável conflito de exclusividade ou substituído por grupos melhores"
        
        logging.info(f"Grupo {group_id}: Cobertura={coverage:.4f}, Nuvens={cloud:.4f}, " +
                    f"Contribuição potencial={potential_contribution:.4f}, Motivo={reason}")
        
        validation_report["group_analysis"][group_id] = {
            "selected": False,
            "coverage": coverage,
            "cloud": cloud,
            "quality": quality,
            "potential_contribution": potential_contribution,
            "rejection_reason": reason
        }
    
    validation_file = os.path.join(os.path.dirname(CPLEX_RESULTS_FILE), 'cplex_decision_validation.json')
    with open(validation_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logging.info(f"\nRelatório de validação salvo em: {validation_file}")
    logging.info("\n====== VALIDAÇÃO COMPLETA ======")
    
    logging.info("\n7. ESTATÍSTICAS PIE PARA ANÁLISE DIDÁTICA")
    
    # Calcular estatísticas para todos os grupos, não apenas os selecionados
    # Isso mostrará o impacto teórico das interseções
    all_groups_ids = [g['group_id'] for g in mosaic_groups]
    
    # A. Cenário real (sem restrições de exclusividade)
    theoretical_raw_coverage = sum(group_coverages[g_id] for g_id in all_groups_ids)
    
    # B. Cálculo PIE pares (ordem 2)
    theoretical_intersections = 0
    intersections_count = 0
    significant_intersections = 0
    
    for (g1_id, g2_id), intersection_value in group_intersections.items():
        theoretical_intersections += intersection_value
        intersections_count += 1
        if intersection_value > 0.01:  # Interseção significativa (>1%)
            significant_intersections += 1
    
    theoretical_pie_coverage = theoretical_raw_coverage - theoretical_intersections
    
    # C. Estatísticas comparativas
    logging.info("A. ESTATÍSTICAS PIE TEÓRICAS (TODOS OS GRUPOS)")
    logging.info(f"Cobertura bruta (soma simples de todos os grupos): {theoretical_raw_coverage:.4f}")
    logging.info(f"Total de interseções calculadas: {intersections_count}")
    logging.info(f"Interseções significativas (>1%): {significant_intersections}")
    logging.info(f"Soma total das interseções: {theoretical_intersections:.4f}")
    logging.info(f"Cobertura estimada PIE: {theoretical_pie_coverage:.4f}")
    logging.info(f"Impacto das interseções: {(theoretical_intersections/theoretical_raw_coverage)*100:.2f}% da soma bruta")
    
    # D. Impacto das interseções na solução atual
    selected_pairs = [(g1, g2) for (g1, g2) in group_intersections.keys() 
                      if g1 in selected_group_ids and g2 in selected_group_ids]
    
    logging.info("\nB. ESTATÍSTICAS DA SOLUÇÃO CPLEX")
    logging.info(f"Cobertura bruta da solução (soma simples): {sum(group_coverages[g_id] for g_id in selected_group_ids):.4f}")
    logging.info(f"Pares possíveis na solução: {len(selected_pairs)}")
    logging.info(f"Total de interseções na solução: calculadas no PIE incremental")
    logging.info(f"Cobertura real PIE incremental: {true_coverage:.4f}")
    
    # E. Salvar estatísticas em arquivo separado
    pie_stats = {
        "theoretical_analysis": {
            "raw_coverage": theoretical_raw_coverage,
            "total_intersections": theoretical_intersections,
            "pie_coverage": theoretical_pie_coverage,
            "intersections_count": intersections_count,
            "significant_intersections": significant_intersections,
            "intersection_impact_percent": (theoretical_intersections/theoretical_raw_coverage)*100
        },
        "solution_analysis": {
            "raw_coverage": sum(group_coverages[g_id] for g_id in selected_group_ids),
            "pie_coverage": true_coverage,
            "possible_pairs": len(selected_pairs),
            "calculation_method": "PIE incremental"
        }
    }
    
    pie_stats_file = os.path.join(os.path.dirname(CPLEX_RESULTS_FILE), 'cplex_pie_extended_stats.json')
    with open(pie_stats_file, 'w') as f:
        json.dump(pie_stats, f, indent=2)
    
    logging.info(f"\nEstatísticas PIE estendidas salvas em: {pie_stats_file}")

    return validation_report

def main():
    """
    Função principal que coordena o fluxo de otimização:
    1. Carrega parâmetros pré-calculados
    2. Resolve o modelo MILP usando CPLEX
    3. Salva os resultados e realiza validações detalhadas
    """
    logging.info("=== INICIANDO OTIMIZAÇÃO CPLEX COM MÉTODO PIE SIMPLIFICADO ===")
    
    logging.info(f"Carregando parâmetros pré-calculados: {OPTIMIZATION_PARAMS_FILE}")
    with open(OPTIMIZATION_PARAMS_FILE, 'r') as f:
        optimization_params = json.load(f)
    
    has_precalc = any('geometric_coverage' in group for group in optimization_params.get('mosaic_groups', []))
    if has_precalc:
        logging.info("✓ Valores de cobertura geométrica encontrados")
    else:
        logging.warning("⚠ Valores de cobertura geométrica não encontrados. Execute area.py primeiro.")
    
    selected_mosaic_groups, model_vars = solve_mosaic_selection_milp(optimization_params)

    if selected_mosaic_groups:
        save_cplex_results(selected_mosaic_groups, CPLEX_RESULTS_FILE)
        
        if model_vars:
            logging.info("\n=== INICIANDO VALIDAÇÃO DETALHADA DAS DECISÕES ===")
            validation_results = validate_cplex_decisions(
                model_vars["mdl"], 
                model_vars["solution"], 
                model_vars["y"], 
                model_vars["group_pairs"], 
                model_vars["group_coverages"],
                model_vars["group_cloud_coverages"], 
                model_vars["group_intersections"],
                model_vars["selected_group_ids"], 
                model_vars["min_total_coverage"], 
                model_vars["mosaic_groups"]
            )
            
            complete_results = {
                "selected_groups": selected_mosaic_groups,
                "validation": validation_results
            }
            
            complete_results_file = os.path.join(os.path.dirname(CPLEX_RESULTS_FILE), 'cplex_complete_results.json')
            with open(complete_results_file, 'w') as f:
                json.dump(complete_results, f, indent=2)
    else:
        logging.warning("Nenhum grupo de mosaico selecionado pela otimização")

    logging.info("=== OTIMIZAÇÃO CPLEX CONCLUÍDA ===")

if __name__ == "__main__":
    main()
"""
Utilitários de validação para modelos CPLEX
===========================================

Este módulo contém funções para validar e analisar soluções geradas pelo CPLEX
para o problema de seleção de mosaicos de imagens.
"""

import os
import json
import logging
from collections import defaultdict

def validate_cplex_decisions(mdl, solution, y, group_pairs, group_coverages, 
                          group_cloud_coverages, group_intersections, 
                          selected_group_ids, min_total_coverage, mosaic_groups, results_file_path):
    """
    Realiza uma validação detalhada das decisões tomadas pelo modelo CPLEX,
    verificando a consistência da solução e analisando o atendimento às restrições.
    
    Este processo valida matematicamente se a solução encontrada pelo solver:
    1. Satisfaz todas as restrições do modelo
    2. Calcula corretamente o valor da função objetivo
    3. Implementa corretamente o método MILP para cálculo de cobertura
    
    A validação inclui:
    - Análise da função objetivo: decomposição em seus componentes
    - Verificação da restrição de nuvens: nenhum grupo com nuvens > threshold
    - Verificação de exclusividade: cada imagem em no máximo um grupo
    - Verificação da cobertura MILP: usando método incremental para confirmação
    - Validação das restrições lógicas: consistência entre variáveis y_j e o_j,k
    - Análise de decisões: motivos para seleção/rejeição de grupos
    
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
        dict: Relatório detalhado da validação, contendo análises de todas as 
              restrições e componentes do modelo
    """
    validation_report = {
        "objective": {
            "total_value": solution.get_objective_value(),
            "components": {}
        },
        "constraints": {
            "cloud_threshold": [],
            "exclusivity": [],
            "coverage_MILP": {},
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
    logging.info("\n2. VERIFICAÇÃO DA RESTRIÇÃO DE NUVENS ")
    cloud_threshold = 0.4
    
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
    
    # 4. VERIFICAÇÃO DA RESTRIÇÃO DE COBERTURA
    logging.info("\n4. VERIFICAÇÃO DA RESTRIÇÃO DE COBERTURA")

    aoi_area = 1.0
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
    
    # Cálculo - cada grupo adiciona apenas sua área única
    MILP_steps = []
    current_covered_groups = set()
    current_MILP_area = 0
 
    MILP_incremental_table = []
    
    for i, (group_id, coverage) in enumerate(sorted_groups):
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
        current_MILP_area += increment
        
        # Adicionar à tabela com colunas simplificadas conforme solicitado
        MILP_incremental_table.append([
            group_id, 
            f"{coverage*100:.2f}%", 
            f"{increment*100:.2f}%",
            f"{increment_area/1e6:.2f} km²"
        ])
        
        # Registrar este passo para o relatório
        MILP_steps.append({
            'group_id': group_id,
            'MILP_individual_pct': coverage*100,
            'MILP_incremento_pct': increment*100,
            'MILP_incremento_km2': increment_area/1e6
        })

    logging.info("\n== COBERTURA MILP INCREMENTAL (PAR A PAR) ==")
    logging.info("Mosaicos ordenados por área MILP (maior primeiro):")
    logging.info("{:<10} {:<20} {:<20} {:<20}".format(
        "Mosaico", "MILP Individual %", "Incremento MILP %", "Incremento MILP km²"
    ))
    
    for row in MILP_incremental_table:
        logging.info("{:<10} {:<20} {:<20} {:<20}".format(
            row[0], row[1], row[2], row[3]
        ))

    true_coverage = current_MILP_area
    coverage_slack = true_coverage - min_total_coverage

    logging.info("\n-- RESUMO DE COBERTURA --")
    logging.info(f"Cobertura final pelo método MILP incremental: {true_coverage*100:.2f}%")
    logging.info(f"Cobertura mínima exigida: {min_total_coverage*100:.2f}%")
    logging.info(f"Folga na restrição: {coverage_slack*100:.2f}%")
    
    status = "SATISFEITA" if true_coverage >= min_total_coverage else "VIOLADA"
    logging.info(f"Restrição de cobertura: {status}")
    
    # Atualizar o relatório de validação
    validation_report["constraints"]["coverage_MILP"] = {
        "method": "incremental_MILP",
        "true_coverage": true_coverage, 
        "min_required": min_total_coverage,
        "slack": coverage_slack,
        "satisfied": coverage_slack >= 0,
        "MILP_steps": MILP_steps
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
    
    validation_file = os.path.join(os.path.dirname(results_file_path), 'cplex_decision_validation.json')
    with open(validation_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logging.info(f"\nRelatório de validação salvo em: {validation_file}")
    logging.info("\n====== VALIDAÇÃO COMPLETA ======")
    
    logging.info("\n7. ESTATÍSTICAS MILP PARA ANÁLISE DIDÁTICA")
    
    # Calcular estatísticas para todos os grupos, não apenas os selecionados
    # Isso mostrará o impacto teórico das interseções
    all_groups_ids = [g['group_id'] for g in mosaic_groups]
    
    # A. Cenário real (sem restrições de exclusividade)
    theoretical_raw_coverage = sum(group_coverages[g_id] for g_id in all_groups_ids)
    
    # B. Cálculo MILP pares (ordem 2)
    theoretical_intersections = 0
    intersections_count = 0
    significant_intersections = 0
    
    for (g1_id, g2_id), intersection_value in group_intersections.items():
        theoretical_intersections += intersection_value
        intersections_count += 1
        if intersection_value > 0.01:  # Interseção significativa (>1%)
            significant_intersections += 1
    
    theoretical_milp_coverage = theoretical_raw_coverage - theoretical_intersections
    
    # C. Estatísticas comparativas
    logging.info("A. ESTATÍSTICAS MILP TEÓRICAS (TODOS OS GRUPOS)")
    logging.info(f"Cobertura bruta (soma simples de todos os grupos): {theoretical_raw_coverage:.4f}")
    logging.info(f"Total de interseções calculadas: {intersections_count}")
    logging.info(f"Interseções significativas (>1%): {significant_intersections}")
    logging.info(f"Soma total das interseções: {theoretical_intersections:.4f}")
    logging.info(f"Cobertura estimada MILP: {theoretical_milp_coverage:.4f}")
    logging.info(f"Impacto das interseções: {(theoretical_intersections/theoretical_raw_coverage)*100:.2f}% da soma bruta")
    
    # D. Impacto das interseções na solução atual
    selected_pairs = [(g1, g2) for (g1, g2) in group_intersections.keys() 
                      if g1 in selected_group_ids and g2 in selected_group_ids]
    
    logging.info("\nB. ESTATÍSTICAS DA SOLUÇÃO CPLEX")
    logging.info(f"Cobertura bruta da solução (soma simples): {sum(group_coverages[g_id] for g_id in selected_group_ids):.4f}")
    logging.info(f"Pares possíveis na solução: {len(selected_pairs)}")
    logging.info(f"Total de interseções na solução: calculadas no incremental")
    logging.info(f"Cobertura real incremental: {true_coverage:.4f}")
    
    # E. Salvar estatísticas em arquivo
    MILP_stats = {
        "theoretical_analysis": {
            "raw_coverage": theoretical_raw_coverage,
            "total_intersections": theoretical_intersections,
            "MILP_coverage": theoretical_milp_coverage,
            "intersections_count": intersections_count,
            "significant_intersections": significant_intersections,
            "intersection_impact_percent": (theoretical_intersections/theoretical_raw_coverage)*100
        },
        "solution_analysis": {
            "raw_coverage": sum(group_coverages[g_id] for g_id in selected_group_ids),
            "MILP_coverage": true_coverage,
            "possible_pairs": len(selected_pairs),
            "calculation_method": "MILP incremental"
        }
    }
    
    MILP_stats_file = os.path.join(os.path.dirname(results_file_path), 'cplex_MILP_extended_stats.json')
    with open(MILP_stats_file, 'w') as f:
        json.dump(MILP_stats, f, indent=2)
    
    logging.info(f"\nEstatísticas estendidas salvas em: {MILP_stats_file}")

    return validation_report
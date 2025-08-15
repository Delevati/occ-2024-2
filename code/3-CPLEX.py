"""
Modelo de Otimização para Seleção de Mosaicos com CPLEX
=======================================================

Este script implementa a Fase 2 da metodologia híbrida descrita no artigo:
primeiro, uma heurística gulosa (Fase 1, ver Algoritmo~1 do artigo) gera grupos
de mosaicos candidatos, e em seguida este modelo de Programação Linear Inteira
Mista (PLIM) seleciona o subconjunto ótimo de mosaicos, maximizando a cobertura
útil qualificada e penalizando a presença de nuvens, conforme a Equação (1) e
restrições (2)--(5) do artigo.

O modelo utiliza o método de cobertura MILP para calcular a cobertura sem
duplicação de áreas sobrepostas, considerando a geometria espacial dos mosaicos
e restrições de exclusividade de imagens, sobreposição mínima e limite máximo de
mosaicos, conforme detalhado na Seção~3.2 do artigo.
"""
import os
import json
from docplex.mp.model import Model
from collections import defaultdict
import logging
from cplex_utils.save_log import save_selected_mosaics_log

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])

METADATA_DIR = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/MG-SP-RJ/greedy"
OUTPUT_DIR = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/MG-SP-RJ"
OPTIMIZATION_PARAMS_FILE = os.path.join(METADATA_DIR, 'optimization_parameters-MG-SP-RJ-precalc.json')
CPLEX_RESULTS_FILE = os.path.join(OUTPUT_DIR, 'cplex_selected_mosaic_groups-MG-SP-RJ-og1og2.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MIN_TOTAL_COVERAGE = 0.85  # Cobertura restrição 4
theta_sobreposicao = 0.80

def prepare_model_data(optimization_params):
    """
    Extrai e prepara todos os dados necessários para o modelo PLIM (Fase 2).

    Esta função corresponde à preparação dos parâmetros e variáveis descritos na
    Tabela~1 do artigo, a partir dos grupos de mosaicos candidatos gerados pela
    heurística gulosa (Fase 1).

    Args:
        optimization_params (dict): Parâmetros de otimização

    Returns:
        tuple: (lista de grupos de mosaicos, dicionário com métricas)
    """
    mosaic_groups = optimization_params.get('mosaic_groups', [])
    if not mosaic_groups:
        logging.error("Nenhum grupo de mosaico encontrado nos parâmetros")
        return [], {}

    image_catalog = optimization_params.get('image_catalog', [])
    
    # Dicionários para armazenar os parâmetros do modelo
    group_cloud_coverages = {}  # Corresponde a N_j (cobertura de nuvens do grupo j)
    group_coverages = {}        # Corresponde a A_j (área/cobertura do grupo j)
    group_qualities = {}        # Fator para calcular E_j (qualidade efetiva)

    logging.info("Analisando métricas dos grupos de mosaico (MILP simplificado)...")
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
    
    # Mapeamento de imagens para grupos
    image_to_groups = defaultdict(list)
    for group in mosaic_groups:
        group_id = group['group_id']
        for image_filename in group.get('images', []):
            image_to_groups[image_filename].append(group_id)
    
    metrics = {
        "group_cloud_coverages": group_cloud_coverages,
        "group_coverages": group_coverages,
        "group_qualities": group_qualities,
        "image_to_groups": image_to_groups,
        "image_catalog": image_catalog
    }
    
    return mosaic_groups, metrics

def create_optimization_model(mosaic_groups):
    """
    Cria o modelo PLIM (Fase 2) e as variáveis de decisão binárias y_j,
    conforme a Equação (1) e restrição (5) do artigo.

    Args:
        mosaic_groups (list): Lista de grupos de mosaicos

    Returns:
        tuple: (modelo CPLEX, variáveis de decisão y)
    """
    mdl = Model(name='mosaic_selection_optimization')
    
    # Variáveis de decisão: y[group_id] = 1 se o grupo for selecionado, 0 caso contrário
    # Estas variáveis binárias representam as decisões fundamentais do modelo (y_j na formulação matemática)
    y = {
        group['group_id']: mdl.binary_var(name=f'y_{group["group_id"]}')
        for group in mosaic_groups
    }

    return mdl, y 

def define_objective_function(mdl, y, mosaic_groups, metrics):
    """
    Define a função objetivo do modelo PLIM, conforme Equação (1) do artigo:
    max ∑(j∈M) E_j·y_j - γ·∑(j∈M) N_j·y_j

    Onde:
    - E_j: cobertura efetiva qualificada do mosaico j
    - N_j: máxima cobertura de nuvens do mosaico j
    - γ: peso de penalização para nuvens

    Args:
        mdl (Model): Modelo CPLEX
        y (dict): Variáveis de decisão
        mosaic_groups (list): Lista de grupos de mosaicos
        metrics (dict): Métricas dos grupos
    """
    group_coverages = metrics["group_coverages"]
    group_qualities = metrics["group_qualities"] 
    group_cloud_coverages = metrics["group_cloud_coverages"]
    
    # Pesos para a função objetivo
    gamma = 3.7  # Penalidade por cobertura de nuvens

    # Função objetivo:
    # max ∑(j∈M) E_j·y_j - γ·∑(j∈M) N_j·y_j

    # Decomposição da função objetivo em suas três componentes:
    # Termo 1. Benefício: Cobertura efetiva qualificada (E_j·y_j)
    total_coverage_quality = mdl.sum(
        group_coverages[group['group_id']] * group_qualities[group['group_id']] * y[group['group_id']]
        for group in mosaic_groups if group['group_id'] in y
    )
    
    # Termo 3: γ·∑(j∈M) N_j·y_j (Penalidade pela cobertura de nuvens)
    penalty_cloud_coverage = gamma * mdl.sum(
        group_cloud_coverages[group['group_id']] * y[group['group_id']]
        for group in mosaic_groups
        if group['group_id'] in y
    )

    mdl.maximize(total_coverage_quality - penalty_cloud_coverage)
    logging.info(f"Função objetivo: Maximizar (Cobertura * Qualidade) - {gamma}(Cobertura de Nuvens)")

def add_model_constraints(mdl, y, mosaic_groups, metrics, theta_sobreposicao):
    """
    Adiciona as restrições do modelo PLIM, conforme Equações (2)--(4) do artigo:

    - Restrição (2): Limite máximo de mosaicos selecionados
    - Restrição (3): Exclusividade de imagens (cada imagem em no máximo um mosaico)
    - Restrição (4): Impede seleção conjunta de mosaicos com sobreposição < θ

    Args:
        mdl (Model): Modelo CPLEX
        y (dict): Variáveis de decisão
        mosaic_groups (list): Lista de grupos de mosaicos
        metrics (dict): Métricas dos grupos

    Returns:
        dict: Variáveis auxiliares para pares de grupos (não utilizadas nesta versão)
    """
    group_coverages = metrics["group_coverages"]
    group_cloud_coverages = metrics["group_cloud_coverages"]
    image_to_groups = metrics["image_to_groups"]

    # RESTRIÇÃO 1: Exclui grupos com cobertura de nuvens acima do threshold.
    # Para cada grupo j, se N_j > threshold, então y_j = 0.
    cloud_threshold = 0.40
    for group_id, cloud_coverage in group_cloud_coverages.items():
        if cloud_coverage > cloud_threshold and group_id in y:
            mdl.add_constraint(y[group_id] == 0, ctname=f"exclude_high_cloud_{group_id}")


    # RESTRIÇÃO 2: Limite máximo de mosaicos selecionados.
    # Garante que no máximo N_max mosaicos podem ser escolhidos.
    # ∑_{j ∈ M} y_j ≤ 6
    mdl.add_constraint(mdl.sum(y[group_id] for group_id in y) <= 6, ctname="max_num_groups")

    # RESTRIÇÃO 3: Exclusividade de imagem.
    # Garante que uma mesma imagem não seja utilizada em diferentes grupos na solução final
    # ∑(j∈M(i)) y_j ≤ 1, ∀i ∈ I'
    for image_filename, groups_with_image in image_to_groups.items():
        if len(groups_with_image) > 1:
            clean_name = image_filename.replace('.', '_').replace('-', '_')[:30]
            mdl.add_constraint(
                mdl.sum(y[group_id] for group_id in groups_with_image) <= 1,
                ctname=f"exclusivity_{clean_name}"
            )
    
    # Calcular interseções entre pares de grupos (I_{j,k})
    global group_intersections
    group_intersections = {}
    for i, group1 in enumerate(mosaic_groups):
        g1_id = group1['group_id']
        coverage1 = group1.get('geometric_coverage', 0)
        for j, group2 in enumerate(mosaic_groups[i+1:], i+1):
            g2_id = group2['group_id']
            coverage2 = group2.get('geometric_coverage', 0)
            intersection_value = min(coverage1, coverage2)
            group_intersections[(g1_id, g2_id)] = intersection_value
    
    # RESTRIÇÃO 4: Sobreposição mínima entre mosaicos selecionados
    for (g1_id, g2_id), intersection in group_intersections.items():
        if intersection < theta_sobreposicao:
            mdl.add_constraint(y[g1_id] + y[g2_id] <= 1, ctname=f"min_overlap_{g1_id}_{g2_id}")

    return {}

def solve_and_extract_results(mdl, y, mosaic_groups, metrics, group_pairs):
    """
    Resolve o modelo PLIM (Fase 2) e extrai os grupos de mosaicos selecionados,
    calculando as métricas finais conforme apresentado nas Tabelas~3 e~4 do artigo.

    Args:
        mdl (Model): Modelo CPLEX
        y (dict): Variáveis de decisão
        mosaic_groups (list): Lista de grupos de mosaicos
        metrics (dict): Métricas dos grupos
        group_pairs (dict): Variáveis para pares de grupos

    Returns:
        tuple: (grupos selecionados, métricas da solução)
    """
    group_coverages = metrics["group_coverages"]
    group_cloud_coverages = metrics["group_cloud_coverages"]
    group_qualities = metrics["group_qualities"]

    mdl.context.solver.log_output = True
    mdl.context.solver.warning_level = 0

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
        
        true_coverage = total_coverage
        
        # Aplicar a subtração das interseções
        global group_intersections
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
            "group_qualities": group_qualities,
            "selected_group_ids": selected_group_ids,
            "mosaic_groups": mosaic_groups
        }
        
        return selected_groups_details, model_vars
    else:
        logging.error("Nenhuma solução viável encontrada pelo CPLEX")
        return [], None

def calculate_group_coverage(group):
    """
    Extrai o valor de cobertura geométrica pré-calculada de um grupo de mosaico,
    conforme definido como A_j na Tabela~1 do artigo.

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
    Implementa e resolve o modelo PLIM (Fase 2) para seleção ótima de grupos de mosaicos,
    conforme metodologia do artigo (Seção~3.2).

    Args:
        optimization_params (dict): Parâmetros contendo grupos de mosaicos e metadados

    Returns:
        tuple: (lista de grupos selecionados, variáveis do modelo para validação)
    """
    global group_intersections
    group_intersections = {}
    
    # 1. Preparar dados
    mosaic_groups, metrics = prepare_model_data(optimization_params)
    if not mosaic_groups:
        return [], None
    
    # 2. Criar modelo e variáveis
    mdl, y = create_optimization_model(mosaic_groups)
    
    # 3. Definir função objetivo
    define_objective_function(mdl, y, mosaic_groups, metrics)
    
    # 4. Adicionar restrições
    group_pairs = add_model_constraints(mdl, y, mosaic_groups, metrics)
    
    # 5. Resolver modelo e processar resultados
    selected_groups, model_vars = solve_and_extract_results(mdl, y, mosaic_groups, metrics, group_pairs)
    
    return selected_groups, model_vars

def save_cplex_results(selected_groups, output_filepath):
    """
    Salva os resultados da otimização PLIM (Fase 2) em um arquivo JSON,
    conforme apresentado nas tabelas de resultados do artigo.

    Args:
        selected_groups (list): Lista de grupos de mosaicos selecionados
        output_filepath (str): Caminho para o arquivo de saída
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w') as f:
        json.dump(selected_groups, f, indent=2)
    logging.info(f"Resultados da otimização CPLEX salvos em: {output_filepath}")
    return True

def main():
    """
    Função principal: executa a Fase 2 da metodologia híbrida do artigo.
    1. Carrega parâmetros e grupos candidatos da heurística gulosa (Fase 1)
    2. Resolve o modelo PLIM (Fase 2)
    3. Salva e valida os resultados conforme métricas do artigo
    """
    logging.info("=== INICIANDO OTIMIZAÇÃO CPLEX COM MÉTODO ===")
    
    logging.info(f"Carregando parâmetros pré-calculados: {OPTIMIZATION_PARAMS_FILE}")
    with open(OPTIMIZATION_PARAMS_FILE, 'r') as f:
        optimization_params = json.load(f)
    
    has_precalc = any('geometric_coverage' in group for group in optimization_params.get('mosaic_groups', []))
    if has_precalc:
        logging.info("Valores de cobertura geométrica encontrados")
    else:
        logging.warning("Valores de cobertura geométrica não encontrados. Execute area.py primeiro.")
    
    selected_mosaic_groups, model_vars = solve_mosaic_selection_milp(optimization_params)

    if selected_mosaic_groups:
        save_cplex_results(selected_mosaic_groups, CPLEX_RESULTS_FILE)
        metrics = {
            "group_coverages": model_vars["group_coverages"],
            "group_cloud_coverages": model_vars["group_cloud_coverages"],
            "group_qualities": model_vars.get("group_qualities", {})
        }
        
        # Chamar save_selected_mosaics_log com os parâmetros adicionais
        save_selected_mosaics_log(
            selected_mosaic_groups, 
            input_file_path=OPTIMIZATION_PARAMS_FILE,
            metrics=metrics,
            group_intersections=model_vars["group_intersections"],
        )

    logging.info("=== OTIMIZAÇÃO CPLEX CONCLUÍDA ===")

if __name__ == "__main__":
    main()
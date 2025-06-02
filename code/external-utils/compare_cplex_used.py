#!/usr/bin/env python3
# filepath: validate_pie_approximation.py

import json
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from itertools import combinations

def calculate_pie_coverage_for_selected_groups(precalc_file, cplex_file):
    """
    Calcula a cobertura PIE apenas para os grupos selecionados pelo CPLEX
    """
    try:
        # Carregar os dados do precalc
        with open(precalc_file, 'r') as f:
            precalc_data = json.load(f)
        
        # Carregar os grupos selecionados pelo CPLEX
        with open(cplex_file, 'r') as f:
            cplex_groups = json.load(f)
        
        # Carregar a área total da AOI (devemos obter do precalc)
        total_aoi_area = 1.0  # Valor padrão para caso não exista
        if 'total_aoi_area' in precalc_data:
            total_aoi_area = precalc_data['total_aoi_area']
        
        # Extrair os IDs dos grupos selecionados pelo CPLEX
        cplex_group_ids = []
        for group in cplex_groups:
            if 'id' in group:
                cplex_group_ids.append(group['id'])
            elif 'group_id' in group:
                cplex_group_ids.append(group['group_id'])
        
        print(f"Grupos selecionados pelo CPLEX: {len(cplex_group_ids)}")
        
        # Encontrar os mesmos grupos no arquivo precalc
        selected_precalc_groups = []
        for group in precalc_data.get('mosaic_groups', []):
            if 'group_id' in group and group['group_id'] in cplex_group_ids:
                selected_precalc_groups.append(group)
        
        print(f"Grupos correspondentes encontrados no precalc: {len(selected_precalc_groups)}")
        
        if not selected_precalc_groups:
            print("ERRO: Não foi possível encontrar os grupos CPLEX no arquivo precalc")
            return None
        
        # Para cada grupo, usar sua cobertura geométrica já normalizada
        individual_coverages = []
        for group in selected_precalc_groups:
            if 'geometric_coverage' in group:
                individual_coverages.append(group['geometric_coverage'])
        
        # Verificar as estruturas de dados para debugging
        print("\nEstrutura de um grupo do precalc:")
        sample_group = selected_precalc_groups[0]
        for key in sample_group.keys():
            print(f"- {key}: {type(sample_group[key])}")
        
        # Verificar estrutura de interseções
        if 'group_intersections' in precalc_data:
            print("\nEstrutura de interseções:")
            # Mostrar 5 primeiras chaves de exemplo
            sample_keys = list(precalc_data['group_intersections'].keys())[:5]
            for key in sample_keys:
                print(f"- {key}: {precalc_data['group_intersections'][key]}")
        
        # Se temos cobertura geométrica normalizada, usamos
        if individual_coverages:
            # Usar a fórmula de PIE: máximo entre coberturas individuais
            pie_coverage = max(individual_coverages)
            print(f"\nUsando cobertura geométrica normalizada: {pie_coverage:.4f}")
            return pie_coverage * 100  # Converter para percentual
        
        # Se não houver coberturas normalizadas, tentamos calcular
        print("\nTentando calcular PIE a partir das áreas absolutas...")
        
        # Calcular a soma das áreas individuais
        total_individual_area = 0
        for group in selected_precalc_groups:
            if 'total_individual_area' in group:
                total_individual_area += group['total_individual_area']
        
        # Calcular interseções 2a2
        total_pairwise_overlap = 0
        intersection_count = 0
        
        # Iterar pelas combinações de grupos
        for i in range(len(selected_precalc_groups)):
            for j in range(i+1, len(selected_precalc_groups)):
                g1_id = selected_precalc_groups[i]['group_id']
                g2_id = selected_precalc_groups[j]['group_id']
                
                # Tentar várias possibilidades de chaves
                intersection_found = False
                
                # Formato "g1_id,g2_id"
                key = f"{g1_id},{g2_id}"
                reverse_key = f"{g2_id},{g1_id}"
                
                if 'group_intersections' in precalc_data:
                    if key in precalc_data['group_intersections']:
                        total_pairwise_overlap += precalc_data['group_intersections'][key]
                        intersection_found = True
                        intersection_count += 1
                    elif reverse_key in precalc_data['group_intersections']:
                        total_pairwise_overlap += precalc_data['group_intersections'][reverse_key]
                        intersection_found = True
                        intersection_count += 1
                
                # Se não encontrou, tentar outros formatos
                if not intersection_found:
                    # Tenta encontrar na estrutura do grupo
                    for group in selected_precalc_groups:
                        if 'intersections' in group:
                            if g1_id in group['intersections'] and g2_id in group['intersections'][g1_id]:
                                total_pairwise_overlap += group['intersections'][g1_id][g2_id]
                                intersection_found = True
                                intersection_count += 1
                                break
        
        print(f"Interseções encontradas: {intersection_count}")
        
        # Calcular PIE e normalizar
        pie_coverage = total_individual_area - total_pairwise_overlap
        normalized_pie = min(pie_coverage / total_aoi_area if total_aoi_area > 0 else 0, 1.0)
        
        print("\nCálculo PIE para grupos selecionados pelo CPLEX:")
        print(f"Área individual total: {total_individual_area:.6f}")
        print(f"Sobreposição 2a2 total: {total_pairwise_overlap:.6f}")
        print(f"Cobertura PIE calculada: {pie_coverage:.6f}")
        print(f"Área total AOI: {total_aoi_area:.6f}")
        print(f"Cobertura normalizada: {normalized_pie:.6f}")
        
        return normalized_pie * 100  # Converter para percentual
        
    except Exception as e:
        print(f"Erro ao calcular PIE para grupos selecionados: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_coverage_methods():
    # Base directories
    base_dir = "/Users/luryand/Documents/encode-image/coverage_otimization"
    cplex_dir = f"{base_dir}/studio/results"
    recapture_dir = f"{base_dir}/code/APA-input/recapture"
    
    # Regiões
    regions = ["AL", "BA", "MG", "MG-SP-RJ", "PI-PE-CE", "RS"]
    
    # Mapear arquivos de AOI
    aoi_files = {
        "RS": f"{recapture_dir}/RS/ibirapuita_31982.shp",
        "PI-PE-CE": f"{recapture_dir}/PI-PE-CE/ucs_pe-pi-ce_31984.shp",
        "MG-SP-RJ": f"{recapture_dir}/MG-SP-RJ/APA_Mantiqueira_mgrjsp_31983.shp",
        "MG": f"{recapture_dir}/MG/ucs_pantanal_mg_31981.shp",
        "BA": f"{recapture_dir}/BA/ucs_bahia_31984.shp",
        "AL": f"{recapture_dir}/AL/multi-polygon-reproj.shp"
    }
    
    
    results = []
    
    # Para cada região
    for region in regions:
        print(f"\n===== Processando região: {region} =====")
        
        # Arquivos necessários
        precalc_file = f"{recapture_dir}/{region}/optimization_parameters-{region}-precalc.json"
        cplex_file = f"{cplex_dir}/cplex_selected_mosaic_groups-{region}.json"
        aoi_file = aoi_files[region]
        
        # Verificar se os arquivos existem
        if not all(os.path.exists(f) for f in [precalc_file, cplex_file, aoi_file]):
            print(f"Arquivos necessários não encontrados para {region}")
            continue
            
        # Obter cobertura PIE para os grupos selecionados pelo CPLEX
        pie_coverage = calculate_pie_coverage_for_selected_groups(precalc_file, cplex_file)
        if pie_coverage is None:
            print(f"Não foi possível calcular cobertura PIE para {region}")
            continue
        
    # Criar DataFrame e salvar resultados
    if results:
        df = pd.DataFrame(results)
        df.to_csv("pie_validation_results.csv", index=False)
        
        # Ajustar valores nulos para visualização
        df['difference'] = df['difference'].fillna(0)
        
        # Visualizar resultados
        plt.figure(figsize=(12, 8))
        
        # Criar barras agrupadas
        regions = df['region'].tolist()
        x = range(len(regions))
        
        pie_values = df['pie_coverage'].tolist()
        ref_values = [v if v is not None else 0 for v in df['reference_coverage'].tolist()]
        
        # Plotar barras
        plt.bar([i-0.2 for i in x], pie_values, width=0.4, label='PIE (2a2)', color='steelblue')
        plt.bar([i+0.2 for i in x], ref_values, width=0.4, label='Cobertura Final', color='darkorange')
        
        # Adicionar texto com os valores das diferenças
        for i, diff in enumerate(df['difference']):
            plt.text(i, min(pie_values[i], ref_values[i]) - 8, 
                     f"{abs(diff):.2f}pp", 
                     ha='center', 
                     fontweight='bold',
                     color='darkred' if abs(diff) > 1 else 'darkgreen')
        
        # Formatar o gráfico
        plt.xlabel('Região', fontsize=12)
        plt.ylabel('Cobertura (%)', fontsize=12)
        plt.title('Validação do PIE: Comparação entre PIE (2a2) e Cobertura Final', fontsize=14)
        plt.xticks(x, regions, fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 110)  # Garantir espaço para os rótulos
        
        # Adicionar estatísticas
        mean_diff = df['difference'].mean()
        max_diff = df['difference'].max()
        plt.figtext(0.5, 0.01, 
                   f"Diferença média: {mean_diff:.2f}pp | Diferença máxima: {max_diff:.2f}pp", 
                   ha="center", fontsize=11, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.savefig('pie_validation_chart.png', dpi=300)
        print("\nResultados salvos em pie_validation_results.csv")
        print("Gráfico salvo em pie_validation_chart.png")

if __name__ == "__main__":
    compare_coverage_methods()
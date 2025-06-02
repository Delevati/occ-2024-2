import json
import os
import logging
import geopandas as gpd
import rasterio
from shapely.geometry import box
from shapely.ops import unary_union
from tabulate import tabulate
import pyproj
from pathlib import Path
from functools import partial
from shapely.ops import transform
import zipfile
import shutil
import itertools

# Configuração
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
JSON_FILE = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/MG-SP-RJ/cplex_selected_mosaic_groups-MG-SP-RJ-og1g2.json"
PRECALC_FILE = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/MG-SP-RJ/greedy/optimization_parameters-MG-SP-RJ-precalc.json"
SHP_FILE = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/MG-SP-RJ/APA_Mantiqueira_mgrjsp_31983.shp"
IMAGE_DIR = Path("/Volumes/luryand/nova_busca")
TEMP_DIR = Path("/Volumes/luryand/temp_mosaics")
OUTPUT_DIR = "/Users/luryand/Documents/encode-image/coverage_otimization/results"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_image_path(image_name):
    base_name = image_name.replace('.zip', '').replace('.SAFE', '')
    for variant in [f"{base_name}.zip", f"{base_name}.SAFE.zip", f"{base_name}.SAFE"]:
        direct_path = IMAGE_DIR / variant
        if direct_path.exists():
            return str(direct_path)
        for root, dirs, files in os.walk(IMAGE_DIR):
            path = Path(root) / variant
            if path.exists():
                return str(path)
    logging.warning(f"Imagem não encontrada: {image_name}")
    return None

def get_tci_path(image_name, temp_dir):
    image_path = find_image_path(image_name)
    if not image_path:
        return None
    if image_path.endswith('.zip'):
        try:
            with zipfile.ZipFile(image_path, 'r') as zip_ref:
                tci_files = [f for f in zip_ref.namelist() if 'TCI_10m.jp2' in f]
                if not tci_files:
                    return None
                tci_file = tci_files[0]
                extract_path = temp_dir / Path(os.path.basename(tci_file))
                with zip_ref.open(tci_file) as source, open(extract_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
                return str(extract_path) if extract_path.exists() else None
        except Exception as e:
            logging.error(f"Erro ao extrair TCI de {image_path}: {e}")
            return None
    else:
        tci_files = list(Path(image_path).glob("**/TCI_10m.jp2"))
        return str(tci_files[0]) if tci_files else None

def get_image_geometry(image_name, temp_dir, aoi_crs):
    tci_path = get_tci_path(image_name, temp_dir)
    if not tci_path:
        return None
    try:
        with rasterio.open(tci_path) as src:
            bounds = src.bounds
            source_crs = src.crs.to_string()
            poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            if source_crs != aoi_crs:
                project = pyproj.Transformer.from_crs(
                    pyproj.CRS.from_user_input(source_crs),
                    pyproj.CRS.from_user_input(aoi_crs),
                    always_xy=True
                ).transform
                poly = transform(project, poly)
                if not poly.is_valid:
                    poly = poly.buffer(0)
            return poly if poly.is_valid and poly.area > 0 else None
    except Exception as e:
        logging.error(f"Erro ao processar {image_name}: {e}")
        return None
    finally:
        if tci_path and os.path.exists(tci_path):
            os.remove(tci_path)

def calculate_pairwise_overlaps(mosaic_geometries):
    pairwise_overlaps = []
    n = len(mosaic_geometries)
    for i, j in itertools.combinations(range(n), 2):
        geom_i = mosaic_geometries[i]
        geom_j = mosaic_geometries[j]
        if geom_i is not None and geom_j is not None:
            intersection = geom_i.intersection(geom_j)
            if not intersection.is_empty:
                overlap_area = intersection.area
                pairwise_overlaps.append((i, j, overlap_area))
    return pairwise_overlaps

def calculate_incremental_coverage(mosaic_geometries, mosaic_ids, aoi_area):
    """
    Calcula a cobertura total usando uma abordagem incremental:
    1. Começa com o primeiro mosaico
    2. Adiciona o segundo mosaico e calcula a união
    3. Adiciona o terceiro à união dos dois primeiros, e assim por diante
    """
    if not mosaic_geometries or all(g is None for g in mosaic_geometries):
        return 0, []
    
    # Filtrar mosaicos válidos
    valid_mosaicos = [(i, g) for i, g in enumerate(mosaic_geometries) if g is not None]
    if not valid_mosaicos:
        return 0, []
    
    # Começar com o primeiro mosaico
    current_union = valid_mosaicos[0][1]
    current_coverage = 100 * current_union.area / aoi_area
    
    # Lista para rastrear o incremento de cada mosaico
    coverage_steps = [
        {
            'mosaico': mosaic_ids[valid_mosaicos[0][0]],
            'cobertura_individual': current_coverage,
            'cobertura_acumulada': current_coverage,
            'incremento': current_coverage
        }
    ]
    
    # Adicionar mosaicos um a um
    for idx, geom in valid_mosaicos[1:]:
        # Cobertura individual do mosaico atual
        individual_coverage = 100 * geom.area / aoi_area
        
        # Cobertura anterior
        previous_coverage = current_coverage
        
        # Adicionar à união atual
        current_union = current_union.union(geom)
        current_coverage = 100 * current_union.area / aoi_area
        
        # Incremento real que este mosaico adicionou
        increment = current_coverage - previous_coverage
        
        coverage_steps.append({
            'mosaico': mosaic_ids[idx],
            'cobertura_individual': individual_coverage,
            'cobertura_acumulada': current_coverage,
            'incremento': increment
        })
    
    return current_coverage, coverage_steps

def calculate_pie_incremental(pie_areas, mosaic_geometries, mosaic_ids, aoi_area, pairwise_overlaps):
    """
    Calcula o PIE de forma incremental corrigida
    """
    if not mosaic_geometries or all(g is None for g in mosaic_geometries):
        return 0, []
    
    # Combinar os dados de PIE, IDs e geometrias
    combined_data = [(i, pie_area, geom) for i, (pie_area, geom) in 
                     enumerate(zip(pie_areas, mosaic_geometries)) if geom is not None]
    
    # Ordenar por área PIE (maior primeiro)
    combined_data.sort(key=lambda x: x[1], reverse=True)
    
    # Lista para registrar cada passo
    pie_steps = []
    
    # Começamos com um conjunto vazio de cobertura
    current_union = None
    
    # Processar cada mosaico incrementalmente
    for idx, pie_area, geom in combined_data:
        # Área PIE individual do mosaico atual
        individual_pie_pct = 100 * pie_area / aoi_area
        
        # Calcular o incremento real utilizando a união geométrica
        if current_union is None:
            # Primeiro mosaico - toda a área é incremento
            increment = pie_area
            current_union = geom  # Geometrias Shapely são imutáveis, não precisa de .copy()
        else:
            # Para os outros mosaicos, calculamos apenas a nova área adicionada
            old_area = current_union.area
            current_union = current_union.union(geom)
            new_area = current_union.area
            increment = new_area - old_area
        
        # Percentuais e valores em km²
        increment_pct = 100 * increment / aoi_area
        current_coverage_pct = 100 * current_union.area / aoi_area
        
        # Registrar este passo
        pie_steps.append({
            'mosaico': mosaic_ids[idx],
            'pie_individual_pct': individual_pie_pct,
            'pie_incremento_pct': increment_pct,
            'pie_acumulado_pct': current_coverage_pct,
            'pie_individual_km2': pie_area / 1e6,
            'pie_incremento_km2': increment / 1e6,
            'pie_acumulado_km2': current_union.area / 1e6
        })
    
    final_coverage_pct = 100 * current_union.area / aoi_area if current_union else 0
    return final_coverage_pct, pie_steps

def compare_pie_sum_vs_union_sum(mosaic_indices=None):
    # 1. CARREGAR DADOS DO CPLEX
    with open(JSON_FILE, 'r') as f:
        mosaics = json.load(f)
    precalc_data = None
    precalc_groups = {}
    try:
        with open(PRECALC_FILE, 'r') as f:
            precalc_data = json.load(f)
            precalc_groups = {
                group.get('group_id', f"unknown_{i}"): group 
                for i, group in enumerate(precalc_data.get('mosaic_groups', []))
            }
            logging.info(f"Arquivo pré-calculado carregado com {len(precalc_groups)} grupos")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Não foi possível carregar o arquivo pré-calculado: {e}")

    aoi_gdf = gpd.read_file(SHP_FILE)
    aoi_area = aoi_gdf.union_all().area
    aoi_crs = aoi_gdf.crs.to_string()

    # Selecionar mosaicos
    if mosaic_indices is None:
        selected_mosaics = mosaics
        mosaic_ids = [m.get('group_id', f'mosaic_{i}') for i, m in enumerate(mosaics)]
    else:
        selected_mosaics = [mosaics[idx] for idx in mosaic_indices if 0 <= idx < len(mosaics)]
        mosaic_ids = [selected_mosaics[i].get('group_id', f'mosaic_{idx}') for i, idx in enumerate(mosaic_indices)]
    logging.info(f"Processando {len(selected_mosaics)} mosaicos")

    # 2. PEGAR ÁREAS PIE (A_j) JÁ DESCONTANDO SOBREPOSIÇÕES INTERNAS
    pie_areas = []
    for i, mosaic in enumerate(selected_mosaics):
        group_id = mosaic.get('group_id', f'mosaic_{i}')
        pie_area = mosaic.get('pie_coverage_area')
        if pie_area is None:
            if group_id in precalc_groups:
                pie_area = precalc_groups[group_id].get('pie_coverage_area')
                if pie_area is None:
                    pie_area = precalc_groups[group_id].get('geometric_coverage_m2', 0)
                    logging.info(f"Usando geometric_coverage_m2 do pré-calculado para {group_id}")
                else:
                    logging.info(f"Usando pie_coverage_area do pré-calculado para {group_id}")
            else:
                pie_area = mosaic.get('geometric_coverage_m2', 0)
        pie_areas.append(pie_area)

    # 3. UNIÃO REAL DE CADA MOSAICO (para calcular I_jk)
    union_areas = []
    mosaic_geometries = []
    mosaic_table = []
    for i, mosaic in enumerate(selected_mosaics):
        group_id = mosaic_ids[i]
        images = mosaic.get('images', [])
        if not images and group_id in precalc_groups:
            images = precalc_groups[group_id].get('images', [])
            if images:
                logging.info(f"Usando imagens do arquivo pré-calculado para {group_id}")
        logging.info(f"Calculando união para mosaico {group_id} com {len(images)} imagens")
        image_geometries = []
        for image_name in images:
            geom = get_image_geometry(image_name, TEMP_DIR, aoi_crs)
            if geom:
                clipped_geom = geom.intersection(aoi_gdf.union_all())
                if not clipped_geom.is_empty and clipped_geom.area > 0:
                    image_geometries.append(clipped_geom)
        if image_geometries:
            mosaic_union = unary_union(image_geometries)
            mosaic_geometries.append(mosaic_union)
            mosaic_union_area = mosaic_union.area
        else:
            mosaic_geometries.append(None)
            mosaic_union_area = 0
        union_areas.append(mosaic_union_area)
        
    # Corrigir valores de PIE se estiverem zerados
    if sum(pie_areas) == 0:
        logging.warning("Valores de PIE zerados. Usando áreas de união como valores PIE")
        pie_areas = union_areas.copy()
    
    # Gerar tabela de mosaicos individuais
    for i, mosaic in enumerate(selected_mosaics):
        group_id = mosaic_ids[i]
        pie_area = pie_areas[i]
        mosaic_union_area = union_areas[i]
        # Percentual em relação à AOI
        pie_pct = 100 * pie_area / aoi_area if aoi_area > 0 else 0
        union_pct = 100 * mosaic_union_area / aoi_area if aoi_area > 0 else 0
        mosaic_table.append([
            group_id,
            f"{pie_pct:.2f}%",
            f"{union_pct:.2f}%",
            f"{len(mosaic.get('images', []))}"
        ])

    # 4. SOBREPOSIÇÕES ENTRE MOSAICOS (I_jk)
    pairwise_mosaic_overlaps = calculate_pairwise_overlaps(mosaic_geometries)
    total_between_mosaic_overlap = sum(area for _, _, area in pairwise_mosaic_overlaps)

    # 5. SOMA DAS ÁREAS DE UNIÃO DE CADA MOSAICO (opcional, só para comparação)
    total_union_sum = sum(union_areas)

    # 6. FÓRMULA FINAL: Soma das PIE - Soma das sobreposições entre mosaicos
    total_pie_minus_overlaps = sum(pie_areas) - total_between_mosaic_overlap

    all_union_geom = unary_union([g for g in mosaic_geometries if g is not None])
    all_union_area = all_union_geom.area

    # Percentuais
    percentual_milp = 100 * total_pie_minus_overlaps / aoi_area if aoi_area > 0 else 0
    percentual_real = 100 * all_union_area / aoi_area if aoi_area > 0 else 0
    diferenca_km2 = (total_pie_minus_overlaps - all_union_area) / 1e6
    diferenca_pp = percentual_milp - percentual_real

    # MÉTODO INCREMENTAL PADRÃO (UNIÃO GEOMÉTRICA)
    print("\n" + "="*80)
    print(" MÉTODO INCREMENTAL (UNIÃO GEOMÉTRICA) ")
    print("="*80)
    
    # Ordenar mosaicos por tamanho para ver impacto maior primeiro
    ordered_indices = sorted(
        [i for i, g in enumerate(mosaic_geometries) if g is not None],
        key=lambda i: mosaic_geometries[i].area if mosaic_geometries[i] is not None else 0,
        reverse=True
    )
    ordered_geometries = [mosaic_geometries[i] if i < len(mosaic_geometries) else None for i in ordered_indices]
    ordered_ids = [mosaic_ids[i] if i < len(mosaic_ids) else f"unknown_{i}" for i in ordered_indices]
    
    incremental_coverage, coverage_steps = calculate_incremental_coverage(
        ordered_geometries, ordered_ids, aoi_area
    )
    
    # Exibir os resultados incrementais
    incremental_table = []
    for step in coverage_steps:
        incremental_table.append([
            step['mosaico'],
            f"{step['cobertura_individual']:.2f}%",
            f"{step['incremento']:.2f}%",
            f"{step['cobertura_acumulada']:.2f}%"
        ])
    
    print("\n== COBERTURA INCREMENTAL (PAR A PAR) ==")
    print("Mosaicos ordenados por tamanho (maior primeiro):")
    print(tabulate(incremental_table, headers=[
        "Mosaico", "Cobertura Individual", "Incremento Real", "Cobertura Acumulada"
    ]))
    
    print(f"\nCobertura total pelo método incremental: {incremental_coverage:.2f}%")
    print(f"Cobertura pela união geométrica direta: {percentual_real:.2f}%")
    
    # MÉTODO PIE INCREMENTAL
    print("\n" + "="*80)
    print(" MÉTODO PIE INCREMENTAL (PAR A PAR) ")
    print("="*80)

    # Calcular o PIE incrementalmente
    pie_incremental_coverage, pie_steps = calculate_pie_incremental(
        pie_areas, mosaic_geometries, mosaic_ids, aoi_area, pairwise_mosaic_overlaps
    )

    # Exibir os resultados PIE incrementais
    pie_incremental_table = []
    for step in pie_steps:
        pie_incremental_table.append([
            step['mosaico'],
            f"{step['pie_individual_pct']:.2f}%",
            f"{step['pie_incremento_pct']:.2f}%",
            f"{step['pie_acumulado_pct']:.2f}%",
            f"{step['pie_individual_km2']:.2f} km²",
            f"{step['pie_incremento_km2']:.2f} km²",
            f"{step['pie_acumulado_km2']:.2f} km²"
        ])

    print("\n== COBERTURA PIE INCREMENTAL (PAR A PAR) ==")
    print("Mosaicos ordenados por área PIE (maior primeiro):")
    print(tabulate(pie_incremental_table, headers=[
        "Mosaico", "PIE Individual %", "Incremento PIE %", "PIE Acumulado %", 
        "PIE Individual km²", "Incremento PIE km²", "PIE Acumulado km²"
    ]))

    print(f"\nCobertura total pelo método PIE incremental: {pie_incremental_coverage:.2f}%")
    print(f"Cobertura pelo MILP tradicional: {percentual_milp:.2f}%")
    print(f"Cobertura real (união geométrica): {percentual_real:.2f}%")

    # COMPARAÇÃO DE MÉTODOS
    comparison_table = [
        ["Método", "Cobertura (%)", "Km²", "Confiabilidade"],
        ["---------------------------------------------", "-------------", "-------------", "-------------"],
        ["MILP tradicional (soma PIE - sobreposições)", f"{percentual_milp:.2f}%", f"{total_pie_minus_overlaps/1e6:.2f} km²", "Baixa"],
        ["PIE Incremental (par a par)", f"{pie_incremental_coverage:.2f}%", f"{pie_steps[-1]['pie_acumulado_km2'] if pie_steps else 0:.2f} km²", "Média"],
        ["União geométrica incremental", f"{incremental_coverage:.2f}%", f"{all_union_area/1e6:.2f} km²", "Alta"],
        ["União geométrica direta", f"{percentual_real:.2f}%", f"{all_union_area/1e6:.2f} km²", "Alta"],
    ]
    
    print("\n" + "="*80)
    print(" COMPARAÇÃO DOS MÉTODOS DE CÁLCULO DE COBERTURA ")
    print("="*80)
    print(tabulate(comparison_table, headers="firstrow", tablefmt="grid"))

    # 7. RESULTADOS GERAIS
    overlap_details = []
    for i, j, area in pairwise_mosaic_overlaps:
        overlap_details.append([
            f"{mosaic_ids[i]} ∩ {mosaic_ids[j]}",
            f"{area/1e6:.2f} km²"
        ])

    results_table = [
        ["Área total da AOI", f"{aoi_area/1e6:.2f} km²"],
        ["Soma das áreas PIE (já com sobreposições internas descontadas)", f"{sum(pie_areas)/1e6:.2f} km²"],
        ["SOMA das uniões individuais dos mosaicos", f"{total_union_sum/1e6:.2f} km²"],
        ["Total de sobreposições entre mosaicos diferentes", f"{total_between_mosaic_overlap/1e6:.2f} km²"],
        ["FÓRMULA MILP: Soma das PIE - Sobreposições entre mosaicos", f"{total_pie_minus_overlaps/1e6:.2f} km²"],
        ["MÉTODO PIE INCREMENTAL: Soma considerando a ordem", f"{pie_steps[-1]['pie_acumulado_km2'] if pie_steps else 0:.2f} km²"],
        ["MÉTODO GEOMÉTRICO: União total dos mosaicos", f"{all_union_area/1e6:.2f} km²"]
    ]

    print("\n" + "="*80)
    print(f" COMPARAÇÃO DE ÁREAS - {len(selected_mosaics)} MOSAICOS ")
    print("="*80)
    print("\nMOSAICOS INDIVIDUAIS:")
    print(tabulate(mosaic_table, headers=["Mosaico", "% PIE (AOI)", "% União Real (AOI)", "# Imagens"]))
    if overlap_details:
        print("\nSOBREPOSIÇÕES ENTRE MOSAICOS:")
        print(tabulate(overlap_details[:10], headers=["Pares de Mosaicos", "Área Sobreposta"]))
    print("\nRESULTADOS FINAIS:")
    print(tabulate(results_table, headers=["Métrica", "Valor"]))

    output_file = os.path.join(OUTPUT_DIR, f"mosaic_area_comparison_methods.txt")
    with open(output_file, "w") as f:
        f.write(f"COMPARAÇÃO DE MÉTODOS DE CÁLCULO DE COBERTURA - {len(selected_mosaics)} MOSAICOS\n")
        f.write("="*80 + "\n\n")
        
        f.write("MOSAICOS INDIVIDUAIS:\n")
        f.write(tabulate(mosaic_table, headers=["Mosaico", "% PIE (AOI)", "% União Real (AOI)", "# Imagens"]) + "\n\n")
        
        f.write("COBERTURA INCREMENTAL (UNIÃO GEOMÉTRICA):\n")
        f.write(tabulate(incremental_table, headers=[
            "Mosaico", "Cobertura Individual", "Incremento Real", "Cobertura Acumulada"
        ]) + "\n\n")
        
        f.write("COBERTURA PIE INCREMENTAL:\n")
        f.write(tabulate(pie_incremental_table, headers=[
            "Mosaico", "PIE Individual %", "Incremento PIE %", "PIE Acumulado %", 
            "PIE Individual km²", "Incremento PIE km²", "PIE Acumulado km²"
        ]) + "\n\n")
        
        f.write("COMPARAÇÃO DOS MÉTODOS:\n")
        f.write(tabulate(comparison_table, headers="firstrow", tablefmt="grid") + "\n\n")
        
        if overlap_details:
            f.write("SOBREPOSIÇÕES ENTRE MOSAICOS:\n")
            f.write(tabulate(overlap_details[:20], headers=["Pares de Mosaicos", "Área Sobreposta"]) + "\n\n")
        
        f.write("RESULTADOS FINAIS:\n")
        f.write(tabulate(results_table, headers=["Métrica", "Valor"]))

    logging.info(f"Resultados com comparação dos métodos salvos em {output_file}")

    return {
        "total_pie_km2": sum(pie_areas)/1e6,
        "total_union_sum_km2": total_union_sum/1e6,
        "total_pie_minus_overlaps_km2": total_pie_minus_overlaps/1e6,
        "total_between_mosaic_overlap_km2": total_between_mosaic_overlap/1e6,
        "cobertura_percentual": {
            "MILP": percentual_milp,
            "uniao_geometrica": percentual_real,
            "incremental": incremental_coverage,
            "pie_incremental": pie_incremental_coverage
        },
        "diferenca_pp": {
            "MILP_vs_real": percentual_milp - percentual_real,
            "incremental_vs_real": incremental_coverage - percentual_real,
            "pie_incremental_vs_real": pie_incremental_coverage - percentual_real
        },
        "incremental": {
            "coverage_pct": incremental_coverage,
            "steps": coverage_steps
        },
        "pie_incremental": {
            "coverage_pct": pie_incremental_coverage,
            "steps": pie_steps
        }
    }

if __name__ == "__main__":
    try:
        compare_pie_sum_vs_union_sum()
    finally:
        for file in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
"""
Cálculo Preciso de Cobertura Geométrica para Otimização de Mosaicos
===================================================================

Este script implementa um método geométrico preciso para calcular a cobertura efetiva
de grupos de imagens utilizando o Princípio da Inclusão-Exclusão (PIE) modificado.

METODOLOGIA:
-----------
Cada imagem é representada como um polígono espacial na projeção da AOI (Área de Interesse).
O cálculo de cobertura utiliza a fórmula do Princípio da Inclusão-Exclusão para áreas de interseção 2a2:

    Cobertura = [Soma(Áreas individuais) - Soma(Interseções 2a2)]

INTEGRAÇÃO:
----------
Os valores de cobertura geométrica são usados pelo otimizador CPLEX (3.1-CPLEX.py)
para selecionar o conjunto ótimo de grupos de mosaicos.
"""

import json
import os
import sys
import zipfile
from pathlib import Path
import geopandas as gpd
import pyproj
from shapely.ops import transform as shapely_transform
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union
import rasterio
import logging
import shutil
import copy
import itertools
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

JSON_PATH = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/RS/greedy/optimization_parameters.json"
DOWNLOAD_PATH = "/Volumes/luryand/nova_busca/RS"
TEMP_DIR = "/Volumes/luryand/temp"
AOI_SHAPEFILE = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/RS/ibirapuita_31982.shp"
PRE_CALCULATED_OUTPUT_FILE = "/Users/luryand/Documents/encode-image/coverage_otimization/Gulosa-opt/optimization_parameters-RS-precalc.json"

def get_aoi_geometry():
    """Carrega e prepara a geometria da AOI."""
    logging.info(f"Carregando AOI: {AOI_SHAPEFILE}")
    os.environ['USE_DEPRECATED_OPEN'] = 'YES'
    
    try:
        gdf = gpd.read_file(AOI_SHAPEFILE)
        
        if gdf.crs is None:
            logging.warning("CRS não definido para AOI. Assumindo EPSG:4674.")
            gdf.set_crs(epsg=4674, inplace=True)
        
        aoi_union = gdf.unary_union
        aoi_crs = gdf.crs.to_string()
        
        if not aoi_union.is_valid:
            aoi_union = aoi_union.buffer(0)
        
        if aoi_union.area <= 0:
            logging.error(f"AOI com área zero ou negativa: {aoi_union.area}")
        
        logging.info(f"AOI CRS: {aoi_crs}, Área: {aoi_union.area:.2f}, Bounds: {aoi_union.bounds}")
        return aoi_crs, aoi_union
    except Exception as e:
        logging.error(f"Erro ao carregar AOI: {e}")
        sys.exit(1)

def extract_tci_from_zip(zip_path, temp_dir):
    """Extrai arquivo TCI de uma imagem Sentinel-2 ZIP"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            tci_files = [f for f in zip_ref.namelist() if 'TCI_10m.jp2' in f]
            if not tci_files:
                return None
            
            tci_file = tci_files[0]
            extract_path = temp_dir / Path(tci_file).name
            
            with zip_ref.open(tci_file) as source, open(extract_path, 'wb') as target:
                shutil.copyfileobj(source, target)
            
            return str(extract_path)
    except Exception as e:
        logging.error(f"Erro ao extrair TCI: {e}")
        return None

def find_and_get_image_geometry(image_name, aoi_crs):
    """Localiza a imagem, extrai e reprojecta sua geometria para o CRS da AOI."""
    base_name = image_name.replace('.zip', '').replace('.SAFE', '')
    temp_dir = Path(TEMP_DIR) / base_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        zip_path = Path(DOWNLOAD_PATH) / image_name
        if not zip_path.exists():
            for variant in [f"{base_name}.zip", f"{base_name}.SAFE.zip"]:
                alt_path = Path(DOWNLOAD_PATH) / variant
                if alt_path.exists():
                    zip_path = alt_path
                    break
        
        tci_path = None
        if not zip_path.exists():
            safe_path = Path(DOWNLOAD_PATH) / f"{base_name}.SAFE"
            if safe_path.exists():
                for tci_file in safe_path.glob("**/TCI_10m.jp2"):
                    tci_path = str(tci_file)
                    break
            else:
                logging.warning(f"Imagem não encontrada: {image_name}")
                return None
        else:
            tci_path = extract_tci_from_zip(zip_path, temp_dir)
        
        if not tci_path:
            logging.warning(f"TCI não encontrado para {image_name}")
            return None
        
        # Extração de geometria da imagem
        with rasterio.open(tci_path) as src:
            bounds = src.bounds
            source_crs = src.crs.to_string() if src.crs else "EPSG:32724"
            
            # Criação do polígono
            poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            # Reprojeção para o CRS da AOI
            if source_crs != aoi_crs:
                logging.info(f"Reprojetando {image_name}: {source_crs} → {aoi_crs}")
                try:
                    proj_source = pyproj.Proj(source_crs)
                    proj_target = pyproj.Proj(aoi_crs)
                    
                    # Transformer com always_xy=True para orientação correta
                    transformer = pyproj.Transformer.from_proj(
                        proj_source, proj_target, always_xy=True
                    ).transform
                    
                    reprojected_poly = shapely_transform(transformer, poly)
                    
                    if not reprojected_poly.is_valid:
                        reprojected_poly = reprojected_poly.buffer(0)
                    
                    if reprojected_poly.is_valid and reprojected_poly.area > 0:
                        poly = reprojected_poly
                    else:
                        logging.error(f"Reprojeção falhou: válido={reprojected_poly.is_valid}, área={reprojected_poly.area}")
                        return None
                        
                except Exception as e:
                    logging.error(f"Erro de reprojeção: {e}")
                    return None
            
            return poly
    
    except Exception as e:
        logging.error(f"Erro ao processar {image_name}: {e}")
        return None
    
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def filter_high_overlap_images(group_geometries, group_images, aoi_geometry, image_weights):
    """
    Detecta sobreposições altas entre PARES de imagens e aplica regras refinadas para manter
    ou remover imagens, considerando tanto a sobreposição quanto a contribuição única à AOI.
    
    Aplica-se apenas para grupos com 3 ou mais imagens.
    
    Args:
        group_geometries: Lista de geometrias das imagens
        group_images: Lista com nomes das imagens
        aoi_geometry: Geometria da área de interesse
        image_weights: Lista de tuplas (área, cobertura_nuvens) para cada imagem
        
    Returns:
        Tupla com (geometrias filtradas, imagens filtradas, pesos filtrados)
    """
    # Aplicar filtro de sobreposição apenas para grupos com 3 ou mais imagens
    if len(group_geometries) < 3:
        return group_geometries, group_images, image_weights
    
    # Área total da AOI
    aoi_area = aoi_geometry.area
    
    # Armazenar informações sobre cada imagem
    images_info = []
    for i, (geom, img_name, weight) in enumerate(zip(group_geometries, group_images, image_weights)):
        # Interseção da imagem com a AOI
        intersection = geom.intersection(aoi_geometry) if aoi_geometry else geom
        area = intersection.area if not intersection.is_empty else 0
        cloud_coverage = weight[1]  # O segundo elemento do peso é a cobertura de nuvens
        images_info.append({
            "index": i, 
            "geom": geom, 
            "name": img_name,
            "area": area,
            "cloud": cloud_coverage,
            "weight": weight,
            "keep": True  # Inicialmente, manteremos todas as imagens
        })
    
    # Calcular sobreposições par a par
    for i in range(len(images_info)):
        if not images_info[i]["keep"]:
            continue  # Já foi marcada para remoção
            
        for j in range(i+1, len(images_info)):
            if not images_info[j]["keep"]:
                continue  # Já foi marcada para remoção
                
            # Calcular sobreposição entre este par específico
            geom_i = images_info[i]["geom"]
            geom_j = images_info[j]["geom"]
            
            # Interseção entre o par de imagens
            pair_intersection = geom_i.intersection(geom_j)
            # Recortar pela AOI para considerar apenas área relevante
            pair_intersection = pair_intersection.intersection(aoi_geometry) if aoi_geometry else pair_intersection
            
            if not pair_intersection.is_empty:
                overlap_area = pair_intersection.area
                # Calcular ratio em relação à menor das duas imagens
                area_i = images_info[i]["area"]
                area_j = images_info[j]["area"]
                smaller_area = min(area_i, area_j)
                
                if smaller_area > 0:
                    pair_overlap_ratio = overlap_area / smaller_area
                    
                    # Se alta sobreposição entre este par específico (80% ou mais)
                    if pair_overlap_ratio > 0.9:
                        logging.info(f"Alta sobreposição ({pair_overlap_ratio:.2f}) entre imagens {images_info[i]['name']} e {images_info[j]['name']}")
                        
                        # NOVA LÓGICA: Calcular contribuição única de cada imagem para a AOI
                        geom_i_aoi = geom_i.intersection(aoi_geometry)
                        geom_j_aoi = geom_j.intersection(aoi_geometry)
                        
                        # Áreas únicas (que não se sobrepõem com a outra imagem)
                        unique_area_i = geom_i_aoi.difference(geom_j_aoi).area
                        unique_area_j = geom_j_aoi.difference(geom_i_aoi).area
                        
                        # Contribuições únicas como percentual da AOI
                        unique_contribution_i = unique_area_i / aoi_area if aoi_area > 0 else 0
                        unique_contribution_j = unique_area_j / aoi_area if aoi_area > 0 else 0
                        
                        min_contribution_threshold = 0.05  # 5% de contribuição única
                        
                        # Decisão baseada na contribuição única
                        keep_i = unique_contribution_i >= min_contribution_threshold
                        keep_j = unique_contribution_j >= min_contribution_threshold
                        
                        if keep_i and keep_j:
                            # Ambas contribuem significativamente - manter ambas
                            logging.info(f"  → Mantendo ambas as imagens: contribuições únicas I:{unique_contribution_i:.2%}, J:{unique_contribution_j:.2%}")
                        elif keep_i:
                            # Apenas i contribui significativamente
                            images_info[j]["keep"] = False
                            logging.info(f"  → Mantendo apenas {images_info[i]['name']} (contribuição única: {unique_contribution_i:.2%})")
                        elif keep_j:
                            # Apenas j contribui significativamente
                            images_info[i]["keep"] = False
                            logging.info(f"  → Mantendo apenas {images_info[j]['name']} (contribuição única: {unique_contribution_j:.2%})")
                            break  # Saímos do loop j, pois i foi removido
                        else:
                            # Nenhuma contribui significativamente - usar regra original da cobertura de nuvens
                            logging.info(f"  → Contribuições insuficientes ({unique_contribution_i:.2%}, {unique_contribution_j:.2%}), selecionando por cobertura de nuvens")
                            if images_info[i]["cloud"] <= images_info[j]["cloud"]:
                                images_info[j]["keep"] = False
                                logging.info(f"  → Mantendo {images_info[i]['name']} (nuvem: {images_info[i]['cloud']:.4f})")
                            else:
                                images_info[i]["keep"] = False
                                logging.info(f"  → Mantendo {images_info[j]['name']} (nuvem: {images_info[j]['cloud']:.4f})")
                                break  # Saímos do loop j, pois i foi removido
    
    # Filtrar apenas imagens marcadas para manter
    filtered_geometries = [info["geom"] for info in images_info if info["keep"]]
    filtered_images = [info["name"] for info in images_info if info["keep"]]
    filtered_weights = [info["weight"] for info in images_info if info["keep"]]
    
    logging.info(f"Após filtragem par a par: {len(filtered_images)}/{len(group_images)} imagens mantidas")
    
    return filtered_geometries, filtered_images, filtered_weights

def calculate_coverage_twotwo(params):
    """
    Calcula a cobertura dos grupos de mosaicos utilizando o Princípio da Inclusão-Exclusão (PIE) modificado.
    
    Este método:
    1. Calcula explicitamente áreas individuais e interseções 2a2
    2. Computa a cobertura usando a fórmula PIE 
    3. Calcula também a área real preenchida usando unary_union
    
    Args:
        params (dict): Parâmetros de otimização contendo grupos de mosaicos
        
    Returns:
        dict: Parâmetros atualizados com valores de cobertura geométrica
    """
    aoi_crs, aoi_geometry = get_aoi_geometry()
    aoi_area = aoi_geometry.area
    
    if aoi_area <= 0:
        logging.error(f"AOI com área inválida: {aoi_area}")
        return params
    
    result_params = copy.deepcopy(params)
    geometries_cache = {}
    
    # Criar lookup para metadados das imagens
    image_metadata = {img["filename"]: img for img in params["image_catalog"]}
    
    group_count = len(result_params.get('mosaic_groups', []))
    logging.info(f"Processando {group_count} grupos de mosaico usando PIE modificado")
    
    for i, group in enumerate(result_params.get('mosaic_groups', []), 1):
        group_id = group.get('group_id', f'unknown_{i}')
        group_images = group.get('images', [])
        
        logging.info(f"Grupo {group_id} ({i}/{group_count}): {len(group_images)} imagens")
        
        # Coletar geometrias das imagens do grupo
        group_geometries = []
        image_weights = []  # Para calcular média ponderada da cobertura de nuvens
        total_area = 0
        
        for img_name in group_images:
            if img_name in geometries_cache:
                geom = geometries_cache[img_name]
                if geom:
                    group_geometries.append(geom)
                    # Obter metadados da imagem
                    img_meta = image_metadata.get(img_name, {})
                    cloud_coverage = img_meta.get("cloud_coverage", 0)
                    img_area = geom.area
                    image_weights.append((img_area, cloud_coverage))
                    total_area += img_area
            else:
                geom = find_and_get_image_geometry(img_name, aoi_crs)
                geometries_cache[img_name] = geom
                if geom:
                    group_geometries.append(geom)
                    # Obter metadados da imagem
                    img_meta = image_metadata.get(img_name, {})
                    cloud_coverage = img_meta.get("cloud_coverage", 0)
                    img_area = geom.area
                    image_weights.append((img_area, cloud_coverage))
                    total_area += img_area
        
        # Cálculo da cobertura usando PIE modificado
        if not group_geometries:
            logging.warning(f"Nenhuma geometria válida para grupo {group_id}")
            group['geometric_coverage'] = 0.0
            group['avg_cloud_coverage'] = 1.0  # Valor padrão se não conseguir calcular
            continue
        
        try:
            # Primeiro: Aplicar filtragem par a par para remover redundâncias (apenas para grupos com 3+ imagens)
            if len(group_geometries) >= 2:
                filtered_geometries, filtered_images, filtered_weights = filter_high_overlap_images(
                    group_geometries, group_images, aoi_geometry, image_weights
                )

                # Verificar se houve remoção de imagens
                if len(filtered_images) < len(group_images):
                    logging.info(f"Grupo {group_id}: Removidas {len(group_images) - len(filtered_images)} imagens redundantes")
                    
                    # Atualizar as listas com as versões filtradas
                    group_geometries = filtered_geometries
                    group_images = filtered_images
                    image_weights = filtered_weights
                    
                    # Atualizar o grupo com a nova seleção
                    group['images'] = group_images
                    group['selected_by_overlap'] = True
            
            # Segundo: Calcular áreas individuais (após filtragem)
            individual_areas = []
            aoi_geometries = []  # Para calcular união real posteriormente
            
            for geom in group_geometries:
                # Interseção com AOI para considerar apenas a área relevante
                intersection = geom.intersection(aoi_geometry)
                if not intersection.is_empty:
                    individual_areas.append(intersection.area)
                    aoi_geometries.append(intersection)
            
            total_individual_area = sum(individual_areas)
            
            # Terceiro: Calcular interseções 2a2 (após filtragem)
            pairwise_intersections = []
            total_pairwise_overlap = 0
            
            for i, j in itertools.combinations(range(len(group_geometries)), 2):
                # Interseção entre o par de imagens
                pair_intersection = group_geometries[i].intersection(group_geometries[j])
                # Recortar pela AOI para considerar apenas área relevante
                pair_intersection = pair_intersection.intersection(aoi_geometry)
                
                if not pair_intersection.is_empty:
                    area = pair_intersection.area
                    pairwise_intersections.append({
                        "images": (i, j),
                        "area": area
                    })
                    total_pairwise_overlap += area
            
            # Calcular a área real preenchida usando unary_union
            real_coverage_geometry = unary_union(aoi_geometries)
            real_coverage_area = real_coverage_geometry.area if real_coverage_geometry else 0
            real_coverage_ratio = real_coverage_area / aoi_area if aoi_area > 0 else 0
            
            # Aplicar PIE considerando apenas interseções 2a2
            pie_coverage = total_individual_area - total_pairwise_overlap
                 
            # Garantir que a cobertura não exceda a área da AOI
            pie_coverage = min(pie_coverage, aoi_area)
            
            # Calcular a razão de cobertura pelo PIE
            coverage_ratio = pie_coverage / aoi_area
            
            # Armazenar valores no grupo
            group['geometric_coverage'] = coverage_ratio
            group['geometric_coverage_m2'] = pie_coverage
            group['total_individual_area'] = total_individual_area
            group['total_pairwise_overlap'] = total_pairwise_overlap
            
            # Novos campos para comparação entre métodos
            group['real_coverage_area'] = real_coverage_area
            group['real_coverage_ratio'] = real_coverage_ratio
            group['pie_coverage_area'] = pie_coverage
            group['pie_coverage_ratio'] = coverage_ratio
            
            group['pairwise_intersections'] = pairwise_intersections
            
            # NOVA IMPLEMENTAÇÃO: Cálculo correto da cobertura de nuvens
            # Considerando apenas a geometria real do mosaico (sem contar áreas sobrepostas múltiplas vezes)
            try:
                # Já temos a união das geometrias (real_coverage_geometry)
                union_area = real_coverage_area
                
                if union_area > 0:
                    # Calcular contribuição proporcional de cada imagem para o mosaico
                    weighted_cloud_coverage = 0
                    considered_areas = []
                    
                    # Para cada imagem, considerar apenas a parte que não foi considerada antes
                    for idx, (geom, (_, cloud_coverage)) in enumerate(zip(group_geometries, image_weights)):
                        # Recortar a geometria pela AOI
                        aoi_geom = geom.intersection(aoi_geometry) if aoi_geometry else geom
                        
                        # Subtrair áreas já consideradas (para evitar contagem dupla)
                        if considered_areas:
                            already_considered = unary_union(considered_areas)
                            unique_area = aoi_geom.difference(already_considered)
                        else:
                            unique_area = aoi_geom
                        
                        # Adicionar esta geometria às áreas consideradas
                        considered_areas.append(aoi_geom)
                        
                        # Calcular área única e contribuição para cobertura de nuvens
                        unique_area_size = unique_area.area if not unique_area.is_empty else 0
                        weighted_cloud_coverage += unique_area_size * cloud_coverage
                    
                    # Calcular média ponderada
                    avg_cloud_coverage = weighted_cloud_coverage / union_area
                else:
                    # Fallback para média simples se não conseguir calcular área corretamente
                    avg_cloud_coverage = sum(cloud for _, cloud in image_weights) / len(image_weights) if image_weights else 0.0
                    
                # Limitar o valor entre 0 e 1
                avg_cloud_coverage = max(0.0, min(1.0, avg_cloud_coverage))
                group['avg_cloud_coverage'] = avg_cloud_coverage
                logging.info(f"  → Nuvens (método geométrico preciso): {avg_cloud_coverage:.6f}")
                
            except Exception as cloud_err:
                # Fallback para o método antigo em caso de erro
                logging.error(f"Erro no cálculo preciso de nuvens: {cloud_err}. Usando método simples.")
                if total_area > 0:
                    avg_cloud_coverage = sum(area * cloud for area, cloud in image_weights) / total_area
                else:
                    avg_cloud_coverage = sum(cloud for _, cloud in image_weights) / len(image_weights) if image_weights else 0.0
                
                avg_cloud_coverage = max(0.0, min(1.0, avg_cloud_coverage))
                group['avg_cloud_coverage'] = avg_cloud_coverage
            
            logging.info(f"Grupo {group_id}: Cobertura Real = {real_coverage_ratio:.4f}, PIE = {coverage_ratio:.4f}, Nuvens = {avg_cloud_coverage:.4f}")
            logging.info(f"  → Áreas individuais: {total_individual_area:.2f}, Overlap 2a2: {total_pairwise_overlap:.2f}")
            
        except Exception as e:
            logging.error(f"Erro no cálculo para grupo {group_id}: {e}")
            import traceback
            traceback.print_exc()
            group['geometric_coverage'] = 0.0
            group['geometric_coverage_m2'] = 0.0
            group['avg_cloud_coverage'] = 1.0  # Valor padrão em caso de erro
    
    return result_params

def main():
    """
    Função principal que coordena o cálculo de cobertura geométrica
    e salva os resultados no arquivo de parâmetros.
    """
    logging.info("=== Cálculo de Cobertura via Princípio da Inclusão-Exclusão (PIE) Modificado ===")
    
    if not os.path.exists(DOWNLOAD_PATH) or not os.path.exists(AOI_SHAPEFILE):
        logging.error(f"Diretórios ou arquivos necessários não encontrados")
        return
    
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        logging.info(f"Carregando parâmetros: {JSON_PATH}")
        with open(JSON_PATH, 'r') as f:
            params = json.load(f)
        
        if not params.get('mosaic_groups'):
            logging.error("Arquivo de parâmetros não contém 'mosaic_groups'")
            return
            
        # Calcular cobertura usando o método 2a2
        result = calculate_coverage_twotwo(params)

        with open(PRE_CALCULATED_OUTPUT_FILE, 'w') as f:
            json.dump(result, f, indent=2)

        success_count = sum(1 for group in result.get('mosaic_groups', []) 
                           if group.get('geometric_coverage', 0) > 0)
        
        logging.info(f"\n=== RESULTADOS FINAIS ===")
        logging.info(f"Cálculo concluído: {success_count}/{len(result.get('mosaic_groups', []))} grupos com cobertura > 0")
        logging.info(f"Resultados salvos em: {PRE_CALCULATED_OUTPUT_FILE}")
        
    except Exception as e:
        logging.error(f"Erro durante o processamento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
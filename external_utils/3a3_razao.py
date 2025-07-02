import json
import numpy as np
import os
import sys
import logging
import traceback
import itertools
from pathlib import Path
import shutil
import geopandas as gpd
import pyproj
import rasterio
import zipfile
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, box
from shapely.ops import unary_union, transform as shapely_transform
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# Caminhos
JSON_PATH = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/PI-PE-CE/optimization_parameters.json"
DOWNLOAD_PATH = "/Volumes/luryand/nova_busca/PE-PI-CE"
TEMP_DIR = "/Volumes/luryand/temp"
AOI_SHAPEFILE = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/PI-PE-CE/ucs_pe-pi-ce_31984.shp"
ANALYSIS_OUTPUT_FILE = "/Users/luryand/Documents/encode-image/coverage_otimization/3a3_analysis_results.json"

# Funções do area.py
def get_aoi_geometry():
    """Carrega e prepara a geometria da Área de Interesse (AOI)."""
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
    """
    Localiza a imagem, extrai e reprojecta sua geometria para o CRS da AOI.
    """
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

def filter_high_overlap_images(group, group_geometries, group_images, aoi_geometry, params):
    """
    Detecta sobreposições altas entre PARES de imagens e remove imagens redundantes,
    mantendo sempre pelo menos uma imagem para cada área.
    """
    if len(group_geometries) <= 1:
        return group_geometries, group_images
    
    # Criar lookup para metadados das imagens
    image_metadata = {img["filename"]: img for img in params["image_catalog"]}
    
    # Armazenar informações sobre cada imagem
    images_info = []
    for i, (geom, img_name) in enumerate(zip(group_geometries, group_images)):
        intersection = geom.intersection(aoi_geometry) if aoi_geometry else geom
        area = intersection.area if not intersection.is_empty else 0
        cloud = image_metadata.get(img_name, {}).get("cloud_coverage", 1.0)
        images_info.append({
            "index": i, 
            "geom": geom, 
            "name": img_name,
            "area": area,
            "cloud": cloud,
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
            intersection = images_info[i]["geom"].intersection(images_info[j]["geom"])
            if not intersection.is_empty:
                intersection = intersection.intersection(aoi_geometry) if aoi_geometry else intersection
                
                if intersection.is_empty:
                    continue
                    
                overlap_area = intersection.area
                # Calcular ratio em relação à menor das duas imagens
                smaller_area = min(images_info[i]["area"], images_info[j]["area"])
                if smaller_area > 0:
                    pair_overlap_ratio = overlap_area / smaller_area
                    
                    # Se alta sobreposição entre este par específico
                    if pair_overlap_ratio > 0.5:  # 50% da menor imagem é sobreposta
                        logging.info(f"Alta sobreposição ({pair_overlap_ratio:.2f}) entre imagens {i+1} e {j+1}")
                        
                        # Decidir qual manter baseado na cobertura de nuvens
                        if images_info[i]["cloud"] <= images_info[j]["cloud"]:
                            images_info[j]["keep"] = False  # Remover j
                            logging.info(f"  → Mantendo imagem {i+1} (nuvem: {images_info[i]['cloud']:.4f})")
                        else:
                            images_info[i]["keep"] = False  # Remover i
                            logging.info(f"  → Mantendo imagem {j+1} (nuvem: {images_info[j]['cloud']:.4f})")
                            break  # Precisamos sair, pois i foi removido
    
    # Filtrar apenas imagens marcadas para manter
    filtered_images = [info["name"] for info in images_info if info["keep"]]
    filtered_geometries = [info["geom"] for info in images_info if info["keep"]]
    
    logging.info(f"Após filtragem: {len(filtered_images)}/{len(group_images)} imagens mantidas")
    
    return filtered_geometries, filtered_images

# Funções específicas para análise de interseções 3a3
def analyze_triple_intersections(group, aoi_crs, aoi_geometry, geometries_cache):
    """
    Analisa as interseções triplas em um grupo usando as geometrias reais das imagens.
    """
    group_id = group.get('group_id', 'unknown')
    group_images = group.get('images', [])
    
    if len(group_images) < 3:
        return {
            "group_id": group_id,
            "num_images": len(group_images),
            "has_triple_intersections": False,
            "message": f"Grupo {group_id} tem menos de 3 imagens"
        }
    
    # Coletar geometrias das imagens do grupo
    group_geometries = []
    for img_name in group_images:
        if img_name in geometries_cache:
            geom = geometries_cache[img_name]
        else:
            geom = find_and_get_image_geometry(img_name, aoi_crs)
            geometries_cache[img_name] = geom
            
        if geom and geom.is_valid:
            # Se temos AOI, recortar a geometria
            if aoi_geometry and not aoi_geometry.is_empty:
                # Recortar a geometria pela AOI
                geom = geom.intersection(aoi_geometry)
                if geom and not geom.is_empty:
                    group_geometries.append(geom)
            else:
                group_geometries.append(geom)
            
            # ERRO CORRIGIDO: Removi esta parte que causava duplicação de geometrias
            # if geom and not geom.is_empty:
            #     group_geometries.append(geom)
    
    if len(group_geometries) < 3:
        return {
            "group_id": group_id,
            "num_images": len(group_geometries),
            "has_triple_intersections": False,
            "message": f"Grupo {group_id}: Menos de 3 geometrias válidas encontradas"
        }
    
    # Calcular interseções 2 a 2
    pair_intersections = []
    for i in range(len(group_geometries)):
        for j in range(i+1, len(group_geometries)):
            intersection = group_geometries[i].intersection(group_geometries[j])
            if not intersection.is_empty:
                pair_intersections.append({
                    "images": (i, j),
                    "image_names": (group_images[i], group_images[j]),
                    "area": intersection.area
                })
    
    # Calcular interseções 3 a 3
    triple_intersections = []
    for i in range(len(group_geometries)):
        for j in range(i+1, len(group_geometries)):
            for k in range(j+1, len(group_geometries)):
                intersection = group_geometries[i].intersection(group_geometries[j])
                if not intersection.is_empty:
                    intersection = intersection.intersection(group_geometries[k])
                    if not intersection.is_empty:
                        triple_intersections.append({
                            "images": (i, j, k),
                            "image_names": (group_images[i], group_images[j], group_images[k]),
                            "area": intersection.area
                        })
    
    # Calcular estatísticas
    total_pair_area = sum(p["area"] for p in pair_intersections)
    total_triple_area = sum(t["area"] for t in triple_intersections)
    avg_pair_area = total_pair_area / len(pair_intersections) if pair_intersections else 0
    avg_triple_area = total_triple_area / len(triple_intersections) if triple_intersections else 0
    
    # Calcular razão entre áreas triplas e pares
    ratio = total_triple_area / total_pair_area if total_pair_area > 0 else 0
    
    return {
        "group_id": group_id,
        "num_images": len(group_geometries),
        "num_pair_intersections": len(pair_intersections),
        "num_triple_intersections": len(triple_intersections),
        "total_pair_area": float(total_pair_area),
        "total_triple_area": float(total_triple_area),
        "avg_pair_area": float(avg_pair_area),
        "avg_triple_area": float(avg_triple_area),
        "triple_to_pair_ratio": float(ratio),
        "has_triple_intersections": len(triple_intersections) > 0
    }

def visualize_intersections(group, group_geometries, group_images, aoi_geometry=None, output_dir="./plots"):
    """Visualiza as interseções 2a2 e 3a3 para um grupo, sem títulos ou labels de imagens."""
    group_id = group.get('group_id', 'unknown')
    
    if len(group_geometries) < 1:
        logging.info(f"Grupo {group_id} não tem imagens suficientes para visualização")
        return
    
    # Criar diretório para salvar os plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Plotar geometrias e interseções
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Se temos AOI, plotar como fundo
    if aoi_geometry and not aoi_geometry.is_empty:
        try:
            if isinstance(aoi_geometry, MultiPolygon):
                for poly in aoi_geometry.geoms:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, alpha=0.1, color='gray')
                # Sem label para AOI
            else:
                x, y = aoi_geometry.exterior.xy
                ax.fill(x, y, alpha=0.1, color='gray')  # Sem label para AOI
        except Exception as e:
            logging.warning(f"AOI não pôde ser plotada: {str(e)}")
    
    # Plotar imagens individuais - sem mostrar labels
    for i, geom in enumerate(group_geometries):
        if geom and not geom.is_empty:
            try:
                if isinstance(geom, MultiPolygon):
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, alpha=0.3)
                    # Sem label para imagens individuais
                else:
                    x, y = geom.exterior.xy
                    ax.fill(x, y, alpha=0.3)  # Sem label para imagens individuais
            except Exception as e:
                logging.warning(f"Geometria {i} não pôde ser plotada: {str(e)}")
    
    # Plotar interseções 2 a 2 (se houver mais de uma imagem)
    if len(group_geometries) > 1:
        for i in range(len(group_geometries)):
            for j in range(i+1, len(group_geometries)):
                intersection = group_geometries[i].intersection(group_geometries[j])
                if not intersection.is_empty:
                    try:
                        if isinstance(intersection, MultiPolygon):
                            for part in intersection.geoms:
                                try:
                                    x, y = part.exterior.xy
                                    ax.fill(x, y, alpha=0.5, color='orange')
                                except:
                                    pass
                            # Adicionar label apenas uma vez
                            if i == 0 and j == 1:
                                ax.plot([], [], color='orange', alpha=0.5, label="Interseção 2a2")
                        else:
                            x, y = intersection.exterior.xy
                            ax.fill(x, y, alpha=0.5, color='orange', 
                                   label="Interseção 2a2" if i == 0 and j == 1 else "")
                    except:
                        logging.warning(f"Interseção 2a2 {i},{j} não pôde ser plotada")
    
    # Plotar interseções 3 a 3 (se houver pelo menos 3 imagens)
    if len(group_geometries) >= 3:
        intersection_3a3_added = False
        for i in range(len(group_geometries)):
            for j in range(i+1, len(group_geometries)):
                for k in range(j+1, len(group_geometries)):
                    intersection = group_geometries[i].intersection(group_geometries[j])
                    if not intersection.is_empty:
                        intersection = intersection.intersection(group_geometries[k])
                        if not intersection.is_empty:
                            try:
                                if isinstance(intersection, MultiPolygon):
                                    for part in intersection.geoms:
                                        try:
                                            x, y = part.exterior.xy
                                            ax.fill(x, y, alpha=0.7, color='red')
                                        except:
                                            pass
                                    if not intersection_3a3_added:
                                        ax.plot([], [], color='red', alpha=0.7, label="Interseção 3a3")
                                        intersection_3a3_added = True
                                else:
                                    x, y = intersection.exterior.xy
                                    label = "Interseção 3a3" if not intersection_3a3_added else None
                                    intersection_3a3_added = True
                                    ax.fill(x, y, alpha=0.7, color='red', label=label)
                            except:
                                logging.warning(f"Interseção 3a3 {i},{j},{k} não pôde ser plotada")
    
    # Filtrar legenda para manter APENAS os elementos de interseção
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles = []
    filtered_labels = []
    
    for h, label in zip(handles, labels):
        if "Interseção" in label:  # Mantém apenas as labels de interseção
            filtered_handles.append(h)
            filtered_labels.append(label)
    
    # Aplicar a legenda filtrada
    ax.legend(filtered_handles, filtered_labels)
    
    # Remover título
    # ax.set_title(f"Grupo {group_id}")  # Esta linha foi removida
    
    ax.set_aspect('equal')
    # ax.set_axis_off()  # Remove eixos para visual mais limpo
    
    # Salvar o plot - nome de arquivo não mostra filenames
    output_file = os.path.join(output_dir, f"group_{group_id}_intersections.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"Visualização salva em {output_file}")
    plt.close()

def main():
    """Função principal para análise e visualização das interseções."""
    logging.info("=== ANÁLISE E VISUALIZAÇÃO DE IMAGENS ===")
    
    # Verificar estruturas de diretórios
    if not os.path.exists(DOWNLOAD_PATH) or not os.path.exists(AOI_SHAPEFILE):
        logging.error("Diretórios ou arquivos necessários não encontrados")
        return
    
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    output_dir = "./plots"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Carregar parâmetros
        logging.info(f"Carregando parâmetros: {JSON_PATH}")
        with open(JSON_PATH, 'r') as f:
            params = json.load(f)
        
        if not params.get('mosaic_groups'):
            logging.error("Arquivo de parâmetros não contém 'mosaic_groups'")
            return
        
        # Carregar AOI
        aoi_crs, aoi_geometry = get_aoi_geometry()
        
        # Cache para geometrias de imagens
        geometries_cache = {}
        
        # Variáveis para estatísticas
        group_results = []
        total_groups = len(params.get('mosaic_groups', []))
        groups_with_triple = 0
        all_ratios = []
        processed_groups = 0
        
        logging.info(f"Processando {total_groups} grupos de mosaico")
        
        # Processar apenas uma amostra de grupos para não gerar muitos plots
        sample_indices = np.random.choice(
            range(total_groups),
            min(20, total_groups),  # No máximo 20 grupos
            replace=False
        )
        
        for i in sample_indices:
            group = params.get('mosaic_groups', [])[i]
            group_id = group.get('group_id', f'unknown_{i}')
            logging.info(f"Processando grupo {group_id} ({i+1}/{len(sample_indices)})")
            
            # Analisar interseções triplas (para estatísticas)
            result = analyze_triple_intersections(group, aoi_crs, aoi_geometry, geometries_cache)
            
            if result["has_triple_intersections"]:
                groups_with_triple += 1
                all_ratios.append(result["triple_to_pair_ratio"])
            
            # Obter imagens e geometrias do grupo
            group_images = group.get('images', [])
            
            # Pular se não há imagens
            if not group_images:
                logging.info(f"Grupo {group_id} não tem imagens, pulando...")
                continue
            
            # Obter geometrias das imagens
            group_geometries_full = []  # Geometrias completas para visualização
            group_geometries_aoi = []   # Geometrias recortadas para análise de sobreposição
            valid_images = []

            for img_name in group_images:
                if img_name in geometries_cache:
                    geom = geometries_cache[img_name]
                else:
                    geom = find_and_get_image_geometry(img_name, aoi_crs)
                    geometries_cache[img_name] = geom
                    
                if geom and geom.is_valid:
                    # Guardar geometria completa para visualização
                    group_geometries_full.append(geom)
                    
                    # Recortar pela AOI para análise de sobreposição
                    if aoi_geometry and not aoi_geometry.is_empty:
                        geom_aoi = geom.intersection(aoi_geometry)
                        if geom_aoi and not geom_aoi.is_empty:
                            group_geometries_aoi.append(geom_aoi)
                            valid_images.append(img_name)
            
            # Aplicar filtro de alto overlap se houver geometrias válidas
            if len(group_geometries_aoi) > 0:
                # Analisar sobreposição usando geometrias recortadas pela AOI
                _, filtered_images = filter_high_overlap_images(
                    group, group_geometries_aoi, valid_images, aoi_geometry, params
                )
                
                # Obter geometrias completas das imagens filtradas
                filtered_geometries_full = []
                for img_name in filtered_images:
                    idx = valid_images.index(img_name)
                    # Usar a geometria completa correspondente
                    full_idx = group_images.index(img_name)
                    if full_idx < len(group_geometries_full):
                        filtered_geometries_full.append(group_geometries_full[full_idx])
                
                # Visualizar usando geometrias COMPLETAS
                visualize_intersections(group, filtered_geometries_full, filtered_images, aoi_geometry, output_dir)
                processed_groups += 1
                logging.info(f"Visualizado grupo {group_id} com {len(filtered_images)}/{len(valid_images)} imagens após filtragem")
            else:
                logging.warning(f"Grupo {group_id} não tem geometrias válidas para visualização")
            
            group_results.append(result)
        
        # Calcular estatísticas globais
        avg_ratio = np.mean(all_ratios) if all_ratios else 0
        median_ratio = np.median(all_ratios) if all_ratios else 0
        min_ratio = min(all_ratios) if all_ratios else 0
        max_ratio = max(all_ratios) if all_ratios else 0
        std_ratio = np.std(all_ratios) if all_ratios else 0
        
        summary = {
            "total_groups": total_groups,
            "processed_groups": processed_groups,
            "groups_with_triple_intersections": groups_with_triple,
            "percentage_with_triple": groups_with_triple / total_groups * 100 if total_groups > 0 else 0,
            "triple_to_pair_ratio": {
                "mean": float(avg_ratio),
                "median": float(median_ratio),
                "min": float(min_ratio),
                "max": float(max_ratio),
                "std": float(std_ratio)
            },
            "group_results": group_results
        }
        
        # Salvar resultados da análise
        with open(ANALYSIS_OUTPUT_FILE, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Relatório estatístico final
        logging.info("\n===== RESULTADOS FINAIS =====")
        logging.info(f"Total de grupos processados: {processed_groups}/{total_groups}")
        logging.info(f"Grupos com interseções triplas: {groups_with_triple} ({summary['percentage_with_triple']:.1f}%)")
        
        if all_ratios:
            logging.info("\n===== ESTATÍSTICAS DE INTERSEÇÕES TRIPLAS =====")
            logging.info(f"Razão média (área 3a3 / área 2a2): {avg_ratio:.4f}")
            logging.info(f"Razão mediana: {median_ratio:.4f}")
            logging.info(f"Razão mínima: {min_ratio:.4f}")
            logging.info(f"Razão máxima: {max_ratio:.4f}")
            logging.info(f"Desvio padrão: {std_ratio:.4f}")
            
            # Histograma das razões
            plt.figure(figsize=(10, 6))
            plt.hist(all_ratios, bins=20, alpha=0.7)
            plt.title("Distribuição da Razão entre Áreas de Interseções 3a3 e 2a2")
            plt.xlabel("Razão (Área 3a3 / Área 2a2)")
            plt.ylabel("Frequência")
            plt.grid(True, alpha=0.3)
            histogram_file = os.path.join(output_dir, "triple_intersection_ratio_histogram.png")
            plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
            logging.info(f"Histograma salvo em {histogram_file}")
        
        logging.info(f"\nResultados salvos em {ANALYSIS_OUTPUT_FILE}")
        logging.info(f"Visualizações salvas em {output_dir}")
        
        if all_ratios:
            logging.info("\nCoeficiente recomendado para o modelo MILP:")
            logging.info(f"triple_intersection_coefficient = {avg_ratio:.4f}")
        
    except Exception as e:
        logging.error(f"Erro durante o processamento: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
import json
import os
import numpy as np
import logging
from pathlib import Path
import zipfile
import shutil
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Patch
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, shape, box
import pyproj
from shapely.ops import transform as shapely_transform
from datetime import datetime
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# Configurar estilo elegante para publicação científica
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.grid'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# Configurações - usando Path para consistência
IMAGE_DIR = Path("/Volumes/luryand/nova_busca")
TEMP_DIR = Path("/Volumes/luryand/temp_mosaics_plot")
JSON_FILE = Path("/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/BA/cplex_selected_mosaic_groups-BA-og1g2.json")
SHP_FILE = Path("/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/BA/ucs_bahia_31984.shp")
OUTPUT_DIR = Path("/Volumes/luryand/mosaicos_plot/BA")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_tci_from_zip(zip_path, temp_dir):
    """Extrai arquivo TCI de uma imagem Sentinel-2 ZIP."""
    try:
        zip_path = Path(zip_path)
        temp_dir = Path(temp_dir)
        
        with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
            tci_files = [f for f in zip_ref.namelist() if 'TCI_10m.jp2' in f]
            if not tci_files:
                return None
            
            tci_file = tci_files[0]
            extract_path = temp_dir / Path(os.path.basename(tci_file))
            
            with zip_ref.open(tci_file) as source, open(extract_path, 'wb') as target:
                shutil.copyfileobj(source, target)
            
            if extract_path.exists():
                return str(extract_path)
            else:
                return None
            
    except Exception as e:
        logging.error(f"Erro ao extrair TCI de {zip_path}: {e}")
        return None

def find_image_path(image_name):
    """Encontra o caminho da imagem em qualquer subdiretório do diretório base."""
    base_name = image_name.replace('.zip', '').replace('.SAFE', '')
    
    for variant in [f"{base_name}.zip", f"{base_name}.SAFE.zip", f"{base_name}.SAFE"]:
        direct_path = IMAGE_DIR / variant
        if direct_path.exists():
            return str(direct_path)
            
        for root, dirs, files in os.walk(IMAGE_DIR):
            root_path = Path(root)
            path = root_path / variant
            if path.exists():
                return str(path)
    
    logging.warning(f"Imagem não encontrada: {image_name}")
    return None

def get_tci_path(image_name, temp_dir):
    """Obtém o caminho do arquivo TCI para a imagem."""
    image_path = find_image_path(image_name)
    if not image_path:
        return None
    
    if image_path.endswith('.zip'):
        tci_path = extract_tci_from_zip(image_path, temp_dir)
        if tci_path and Path(tci_path).exists():
            return tci_path
        else:
            return None
    else:  # SAFE directory
        tci_files = list(Path(image_path).glob("**/TCI_10m.jp2"))
        if tci_files:
            return str(tci_files[0])
        else:
            return None

def get_image_geometry(image_name, temp_dir, aoi_crs):
    """Obtém a geometria da imagem reprojetada para o CRS da AOI."""
    tci_path = get_tci_path(image_name, temp_dir)
    if not tci_path:
        return None
    
    try:
        with rasterio.open(tci_path) as src:
            bounds = src.bounds
            source_crs = src.crs.to_string()
            
            # Criação do polígono na projeção original
            poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            # Reprojeção para o CRS da AOI
            if source_crs != aoi_crs:
                try:
                    proj_source = pyproj.Proj(source_crs)
                    proj_target = pyproj.Proj(aoi_crs)
                    
                    transformer = pyproj.Transformer.from_proj(
                        proj_source, proj_target, always_xy=True
                    ).transform
                    
                    reprojected_poly = shapely_transform(transformer, poly)
                    
                    if not reprojected_poly.is_valid:
                        reprojected_poly = reprojected_poly.buffer(0)
                    
                    if reprojected_poly.is_valid and reprojected_poly.area > 0:
                        poly = reprojected_poly
                    else:
                        return None
                except Exception:
                    return None
            
            return poly
            
    except Exception as e:
        logging.error(f"Erro ao processar geometria de {image_name}: {e}")
        return None

def plot_raster(out_image, out_transform, ax):
    """
    Plotar raster usando matplotlib com melhor visualização e transparência fora da AOI.
    Agora, as áreas com valor 0 (fora da AOI) ficam transparentes.
    """
    # Combinar as bandas RGB (R=0, G=1, B=2)
    img_rgb = np.dstack((out_image[0], out_image[1], out_image[2]))
    
    # Criar máscara de transparência onde todos os canais são 0
    alpha_mask = np.ones(img_rgb.shape[:2], dtype=np.float32)
    zeros_mask = np.all(img_rgb == 0, axis=2)
    alpha_mask[zeros_mask] = 0  # Áreas com 0 em todos os canais RGB ficam transparentes
    
    # Melhor normalização com percentis mais extremos para maior contraste
    p99 = np.percentile(img_rgb[~zeros_mask], 99.5) if np.any(~zeros_mask) else 3000  
    p01 = np.percentile(img_rgb[~zeros_mask], 0.5) if np.any(~zeros_mask) else 0
    
    # Calcular um intervalo dinâmico de contraste
    scale_factor = p99 - p01
    if scale_factor <= 0:
        scale_factor = 3000
    
    # Normalizar usando a escala dinâmica ajustada
    img_rgb = np.clip((img_rgb - p01) / scale_factor, 0, 1)
    
    # Ajuste de gamma para melhorar áreas escuras - valor menor = maior brilho
    gamma = 0.7
    img_rgb = np.power(img_rgb, gamma)
    
    # Melhorar saturação levemente
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    delta = (max_rgb - min_rgb) + 1e-7
    
    # Aumentar saturação
    sat_boost = 1.2
    r = np.clip(max_rgb - (max_rgb - r) * sat_boost, 0, 1)
    g = np.clip(max_rgb - (max_rgb - g) * sat_boost, 0, 1)
    b = np.clip(max_rgb - (max_rgb - b) * sat_boost, 0, 1)
    
    # Recombinar com canal alpha
    img_rgba = np.dstack((r, g, b, alpha_mask))
    
    # Calcular a extensão da imagem
    width = out_image.shape[2]
    height = out_image.shape[1]
    
    left = out_transform[2]
    top = out_transform[5]
    right = left + width * out_transform[0]
    bottom = top + height * out_transform[4]
    
    # Plotar usando imshow com canal alpha para transparência
    ax.imshow(img_rgba, extent=[left, right, bottom, top], zorder=10, 
              interpolation='bilinear')

def get_polygon_path(geom):
    """Cria um matplotlib Path a partir de uma geometria shapely."""
    if hasattr(geom, "exterior"):
        # Polygon
        return MplPath(np.array(geom.exterior.coords))
    elif hasattr(geom, "geoms"):
        # MultiPolygon - pega apenas o primeiro
        return MplPath(np.array(geom.geoms[0].exterior.coords))
    return None

def detect_image_brightness(ax, bbox):
    """Detecta se a área sob a legenda é predominantemente clara ou escura."""
    # Converter bbox para coordenadas de figura
    fig = ax.get_figure()
    x0, y0, width, height = bbox.bounds
    
    # Limites do bbox em coordenadas da figura
    bbox_display = ax.transData.transform([(x0, y0), (x0+width, y0+height)])
    x0, y0 = bbox_display[0]
    x1, y1 = bbox_display[1]
    
    # Obter dados RGB da área sob a legenda
    try:
        buf = fig.canvas.copy_from_bbox(mpl.transforms.Bbox([[x0, y0], [x1, y1]]))
        width, height = int(x1-x0), int(y1-y0)
        rgb_data = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
        
        # Calcular luminosidade média
        luminosity = 0.299*rgb_data[:,:,0] + 0.587*rgb_data[:,:,1] + 0.114*rgb_data[:,:,2]
        mean_luminosity = np.mean(luminosity)
        
        # Retorna True para áreas claras (usar texto preto)
        return mean_luminosity > 128
    except:
        # Em caso de erro, assumir fundo claro por padrão
        return True

def create_overlap_legend(ax, has_overlap_2, has_overlap_3, x_pos=0.80, y_pos=0.02):
    """
    Cria legendas para as sobreposições de imagens no mosaico com adaptação automática
    de cores baseada no fundo da imagem.
    """
    # Adicionar legendas específicas para os tipos de sobreposição
    legend_patches = []
    
    if has_overlap_2:
        overlap_2_patch = Patch(facecolor='yellow', alpha=0.3, edgecolor='#FF5733', 
                              linewidth=1.0, label='Sobreposição 2 a 2')
        legend_patches.append(overlap_2_patch)
    
    if has_overlap_3:
        overlap_3_patch = Patch(facecolor='#FF9900', alpha=0.4, edgecolor='#FF5500', 
                              linewidth=1.0, label='Sobreposição 3 a 3')
        legend_patches.append(overlap_3_patch)
    
    # Se não temos nenhuma sobreposição, adicionar um patch genérico
    if not legend_patches:
        generic_patch = Patch(facecolor='yellow', alpha=0.3, edgecolor='#FF5733', 
                           linewidth=1.0, label='Sobreposição de imagens')
        legend_patches.append(generic_patch)
    
    # IMPORTANTE: Desativar clipping para a legenda
    original_clip_state = ax.get_clip_on()
    ax.set_clip_on(False)
    
    # Primeiro criar uma legenda temporária para obter dimensões
    temp_leg = ax.legend(handles=legend_patches, 
                       loc='upper left',
                       fontsize='medium',
                       framealpha=0.0,  # Temporariamente transparente
                       fancybox=True,
                       bbox_to_anchor=(x_pos, y_pos))
    
    # Forçar desenho para poder analisar a área
    plt.draw()
    
    # Obter a bbox da legenda
    bbox = temp_leg.get_window_extent().transformed(ax.transData.inverted())
    
    # Remover legenda temporária
    temp_leg.remove()
    
    # Analisar o fundo (isto requer interatividade, então é uma aproximação)
    # Assumindo inicialmente que o fundo é claro
    is_light_background = True
    
    # Cores adaptativas com base no fundo
    if is_light_background:
        bg_color = 'white'
        text_color = 'black'
        edge_color = 'black'
    else:
        bg_color = 'black'
        text_color = 'white'
        edge_color = 'white'
    
    # Criar legenda final com cores apropriadas
    leg = ax.legend(handles=legend_patches, 
                   loc='upper left',
                   fontsize='medium',
                   framealpha=0.9,
                   fancybox=True,
                   bbox_to_anchor=(x_pos, y_pos))
    
    # Definir zorder separadamente após criar a legenda
    leg.set_zorder(100)
    
    # Aplicar estilo à legenda com cores adaptativas
    frame = leg.get_frame()
    frame.set_facecolor(bg_color)
    frame.set_linewidth(1.0)
    frame.set_edgecolor(edge_color)
    
    # Ajustar cor do texto
    for text in leg.get_texts():
        text.set_color(text_color)
    
    # Restaurar estado original de clipping
    ax.set_clip_on(original_clip_state)
    
    return leg

def plot_single_mosaic(mosaic_idx, max_mosaicos):
    """Estilo factivel.py: plota um único mosaico e salva com qualidade para artigo científico."""
    # Carregar dados necessários
    aoi_gdf = gpd.read_file(SHP_FILE)
    
    with open(JSON_FILE, 'r') as f:
        json_data = json.load(f)
    
    # Verificar formato do JSON e obter o mosaico correto
    if isinstance(json_data, list):
        mosaics = json_data
    elif isinstance(json_data, dict) and 'mosaic_groups' in json_data:
        mosaics = json_data['mosaic_groups']
    else:
        logging.error("Formato do JSON não suportado")
        return
    
    if mosaic_idx >= len(mosaics):
        logging.error(f"Índice {mosaic_idx} fora dos limites ({len(mosaics)} mosaicos)")
        return
    
    mosaic = mosaics[mosaic_idx]
    group_id = mosaic.get('group_id', f"mosaic_{mosaic_idx+1}")
    image_names = mosaic.get('images', [])
    date_str = ""
    
    # Obter data do mosaico apenas para log
    if 'time_window_start' in mosaic:
        try:
            date = datetime.fromisoformat(mosaic['time_window_start'].replace('Z', '+00:00'))
            date_str = date.strftime('%Y-%m-%d')
        except:
            date_str = mosaic.get('time_window_start', '')[:10]
    
    logging.info(f"Plotando mosaico {group_id} ({mosaic_idx+1}/{max_mosaicos}) - Data: {date_str}")
    
    # Limpar diretório temporário
    for file in os.listdir(TEMP_DIR):
        file_path = TEMP_DIR / file
        if file_path.is_file():
            os.remove(file_path)
    
    # Criar figura com alta resolução e fundo branco explícito
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300, facecolor='white')
    fig.patch.set_facecolor('white')  # Certeza absoluta que o fundo é branco
    ax.set_facecolor('white')  # Fundo do axes também branco
    
    # Preparar AOI
    aoi_geometry = aoi_gdf.union_all()
    aoi_geojson = [mapping(aoi_geometry)]
    
    # Definir limites exatos da AOI, nada de buffer
    minx, miny, maxx, maxy = aoi_geometry.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    # Função auxiliar para obter vértices de uma geometria
    def get_path_vertices(geometry):
        if hasattr(geometry, "exterior"):
            return np.array(geometry.exterior.coords)
        elif hasattr(geometry, "geoms"):
            # Para MultiPolygon, retorna todos os vértices
            vertices = []
            for poly in geometry.geoms:
                if hasattr(poly, "exterior"):
                    vertices.extend(np.array(poly.exterior.coords))
            return np.array(vertices)
        return np.array([])
    
    # Criar patch para AOI - este será usado como máscara de recorte
    aoi_vertices = get_path_vertices(aoi_geometry)
    if len(aoi_vertices) > 0:
        aoi_path = MplPath(aoi_vertices)
        patch = PathPatch(aoi_path, transform=ax.transData, 
                          facecolor='none', edgecolor='none')
        ax.add_patch(patch)
        ax.set_clip_path(patch)

    # Preencher a AOI com BRANCO para áreas sem imagem
    if hasattr(aoi_geometry, "geoms"):
        # MultiPolygon
        for poly in aoi_geometry.geoms:
            ax.add_patch(plt.Polygon(np.array(poly.exterior.coords), 
                                    closed=True, 
                                    facecolor='white',
                                    edgecolor='none', 
                                    zorder=5))
    else:
        # Polygon
        ax.add_patch(plt.Polygon(np.array(aoi_geometry.exterior.coords), 
                                closed=True, 
                                facecolor='white',
                                edgecolor='none', 
                                zorder=5))
    
    # Coletar geometrias das imagens com cores mais suaves
    image_geometries = []
    # Cores mais suaves e elegantes
    image_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', 
                    '#34495e', '#16a085', '#27ae60', '#2980b9', '#8e44ad', '#f1c40f']
    
    # Contador de imagens processadas com sucesso
    processed_images = 0
    
    # Plotar cada imagem
    for i, image_name in enumerate(image_names):
        logging.info(f"Processando imagem {i+1}/{len(image_names)}: {image_name}")
        
        tci_path = get_tci_path(image_name, TEMP_DIR)
        if not tci_path:
            logging.warning(f"TCI não encontrado para {image_name}, pulando...")
            continue
        
        # Verificação de tamanho do arquivo
        file_size = os.path.getsize(tci_path)
        if file_size < 1000:
            logging.warning(f"Arquivo TCI suspeito (muito pequeno: {file_size} bytes)")
            continue
        
        # Obter geometria para bounding box
        image_geom = get_image_geometry(image_name, TEMP_DIR, aoi_gdf.crs.to_string())
        if image_geom:
            image_geometries.append((image_geom, image_name, image_colors[i % len(image_colors)]))
        
        # Plotar a imagem mascarada pela AOI com transparência fora da AOI
        try:
            with rasterio.open(tci_path) as src:
                # Verificar se temos pelo menos 3 bandas (RGB)
                if src.count < 3:
                    logging.warning(f"Imagem {image_name} não tem bandas RGB suficientes")
                    continue
                
                # Usar filled=True para mostrar a imagem dentro da AOI
                out_image, out_transform = mask(
                    src, aoi_geojson, crop=True, nodata=0, filled=True, all_touched=True
                )
                
                # Verificar se a máscara retornou dados
                if np.max(out_image) <= 0:
                    logging.warning(f"Máscara não retornou dados para {image_name}")
                    continue
                
                # Usar a função de plotagem melhorada com transparência
                plot_raster(out_image, out_transform, ax)
                logging.info(f"Imagem {image_name} plotada com sucesso")
                processed_images += 1
                
        except Exception as e:
            logging.warning(f"Erro ao plotar {image_name}: {e}")
        finally:
            # Remover TCI imediatamente
            if tci_path and os.path.exists(tci_path):
                try:
                    os.remove(tci_path)
                except:
                    pass
    
    # Verificar se alguma imagem foi processada
    if processed_images == 0:
        logging.warning(f"Nenhuma imagem processada com sucesso para o mosaico {group_id}")
        plt.close('all')
        return
    
    # Variáveis para acompanhar os diferentes tipos de sobreposição
    has_overlap_2 = False  # Sobreposição 2-a-2
    has_overlap_3 = False  # Sobreposição 3-ou-mais
    
    # Detectar sobreposições e contar quantas imagens se sobrepõem em cada ponto
    all_polygons = [geom for geom, _, _ in image_geometries]
    
    # Calculando todas as interseções 2-a-2 primeiro
    overlap_2_polygons = []  # Lista para armazenar sobreposições 2-a-2
    
    for i, poly1 in enumerate(all_polygons):
        for j, poly2 in enumerate(all_polygons):
            if i >= j:  # Evitar duplicação
                continue
            if poly1.intersects(poly2):
                overlap = poly1.intersection(poly2)
                # Limitar a sobreposição à AOI
                overlap = overlap.intersection(aoi_geometry)
                if not overlap.is_empty and overlap.area > 0:
                    overlap_2_polygons.append(overlap)
    
    # Identificar áreas com sobreposição de 3+ imagens
    overlap_3_plus_areas = []
    
    for i, overlap1 in enumerate(overlap_2_polygons):
        for j, overlap2 in enumerate(overlap_2_polygons):
            if i >= j:  # Evitar duplicação
                continue
            # Se dois pares diferentes se sobrepõem, temos pelo menos 3 imagens se sobrepondo
            if overlap1.intersects(overlap2):
                overlap_3 = overlap1.intersection(overlap2)
                if not overlap_3.is_empty and overlap_3.area > 0:
                    overlap_3_plus_areas.append(overlap_3)
    
    # Processar áreas com sobreposição 3+ primeiro (para que apareçam por cima)
    for overlap in overlap_3_plus_areas:
        has_overlap_3 = True
        if hasattr(overlap, 'geoms'):
            for part in overlap.geoms:
                if part.geom_type == 'Polygon':
                    x, y = part.exterior.xy
                    # Usar uma cor diferente para sobreposições 3+
                    ax.fill(x, y, color='#FF9900', alpha=0.4, zorder=16)
                    ax.plot(x, y, color='#FF5500', linewidth=1.0, zorder=26)
        else:
            if overlap.geom_type == 'Polygon':
                x, y = overlap.exterior.xy
                ax.fill(x, y, color='#FF9900', alpha=0.4, zorder=16)
                ax.plot(x, y, color='#FF5500', linewidth=1.0, zorder=26)
    
    # Agora processar sobreposições 2-a-2, evitando áreas já marcadas como 3+
    for overlap_2 in overlap_2_polygons:
        # Subtrair as áreas de sobreposição 3+
        for overlap_3 in overlap_3_plus_areas:
            if overlap_2.intersects(overlap_3):
                overlap_2 = overlap_2.difference(overlap_3)
        
        # Plotar as áreas restantes como sobreposição 2-a-2
        if not overlap_2.is_empty:
            has_overlap_2 = True
            if hasattr(overlap_2, 'geoms'):
                for part in overlap_2.geoms:
                    if part.geom_type == 'Polygon':
                        x, y = part.exterior.xy
                        ax.fill(x, y, color='yellow', alpha=0.3, zorder=15)
                        ax.plot(x, y, color='#FF5733', linewidth=1.0, zorder=25)
            else:
                if overlap_2.geom_type == 'Polygon':
                    x, y = overlap_2.exterior.xy
                    ax.fill(x, y, color='yellow', alpha=0.3, zorder=15)
                    ax.plot(x, y, color='#FF5733', linewidth=1.0, zorder=25)
    
    # Desenhar as bounding boxes com linhas mais finas e elegantes
    # Mas apenas dentro da AOI
    for i, (geom, name, color) in enumerate(image_geometries):
        # Clipar a geometria para mostrar somente dentro da AOI
        clipped_geom = geom.intersection(aoi_geometry)
        if not clipped_geom.is_empty:
            if hasattr(clipped_geom, 'geoms'):
                for part in clipped_geom.geoms:
                    if part.geom_type == 'LineString':
                        x, y = part.xy
                        ax.plot(x, y, color=color, linewidth=0.7, alpha=0.7, zorder=20)
                    elif part.geom_type == 'Polygon':
                        x, y = part.exterior.xy
                        ax.plot(x, y, color=color, linewidth=0.7, alpha=0.7, zorder=20)
            else:
                if clipped_geom.geom_type == 'LineString':
                    x, y = clipped_geom.xy
                    ax.plot(x, y, color=color, linewidth=0.7, alpha=0.7, zorder=20)
                elif clipped_geom.geom_type == 'Polygon':
                    x, y = clipped_geom.exterior.xy
                    ax.plot(x, y, color=color, linewidth=0.7, alpha=0.7, zorder=20)
    
    # Contorno da AOI - linha muito mais fina
    aoi_gdf.boundary.plot(ax=ax, color='#e74c3c', linewidth=0.5, zorder=40)
    
    # Posicionar legenda no canto superior esquerdo com adaptação de cores
    # Algumas imagens podem ter fundo escuro, outras claro, então adaptamos
    legend_bg_color = 'white' if processed_images > 0 else 'white'
    
    # Usar a função de criação de legenda adaptativa
    leg = create_overlap_legend(ax, has_overlap_2, has_overlap_3)
    
    # Criar uma figura secundária apenas para a legenda
    figlegend = plt.figure(figsize=(2, 1), dpi=300)
    if hasattr(leg, 'legend_handles'):
        figlegend.legend(handles=leg.legend_handles, loc='center')
    else:
        handles, labels = ax.get_legend_handles_labels()
        figlegend.legend(handles=handles, loc='center')
    
    # legend_path = str(OUTPUT_DIR / f"{group_id}_legend.png")
    # figlegend.savefig(legend_path)
    plt.close(figlegend)
    
    # Configurar eixos - Sem título e sem marcadores
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    # Ajustar layout para melhor visualização
    plt.tight_layout(pad=0)
    
    # Salvar figura com alta qualidade
    output_file = f"{group_id}.jpg"
    output_path = OUTPUT_DIR / output_file
    
    # Salvar com fundo branco fora da AOI
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                format='jpeg', facecolor='white', edgecolor='none', 
                pad_inches=0.0)
    
    logging.info(f"Figura salva em: {output_path}")
    
    plt.close('all')
    
    # Limpar memória
    import gc
    gc.collect()

def main():
    # Obter número de mosaicos
    with open(JSON_FILE, 'r') as f:
        json_data = json.load(f)
    
    if isinstance(json_data, list):
        mosaics = json_data
    elif isinstance(json_data, dict) and 'mosaic_groups' in json_data:
        mosaics = json_data['mosaic_groups']
    else:
        logging.error("Formato do JSON não suportado")
        return
    
    max_mosaicos = len(mosaics)
    logging.info(f"Processando {max_mosaicos} mosaicos individualmente")
    
    # Processar cada mosaico individualmente
    for idx in range(max_mosaicos):
        plot_single_mosaic(idx, max_mosaicos)
        
        # Forçar coleta de lixo
        import gc
        gc.collect()
    
    print(f"Plotagem concluída: {max_mosaicos} mosaicos visualizados")

if __name__ == "__main__":
    main()
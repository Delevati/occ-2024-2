import os
import json
import glob
import re
import argparse
import tempfile
import zipfile
import shutil
import subprocess
from pathlib import Path
import time

# Diretórios principais
INPUT_DIR = "/Users/luryand/Documents/encode-image/coverage_otimization/code/output_log_cplex"
SHAPEFILE_DIR = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture"
IMAGES_BASE_DIR = "/Volumes/luryand/nova_busca"
OUTPUT_DIR = "/Users/luryand/Documents/encode-image/coverage_otimization/results/final_mosaics"

def find_state_shapefile(state_code):
    """Encontra o shapefile da área correspondente ao código do estado"""
    state_dir = os.path.join(SHAPEFILE_DIR, state_code)
    if os.path.exists(state_dir):
        shp_files = glob.glob(os.path.join(state_dir, "*.shp"))
        if shp_files:
            return shp_files[0]
    
    for root, dirs, files in os.walk(SHAPEFILE_DIR):
        for file in files:
            if file.endswith(".shp") and state_code in file:
                return os.path.join(root, file)
    
    return None

def extract_state_from_filename(filename):
    """Extrai o código do estado do nome do arquivo"""
    matches = re.findall(r'([A-Z]{2,3}(?:-[A-Z]{2,3})*)', filename)
    if matches:
        return matches[0]
    return None

def find_image_files(image_names, state_code):
    """Encontra os arquivos de imagem no SSD externo"""
    image_paths = []
    state_image_dir = os.path.join(IMAGES_BASE_DIR, state_code)
    
    if not os.path.exists(state_image_dir):
        print(f"Diretório de imagens não encontrado: {state_image_dir}")
        for dir_name in os.listdir(IMAGES_BASE_DIR):
            if state_code in dir_name:
                state_image_dir = os.path.join(IMAGES_BASE_DIR, dir_name)
                print(f"Usando diretório alternativo: {state_image_dir}")
                break
    
    for image_name in image_names:
        found = False
        for root, dirs, files in os.walk(state_image_dir):
            for file in files:
                if image_name in file or os.path.splitext(image_name)[0] in file:
                    image_paths.append(os.path.join(root, file))
                    found = True
                    break
            if found:
                break
        
        if not found:
            print(f"Imagem não encontrada: {image_name}")
    
    return image_paths

def check_jp2_support():
    """Verifica quais ferramentas estão disponíveis para conversão JP2"""
    tools = {}
    
    # Verifica GDAL com suporte JP2
    try:
        result = subprocess.run(['gdalinfo', '--formats'], capture_output=True, text=True)
        tools['gdal_jp2'] = 'JP2' in result.stdout or 'JPEG2000' in result.stdout
    except:
        tools['gdal_jp2'] = False
    
    # Verifica OpenJPEG
    try:
        subprocess.run(['opj_decompress', '-h'], capture_output=True, text=True)
        tools['openjpeg'] = True
    except:
        tools['openjpeg'] = False
    
    # Verifica ImageMagick
    try:
        subprocess.run(['convert', '-version'], capture_output=True, text=True)
        tools['imagemagick'] = True
    except:
        tools['imagemagick'] = False
    
    return tools

def convert_jp2_with_openjpeg(jp2_file, output_tif):
    """Converte JP2 para TIF usando OpenJPEG"""
    try:
        # Primeiro converte JP2 para PGM/PPM
        temp_pgm = jp2_file.replace('.jp2', '.pgm')
        
        result = subprocess.run([
            'opj_decompress',
            '-i', jp2_file,
            '-o', temp_pgm
        ], capture_output=True, text=True)
        
        if not os.path.exists(temp_pgm):
            return False
        
        # Depois converte PGM para TIF usando GDAL
        result = subprocess.run([
            'gdal_translate',
            '-of', 'GTiff',
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'PREDICTOR=2',
            temp_pgm, output_tif
        ], capture_output=True, text=True)
        
        # Limpa arquivo temporário
        if os.path.exists(temp_pgm):
            os.remove(temp_pgm)
        
        return os.path.exists(output_tif) and os.path.getsize(output_tif) > 1000
        
    except Exception as e:
        print(f"    Erro com OpenJPEG: {e}")
        return False

def convert_jp2_with_imagemagick(jp2_file, output_tif):
    """Converte JP2 para TIF usando ImageMagick"""
    try:
        result = subprocess.run([
            'convert',
            jp2_file,
            '-compress', 'lzw',
            output_tif
        ], capture_output=True, text=True)
        
        return os.path.exists(output_tif) and os.path.getsize(output_tif) > 1000
        
    except Exception as e:
        print(f"    Erro com ImageMagick: {e}")
        return False

def extract_and_convert_jp2(zip_file, work_dir):
    """Extrai e converte JP2 para TIF usando múltiplas estratégias"""
    print(f"Processando: {os.path.basename(zip_file)}")
    
    # Verifica ferramentas disponíveis uma única vez
    if not hasattr(extract_and_convert_jp2, '_tools_checked'):
        extract_and_convert_jp2._available_tools = check_jp2_support()
        extract_and_convert_jp2._tools_checked = True
        
        print(f"  Ferramentas JP2 disponíveis:")
        for tool, available in extract_and_convert_jp2._available_tools.items():
            status = "✓" if available else "✗"
            print(f"    {status} {tool}")
    
    tools = extract_and_convert_jp2._available_tools
    
    # Define nomes para arquivos
    base_name = os.path.basename(zip_file).replace('.zip', '').replace('.SAFE', '')
    extract_dir = os.path.join(work_dir, f"{base_name}_extract")
    os.makedirs(extract_dir, exist_ok=True)
    
    # Arquivos de saída
    red_tif = os.path.join(work_dir, f"{base_name}_red.tif")
    green_tif = os.path.join(work_dir, f"{base_name}_green.tif")
    blue_tif = os.path.join(work_dir, f"{base_name}_blue.tif")
    rgb_tif = os.path.join(work_dir, f"{base_name}_rgb.tif")
    
    try:
        # Extrai os arquivos JP2 necessários
        with zipfile.ZipFile(zip_file, 'r') as zf:
            jp2_files = [f for f in zf.namelist() if f.endswith('.jp2') and 
                        ('_B04_10m.jp2' in f or '_B03_10m.jp2' in f or '_B02_10m.jp2' in f) and
                        not f.startswith('MSK_')]
            
            # Dicionário para armazenar os caminhos
            band_files = {'red': None, 'green': None, 'blue': None}
            
            # Extrai cada arquivo JP2
            for jp2_file in jp2_files:
                file_name = os.path.basename(jp2_file)
                output_file = os.path.join(extract_dir, file_name)
                
                # Extrai o arquivo
                with zf.open(jp2_file) as source, open(output_file, 'wb') as target:
                    shutil.copyfileobj(source, target)
                
                print(f"  Extraído: {file_name}")
                
                # Identifica a banda
                if '_B04_10m' in file_name:  # Vermelho
                    band_files['red'] = output_file
                elif '_B03_10m' in file_name:  # Verde
                    band_files['green'] = output_file
                elif '_B02_10m' in file_name:  # Azul
                    band_files['blue'] = output_file
        
        # Verifica se todas as bandas foram encontradas
        if not all(band_files.values()):
            missing = [b for b, f in band_files.items() if f is None]
            print(f"  Erro: Bandas não encontradas: {', '.join(missing)}")
            return None
        
        # Converte cada banda para TIF usando estratégias múltiplas
        success_count = 0
        for band, jp2_file in band_files.items():
            output_tif = locals()[f"{band}_tif"]
            converted = False
            
            print(f"  Convertendo banda {band}...")
            
            # Estratégia 1: GDAL direto (se suporta JP2)
            if tools['gdal_jp2'] and not converted:
                print(f"    Tentando com GDAL...")
                result = subprocess.run([
                    'gdal_translate',
                    '-of', 'GTiff',
                    '-co', 'COMPRESS=DEFLATE',
                    '-co', 'PREDICTOR=2',
                    jp2_file, output_tif
                ], capture_output=True, text=True)
                
                if os.path.exists(output_tif) and os.path.getsize(output_tif) > 1000:
                    converted = True
                    print(f"    ✓ Sucesso com GDAL")
                else:
                    if os.path.exists(output_tif):
                        os.remove(output_tif)
            
            # Estratégia 2: OpenJPEG + GDAL
            if tools['openjpeg'] and not converted:
                print(f"    Tentando com OpenJPEG...")
                if convert_jp2_with_openjpeg(jp2_file, output_tif):
                    converted = True
                    print(f"    ✓ Sucesso com OpenJPEG")
                else:
                    if os.path.exists(output_tif):
                        os.remove(output_tif)
            
            # Estratégia 3: ImageMagick
            if tools['imagemagick'] and not converted:
                print(f"    Tentando com ImageMagick...")
                if convert_jp2_with_imagemagick(jp2_file, output_tif):
                    converted = True
                    print(f"    ✓ Sucesso com ImageMagick")
                else:
                    if os.path.exists(output_tif):
                        os.remove(output_tif)
            
            if converted:
                success_count += 1
                print(f"  ✓ Banda {band} convertida com sucesso")
            else:
                print(f"  ✗ Falha ao converter banda {band}")
                # Tenta verificar o conteúdo do arquivo JP2
                file_size = os.path.getsize(jp2_file) if os.path.exists(jp2_file) else 0
                print(f"    Tamanho do arquivo JP2: {file_size} bytes")
                
                # Lista primeiros bytes para debug
                if file_size > 0:
                    with open(jp2_file, 'rb') as f:
                        header = f.read(20)
                        print(f"    Cabeçalho do arquivo: {header.hex()}")
        
        # Verifica se pelo menos algumas bandas foram convertidas
        if success_count == 0:
            print(f"  ✗ Nenhuma banda foi convertida com sucesso")
            return None
        elif success_count < 3:
            print(f"  ⚠ Apenas {success_count}/3 bandas convertidas")
            # Continua mesmo assim, pode funcionar
        
        # Combina as bandas em uma imagem RGB
        print("  Combinando bandas em RGB...")
        valid_tifs = [tif for tif in [red_tif, green_tif, blue_tif] if os.path.exists(tif)]
        
        if len(valid_tifs) < 2:
            print(f"  Erro: Bandas insuficientes para RGB ({len(valid_tifs)}/3)")
            return None
        
        result = subprocess.run([
            'gdal_merge.py',
            '-separate',
            '-o', rgb_tif,
            '-co', 'PHOTOMETRIC=RGB',
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'PREDICTOR=2'
        ] + valid_tifs, capture_output=True, text=True)
        
        if not os.path.exists(rgb_tif) or os.path.getsize(rgb_tif) < 1000:
            print(f"  Erro ao combinar bandas em RGB")
            print(f"  Detalhes: {result.stderr}")
            return None
        
        # Aplica melhoria de contraste
        enhanced_rgb = os.path.join(work_dir, f"{base_name}_enhanced.tif")
        result = subprocess.run([
            'gdal_translate',
            '-scale', '0', '2000', '0', '255',  # Ajuste de contraste
            '-exponent', '0.85',  # Ajuste de gamma
            '-ot', 'Byte',  # Converte para 8-bit
            '-a_nodata', '0',
            rgb_tif, enhanced_rgb
        ], capture_output=True, text=True)
        
        if os.path.exists(enhanced_rgb) and os.path.getsize(enhanced_rgb) > 1000:
            print(f"  ✓ RGB criado com sucesso: {os.path.basename(enhanced_rgb)}")
            return enhanced_rgb
        else:
            print(f"  ⚠ Falha na melhoria, usando RGB básico")
            return rgb_tif if os.path.exists(rgb_tif) else None
            
    except Exception as e:
        print(f"  Erro ao processar {base_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_mosaic(rgb_images, output_file, shapefile=None):
    """Cria um mosaico a partir de imagens RGB"""
    if not rgb_images:
        print("Nenhuma imagem para criar mosaico")
        return None
    
    print(f"Criando mosaico com {len(rgb_images)} imagens...")
    
    # Cria o mosaico
    temp_mosaic = f"{output_file}_temp.tif"
    
    # Usa gdal_merge para criar o mosaico
    cmd = ['gdal_merge.py', '-o', temp_mosaic, 
          '-of', 'GTiff',
          '-co', 'COMPRESS=DEFLATE',
          '-co', 'PREDICTOR=2',
          '-n', '0']  # Define 0 como valor nodata
    
    # Adiciona as imagens
    cmd.extend(rgb_images)
    
    print(f"Executando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if not os.path.exists(temp_mosaic) or os.path.getsize(temp_mosaic) < 1000:
        print(f"Erro ao criar mosaico")
        print(f"Detalhes: {result.stderr}")
        return None
    
    # Reorienta o mosaico para norte acima
    oriented_mosaic = f"{output_file}_oriented.tif"
    result = subprocess.run([
        'gdalwarp',
        '-t_srs', 'EPSG:4326',  # WGS84
        '-r', 'bilinear',
        '-co', 'COMPRESS=DEFLATE',
        '-co', 'PREDICTOR=2',
        '-multi',
        temp_mosaic, oriented_mosaic
    ], capture_output=True, text=True)
    
    if not os.path.exists(oriented_mosaic) or os.path.getsize(oriented_mosaic) < 1000:
        print(f"Erro ao reorientar mosaico")
        print(f"Detalhes: {result.stderr}")
        # Usa o mosaico não orientado
        oriented_mosaic = temp_mosaic
    
    # Recorta com shapefile se fornecido
    final_output = f"{output_file}.tif"
    if shapefile and os.path.exists(shapefile):
        print(f"Recortando com shapefile: {os.path.basename(shapefile)}")
        result = subprocess.run([
            'gdalwarp',
            '-cutline', shapefile,
            '-crop_to_cutline',
            '-dstalpha',
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'PREDICTOR=2',
            oriented_mosaic, final_output
        ], capture_output=True, text=True)
        
        if not os.path.exists(final_output) or os.path.getsize(final_output) < 1000:
            print(f"Erro ao recortar com shapefile")
            print(f"Detalhes: {result.stderr}")
            # Usa o mosaico sem recorte
            os.rename(oriented_mosaic, final_output)
    else:
        # Se não tem shapefile, usa o mosaico orientado
        os.rename(oriented_mosaic, final_output)
    
    # Limpa arquivos temporários
    for file in [temp_mosaic, oriented_mosaic]:
        if os.path.exists(file) and file != final_output:
            os.remove(file)
    
    # Gera uma visualização
    preview_png = f"{output_file}_preview.png"
    result = subprocess.run([
        'gdal_translate',
        '-of', 'PNG',
        '-outsize', '800', '0',
        final_output, preview_png
    ], capture_output=True, text=True)
    
    if os.path.exists(preview_png):
        print(f"Visualização criada: {preview_png}")
    
    print(f"Mosaico criado com sucesso: {final_output}")
    return final_output

def process_selected_mosaics(json_file):
    """Processa os mosaicos selecionados de um arquivo JSON"""
    # Cria diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extrai código do estado
    state_code = extract_state_from_filename(os.path.basename(json_file))
    if not state_code:
        print(f"Não foi possível identificar o estado: {json_file}")
        return
    
    print(f"\n{'='*50}")
    print(f"Processando mosaicos para: {state_code}")
    print(f"{'='*50}")
    
    # Encontra o shapefile
    shapefile_path = find_state_shapefile(state_code)
    if shapefile_path:
        print(f"Usando shapefile: {shapefile_path}")
    else:
        print("Shapefile não encontrado - mosaico não será recortado")
    
    # Carrega o JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if "mosaic_groups" not in data:
        print(f"Formato JSON inválido: {json_file}")
        return
    
    # Processa cada grupo
    for i, group in enumerate(data["mosaic_groups"]):
        group_id = group.get("group_id", f"group_{i}")
        images = group.get("images", [])
        
        if not images:
            print(f"Nenhuma imagem no grupo: {group_id}")
            continue
        
        print(f"\n{'-'*50}")
        print(f"Processando grupo: {group_id} com {len(images)} imagens")
        print(f"{'-'*50}")
        
        # Encontra os arquivos
        image_paths = find_image_files(images, state_code)
        if not image_paths:
            print(f"Nenhuma imagem encontrada para: {group_id}")
            continue
        
        print(f"Encontradas {len(image_paths)} de {len(images)} imagens")
        
        # Cria diretório temporário para processamento
        with tempfile.TemporaryDirectory() as temp_dir:
            # Processa cada imagem
            rgb_images = []
            
            for img_path in image_paths:
                if img_path.endswith('.zip'):
                    rgb_tif = extract_and_convert_jp2(img_path, temp_dir)
                    if rgb_tif and os.path.exists(rgb_tif):
                        rgb_images.append(rgb_tif)
            
            # Cria o mosaico
            if rgb_images:
                output_name = f"{state_code}_{group_id.replace('selected_mosaics_', '')}"
                output_path = os.path.join(OUTPUT_DIR, output_name)
                
                final_mosaic = create_mosaic(rgb_images, output_path, shapefile_path)
                
                if final_mosaic and os.path.exists(final_mosaic):
                    print(f"\nResultados:")
                    print(f"  - Mosaico: {final_mosaic}")
                    print(f"  - Visualização: {output_path}_preview.png")
                else:
                    print("Falha ao criar mosaico final")
            else:
                print("Nenhuma imagem processada com sucesso")

def main():
    """Função principal do script"""
    print("=" * 80)
    print("GERADOR DE MOSAICOS RGB SENTINEL-2")
    print("=" * 80)
    
    # Verifica ferramentas disponíveis
    print("\n1. Verificando ferramentas JP2...")
    tools = check_jp2_support()
    
    available_tools = [tool for tool, available in tools.items() if available]
    if not available_tools:
        print("\n⚠️  AVISO: Nenhuma ferramenta JP2 detectada!")
        print("\nPara instalar ferramentas JP2:")
        print("• OpenJPEG: brew install openjpeg")
        print("• ImageMagick: brew install imagemagick")
        print("• GDAL com JP2: conda install -c conda-forge gdal=3.8.3")
        print("\nRecomendação: Execute um dos comandos acima e tente novamente.")
        
        # Pergunta se deve continuar mesmo assim
        response = input("\nDeseja continuar mesmo assim? (s/n): ").lower()
        if response != 's':
            return
    else:
        print(f"✓ Ferramentas encontradas: {', '.join(available_tools)}")
    
    # Configurações
    base_dir = "/Users/luryand/Documents/encode-image/coverage_otimization"
    images_base = "/Volumes/luryand/nova_busca"
    output_dir = os.path.join(base_dir, "results", "final_mosaics")
    json_dir = os.path.join(base_dir, "code", "output_log_cplex")
    shapefile_dir = os.path.join(base_dir, "code", "APA-input", "recapture")
    
    # Cria diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n2. Configurações:")
    print(f"   Base de imagens: {images_base}")
    print(f"   Saída: {output_dir}")
    print(f"   JSONs CPLEX: {json_dir}")
    print(f"   Shapefiles: {shapefile_dir}")
    
    # Lista arquivos JSON de resultados do CPLEX
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json') and 
                  ('AL_' in f or 'BA_' in f or 'MG_' in f or 'RS_' in f or 'PI-PE-CE' in f)]
    
    if not json_files:
        print("❌ Nenhum arquivo JSON encontrado!")
        return
    
    print(f"\n3. Arquivos JSON encontrados: {len(json_files)}")
    for json_file in json_files:
        print(f"   • {json_file}")
    
    # Processa cada área
    total_processed = 0
    total_success = 0
    
    for json_file in json_files:
        area_name = extract_area_from_filename(json_file)
        if not area_name:
            print(f"\n❌ Não foi possível determinar área para: {json_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"PROCESSANDO ÁREA: {area_name}")
        print(f"{'='*60}")
        
        json_path = os.path.join(json_dir, json_file)
        selected_images = load_selected_images(json_path)
        
        if not selected_images:
            print(f"❌ Nenhuma imagem selecionada encontrada em {json_file}")
            continue
        
        # Determina o diretório de imagens para esta área
        area_image_dir = get_area_image_directory(area_name, images_base)
        if not area_image_dir:
            print(f"❌ Diretório de imagens não encontrado para área {area_name}")
            continue
        
        # Encontra shapefile para a área
        shapefile = find_shapefile_for_area(area_name, shapefile_dir)
        if not shapefile:
            print(f"❌ Shapefile não encontrado para área {area_name}")
            continue
        
        # Processa a área
        total_processed += 1
        if generate_mosaic_for_area(area_name, selected_images, area_image_dir, shapefile, output_dir):
            total_success += 1
            print(f"✅ Mosaico criado com sucesso para {area_name}")
        else:
            print(f"❌ Falha ao criar mosaico para {area_name}")
    
    # Resumo final
    print(f"\n{'='*80}")
    print(f"RESUMO FINAL")
    print(f"{'='*80}")
    print(f"Áreas processadas: {total_processed}")
    print(f"Mosaicos criados: {total_success}")
    print(f"Taxa de sucesso: {(total_success/total_processed*100) if total_processed > 0 else 0:.1f}%")
    
    if total_success > 0:
        print(f"\n🎉 Mosaicos salvos em: {output_dir}")
        print("Cada área possui:")
        print("  • [AREA]_mosaic.tif - Mosaico RGB completo")
        print("  • [AREA]_mosaic_clipped.tif - Mosaico recortado pelo shapefile")
        print("  • [AREA]_preview.png - Visualização rápida")
    
    if total_success < total_processed:
        print(f"\n⚠️  {total_processed - total_success} áreas falharam.")
        print("Verifique os logs acima para detalhes dos erros.")

if __name__ == "__main__":
    main()

def extract_area_from_filename(filename):
    """Extrai o nome da área do nome do arquivo JSON"""
    if 'AL_' in filename:
        return 'AL'
    elif 'BA_' in filename:
        return 'BA'
    elif 'MG_' in filename and 'MG-SP-RJ' not in filename:
        return 'MG'
    elif 'MG-SP-RJ' in filename:
        return 'MG-SP-RJ'
    elif 'PI-PE-CE' in filename:
        return 'PI-PE-CE'
    elif 'RS_' in filename:
        return 'RS'
    return None

def load_selected_images(json_path):
    """Carrega lista de imagens selecionadas do arquivo JSON do CPLEX"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Procura por diferentes estruturas possíveis no JSON
        if 'selected_mosaics' in data:
            return data['selected_mosaics']
        elif 'mosaics' in data:
            return data['mosaics']
        elif isinstance(data, list):
            return data
        else:
            # Procura por chaves que contenham nomes de imagem
            for key, value in data.items():
                if isinstance(value, list) and value:
                    # Verifica se o primeiro item parece um nome de imagem Sentinel
                    first_item = value[0]
                    if isinstance(first_item, str) and ('S2' in first_item or 'MSIL2A' in first_item):
                        return value
        
        return []
        
    except Exception as e:
        print(f"Erro ao carregar JSON {json_path}: {e}")
        return []

def get_area_image_directory(area_name, images_base):
    """Determina o diretório de imagens para uma área"""
    area_dirs = {
        'AL': 'AL',
        'BA': 'BA', 
        'MG': 'MG',
        'MG-SP-RJ': 'MG-SP-RJ',
        'PI-PE-CE': 'PI-PE-CE',
        'RS': 'RS'
    }
    
    if area_name in area_dirs:
        area_dir = os.path.join(images_base, area_dirs[area_name])
        if os.path.exists(area_dir):
            return area_dir
    
    return None

def find_shapefile_for_area(area_name, shapefile_dir):
    """Encontra o shapefile correspondente à área"""
    area_mapping = {
        'AL': 'AL',
        'BA': 'BA',
        'MG': 'MG', 
        'MG-SP-RJ': 'MG',  # Usa shapefile do MG
        'PI-PE-CE': 'PI',  # Usa shapefile do PI
        'RS': 'RS'
    }
    
    if area_name not in area_mapping:
        return None
    
    shapefile_state = area_mapping[area_name]
    state_dir = os.path.join(shapefile_dir, shapefile_state)
    
    if not os.path.exists(state_dir):
        return None
    
    # Procura por arquivo .shp
    for file in os.listdir(state_dir):
        if file.endswith('.shp'):
            return os.path.join(state_dir, file)
    
    return None

def generate_mosaic_for_area(area_name, selected_images, area_image_dir, shapefile, output_dir):
    """Gera mosaico para uma área específica"""
    print(f"\nIniciando geração de mosaico para {area_name}...")
    print(f"Imagens selecionadas: {len(selected_images)}")
    print(f"Diretório de imagens: {area_image_dir}")
    print(f"Shapefile: {shapefile}")
    
    # Cria diretório de trabalho temporário
    work_dir = os.path.join(output_dir, f"temp_{area_name}")
    os.makedirs(work_dir, exist_ok=True)
    
    # Lista para armazenar TIFs processados
    processed_tifs = []
    
    try:
        # Processa cada imagem selecionada
        for i, image_name in enumerate(selected_images):
            print(f"\n  Processando imagem {i+1}/{len(selected_images)}: {image_name}")
            
            # Encontra o arquivo .zip correspondente
            zip_pattern = f"{image_name}*.zip"
            zip_files = glob.glob(os.path.join(area_image_dir, zip_pattern))
            
            if not zip_files:
                print(f"    ⚠ Arquivo ZIP não encontrado para: {image_name}")
                continue
            
            zip_file = zip_files[0]
            
            # Extrai e converte para RGB TIF
            rgb_tif = extract_and_convert_jp2(zip_file, work_dir)
            if rgb_tif:
                processed_tifs.append(rgb_tif)
                print(f"    ✓ Processado: {os.path.basename(rgb_tif)}")
            else:
                print(f"    ✗ Falha ao processar: {image_name}")
        
        if not processed_tifs:
            print(f"❌ Nenhuma imagem foi processada com sucesso para {area_name}")
            return False
        
        print(f"\n  {len(processed_tifs)} imagens processadas com sucesso")
        
        # Cria mosaico
        mosaic_file = os.path.join(output_dir, f"{area_name}_mosaic.tif")
        print(f"  Criando mosaico: {os.path.basename(mosaic_file)}")
        
        if len(processed_tifs) == 1:
            # Apenas uma imagem, apenas copia
            shutil.copy2(processed_tifs[0], mosaic_file)
        else:
            # Múltiplas imagens, cria mosaico
            result = subprocess.run([
                'gdal_merge.py',
                '-o', mosaic_file,
                '-co', 'COMPRESS=DEFLATE',
                '-co', 'PREDICTOR=2',
                '-n', '0',  # NoData value
                '-a_nodata', '0'
            ] + processed_tifs, capture_output=True, text=True)
            
            if not os.path.exists(mosaic_file):
                print(f"    ✗ Falha ao criar mosaico")
                print(f"    Erro: {result.stderr}")
                return False
        
        # Reprojecta para EPSG:4326 e garante orientação norte-up
        reprojected_file = os.path.join(output_dir, f"{area_name}_mosaic_reprojected.tif")
        result = subprocess.run([
            'gdalwarp',
            '-t_srs', 'EPSG:4326',
            '-r', 'bilinear',
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'PREDICTOR=2',
            '-dstnodata', '0',
            mosaic_file, reprojected_file
        ], capture_output=True, text=True)
        
        if os.path.exists(reprojected_file):
            os.replace(reprojected_file, mosaic_file)
            print(f"  ✓ Mosaico reprojetado para EPSG:4326")
        
        # Recorta usando shapefile
        clipped_file = os.path.join(output_dir, f"{area_name}_mosaic_clipped.tif")
        result = subprocess.run([
            'gdalwarp',
            '-cutline', shapefile,
            '-crop_to_cutline',
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'PREDICTOR=2',
            '-dstnodata', '0',
            mosaic_file, clipped_file
        ], capture_output=True, text=True)
        
        if not os.path.exists(clipped_file):
            print(f"    ⚠ Falha no recorte, mantendo mosaico original")
            shutil.copy2(mosaic_file, clipped_file)
        else:
            print(f"  ✓ Mosaico recortado pelo shapefile")
        
        # Cria preview PNG
        preview_file = os.path.join(output_dir, f"{area_name}_preview.png")
        result = subprocess.run([
            'gdal_translate',
            '-of', 'PNG',
            '-outsize', '1024', '1024',
            '-scale',
            clipped_file, preview_file
        ], capture_output=True, text=True)
        
        if os.path.exists(preview_file):
            print(f"  ✓ Preview criado: {os.path.basename(preview_file)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao processar área {area_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Limpa diretório temporário
        if os.path.exists(work_dir):
            try:
                shutil.rmtree(work_dir)
            except:
                pass  # Ignora erros de limpeza
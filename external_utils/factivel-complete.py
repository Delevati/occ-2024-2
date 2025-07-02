import os
import glob
import re
from PIL import Image

# Parâmetros
IMAGES_DIR = "/Volumes/luryand/mosaicos_plot"
OUTPUT_JPG = "/Volumes/luryand/painel_final.jpg"
N_COLS = 3  # Fixado em 4 colunas
ROW_SPACING = 80  # pixels de espaçamento entre as linhas
COL_SPACING = 80  # pixels de espaçamento entre colunas
JPEG_QUALITY = 10  # Alta qualidade, valor de 1 a 100

# Função para extrair o ID numérico do nome do arquivo
def extract_mosaic_id(filename):
    # Usa expressão regular para extrair o número após "mosaic_" ou similar
    match = re.search(r'mosaic_(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))  # Converte para inteiro para ordenar numericamente
    return 0  # Valor padrão se não encontrar um ID

# Buscar arquivos JPG
image_files = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
if not image_files:
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))
if not image_files:
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*_simple.jpg"))
    if not image_files:
        image_files = glob.glob(os.path.join(IMAGES_DIR, "*_simple.jpeg"))

# Verificar se temos arquivos para processar
if not image_files:
    print(f"Não foram encontradas imagens JPG em {IMAGES_DIR}")
    exit(1)

# Ordenar por ID numérico do mosaico (em vez da ordenação alfabética)
image_files.sort(key=extract_mosaic_id)
print(f"Encontradas {len(image_files)} imagens JPG para processar")

# Abrir imagens na ordem por ID
imgs = []
for f in image_files:
    try:
        img = Image.open(f).convert("RGB")
        imgs.append(img)
        mosaic_id = extract_mosaic_id(f)
        print(f"Carregada: {os.path.basename(f)} - ID: {mosaic_id} - {img.size}")
    except Exception as e:
        print(f"Erro ao abrir {f}: {e}")

# Calcular número de linhas necessárias com 4 colunas fixas
n_images = len(imgs)
if n_images == 0:
    print("Nenhuma imagem válida foi carregada")
    exit(1)

# Calcular o número de linhas necessárias com base no número fixo de colunas
n_rows = (n_images + N_COLS - 1) // N_COLS  # Arredonda para cima

# Determinar tamanho máximo das imagens
max_width = max(img.width for img in imgs)
max_height = max(img.height for img in imgs)

# Calcular dimensões do painel
painel_width = N_COLS * max_width + (N_COLS - 1) * COL_SPACING
painel_height = n_rows * max_height + (n_rows - 1) * ROW_SPACING

# Criar painel vazio (com fundo branco)
painel = Image.new('RGB', (painel_width, painel_height), color=(255, 255, 255))

# Preencher painel com imagens
for i, img in enumerate(imgs):
    row = i // N_COLS
    col = i % N_COLS
    
    # Calcular posição x, y
    x = col * (max_width + COL_SPACING)
    y = row * (max_height + ROW_SPACING)
    
    # Centralizar a imagem em seu espaço alocado
    x_offset = (max_width - img.width) // 2
    y_offset = (max_height - img.height) // 2
    
    # Colar a imagem no painel
    painel.paste(img, (x + x_offset, y + y_offset))
    print(f"Imagem {extract_mosaic_id(image_files[i])} posicionada em ({x+x_offset}, {y+y_offset})")

# Salvar painel como arquivo JPEG
try:
    painel.save(
        OUTPUT_JPG,
        "JPEG",
        quality=JPEG_QUALITY,
        optimize=True
    )
    print(f"Painel salvo como {OUTPUT_JPG}")
    print(f"Dimensões: {painel_width}x{painel_height}, Qualidade: {JPEG_QUALITY}")
except Exception as e:
    print(f"Erro ao salvar o painel: {e}")

print("Processamento finalizado.")
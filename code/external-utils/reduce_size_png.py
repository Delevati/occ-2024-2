import os
from PIL import Image
import sys # To get command line arguments if needed

# --- Parâmetros ---
# Option 1: Hardcode the input file
INPUT_IMAGE_PATH = "/Users/luryand/Documents/encode-image/coverage_otimization/artigo-sbpo/painel_final.png"
# Option 2: Get from command line (uncomment below and run as: python3 code/reduce_size_png.py <path_to_image>)
# if len(sys.argv) < 2:
#     print("Usage: python3 reduce_size_png.py <path_to_image>")
#     sys.exit(1)
# INPUT_IMAGE_PATH = sys.argv[1]

# Construct output name based on input name
input_dir = os.path.dirname(INPUT_IMAGE_PATH)
input_filename = os.path.basename(INPUT_IMAGE_PATH)
input_name_part, input_ext = os.path.splitext(input_filename)
OUTPUT_BASE_NAME = os.path.join(input_dir, f"{input_name_part}_reduced") # Output in the same dir with suffix

# --- Opções de Saída ---
OUTPUT_FORMAT = "JPEG"  # Mude para "JPEG" se preferir (perde transparência)
RESIZE_FACTOR = 0.3    # Fator para redimensionar a imagem final (1.0 = sem redimensionamento)

# Opções PNG (usadas se OUTPUT_FORMAT = "PNG")
PNG_COMPRESSION_LEVEL = 9 # Nível de compressão PNG (0-9, 9 é máximo)
PNG_OPTIMIZE = True       # Tenta otimizar a codificação PNG

# Opções JPEG (usadas se OUTPUT_FORMAT = "JPEG")
JPEG_QUALITY = 75         # Qualidade JPEG (1-95, ~75-85 é um bom equilíbrio)
JPEG_BACKGROUND_COLOR = (255, 255, 255) # Cor de fundo para substituir transparência

# --- Início do Script ---

if not os.path.exists(INPUT_IMAGE_PATH):
    print(f"Erro: Arquivo de entrada não encontrado: {INPUT_IMAGE_PATH}")
    sys.exit(1)

# Abre a imagem única
try:
    print(f"Abrindo imagem: {INPUT_IMAGE_PATH}...")
    # Use 'RGBA' for PNG to preserve transparency, 'RGB' might be needed before JPEG save
    img = Image.open(INPUT_IMAGE_PATH)
    # Ensure it's in a modifiable format (e.g., RGBA for potential transparency)
    if OUTPUT_FORMAT.upper() == 'PNG' and img.mode != 'RGBA':
         print(f"Convertendo imagem para RGBA (modo original: {img.mode})")
         img = img.convert("RGBA")
    elif OUTPUT_FORMAT.upper() == 'JPEG' and img.mode == 'P': # Handle Paletted images for JPEG
         print(f"Convertendo imagem paletada para RGB")
         img = img.convert("RGB")


except Exception as e:
    print(f"Erro ao abrir a imagem: {e}")
    sys.exit(1)

original_width, original_height = img.size
print(f"Dimensões originais: {original_width}x{original_height}")

# --- Controle de Qualidade/Tamanho ---

# 1. Redimensionar (Opcional)
processed_img = img # Start with the original image
if RESIZE_FACTOR < 1.0 and RESIZE_FACTOR > 0:
    new_width = int(original_width * RESIZE_FACTOR)
    new_height = int(original_height * RESIZE_FACTOR)
    print(f"Redimensionando imagem para {new_width}x{new_height} (fator: {RESIZE_FACTOR})...")
    try:
        processed_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Erro durante o redimensionamento: {e}")
        sys.exit(1)
elif RESIZE_FACTOR != 1.0:
    print("Aviso: RESIZE_FACTOR inválido. Usando 1.0 (sem redimensionamento).")
else:
    print("Sem redimensionamento (RESIZE_FACTOR = 1.0).")


# 2. Salvar no formato escolhido
output_format_upper = OUTPUT_FORMAT.upper()

if output_format_upper == "JPEG":
    output_path = f"{OUTPUT_BASE_NAME}.jpg"
    print(f"Preparando para salvar como JPEG (qualidade: {JPEG_QUALITY})...")

    # Ensure image is RGB before saving as JPEG
    if processed_img.mode == 'RGBA':
        print("Convertendo RGBA para RGB (removendo transparência)...")
        # Create a background image with the specified color
        rgb_image = Image.new("RGB", processed_img.size, JPEG_BACKGROUND_COLOR)
        # Paste the RGBA image onto the background using its alpha channel as mask
        try:
            mask = processed_img.split()[3] # Get alpha channel
            rgb_image.paste(processed_img, mask=mask)
            processed_img = rgb_image # Replace with the flattened image
        except IndexError:
             print("Aviso: Imagem não possui canal alfa para máscara. Convertendo diretamente.")
             processed_img = processed_img.convert("RGB")
        except Exception as e:
             print(f"Erro ao aplicar máscara alfa para conversão RGB: {e}. Convertendo diretamente.")
             processed_img = processed_img.convert("RGB")

    elif processed_img.mode != 'RGB':
         print(f"Convertendo modo {processed_img.mode} para RGB...")
         processed_img = processed_img.convert("RGB")


    try:
        print(f"Salvando em: {output_path}")
        processed_img.save(
            output_path,
            "JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
            progressive=True
        )
        print("Imagem salva como JPEG com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar a imagem como JPEG: {e}")

elif output_format_upper == "PNG":
    output_path = f"{OUTPUT_BASE_NAME}.png"
    print(f"Preparando para salvar como PNG (compressão: {PNG_COMPRESSION_LEVEL}, otimização: {PNG_OPTIMIZE})...")

    # Ensure RGBA if original had transparency, otherwise keep original mode if possible
    if img.mode == 'RGBA' and processed_img.mode != 'RGBA':
         print("Convertendo de volta para RGBA para salvar PNG com transparência.")
         processed_img = processed_img.convert("RGBA") # Should ideally not happen if resize keeps mode
    elif img.mode != 'RGBA' and processed_img.mode == 'RGBA':
         print("Aviso: Imagem foi convertida para RGBA mas original não era. Salvando como RGBA.")


    try:
        print(f"Salvando em: {output_path}")
        processed_img.save(
            output_path,
            "PNG",
            compress_level=PNG_COMPRESSION_LEVEL,
            optimize=PNG_OPTIMIZE
        )
        print("Imagem salva como PNG com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar a imagem como PNG: {e}")
else:
    print(f"Erro: Formato de saída não suportado: {OUTPUT_FORMAT}. Use 'PNG' ou 'JPEG'.")

print("Script finalizado.")
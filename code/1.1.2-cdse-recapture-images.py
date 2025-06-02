import os
import json
import certifi
import pandas as pd
from cdsetool.credentials import Credentials
from cdsetool.download import download_features
from cdsetool.monitor import StatusMonitor
from cdsetool.query import query_features
from dotenv import load_dotenv
from datetime import date
import geopandas as gpd

# Carrega variáveis de ambiente
load_dotenv('/Users/luryand/Documents/encode-image/coverage_otimization/code/.env')

# Definições de caminhos
json_path = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/recapture/MG/optimization_parameters.json"
download_path = "/Volumes/luryand/nova_busca/MG"
relatorio_csv_path = "relatorio_downloads_especificos.csv"

# Configurações SSL
os.environ['SSL_CERT_FILE'] = certifi.where()

def extract_image_filenames(json_path):
    """Extrai nomes de arquivo dos mosaicos no JSON"""
    print(f"Carregando arquivo JSON: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        needed_images = set()
        
        # Extrai filenames dos grupos de mosaico
        if 'mosaic_groups' in data:
            for group in data['mosaic_groups']:
                if 'images' in group:
                    needed_images.update(group['images'])
        
        print(f"Encontradas {len(needed_images)} imagens necessárias para os mosaicos")
        return needed_images
    except Exception as e:
        print(f"Erro ao processar JSON: {e}")
        return set()

def file_to_wkt(file_path):
    """Converte arquivo de geometria para WKT"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.shp':
        gdf = gpd.read_file(file_path)
    elif ext == '.kml':
        gdf = gpd.read_file(file_path, driver='KML')
    else:
        raise ValueError(f"Formato de arquivo '{ext}' não suportado. Use KML ou SHP.")
    
    return gdf.geometry.iloc[0].wkt

def download_specific_files():
    """Função principal para baixar arquivos específicos"""
    # Extrai os nomes dos arquivos específicos
    needed_images = extract_image_filenames(json_path)
    if not needed_images:
        print("Nenhum arquivo para baixar. Verifique o JSON.")
        return
    
    # Salva lista para referência
    with open('images_to_download.txt', 'w') as f:
        for img in sorted(needed_images):
            f.write(f"{img}\n")
    print(f"Lista de {len(needed_images)} imagens salva em images_to_download.txt")
    
    # Credenciais
    credentials = Credentials(os.getenv('CDSE_USER'), os.getenv('CDSE_PASSWORD'))
    
    # Verifica quais arquivos já existem
    existing_files = set()
    for root, dirs, files in os.walk(download_path):
        for file in files:
            if file.endswith('.zip'):
                existing_files.add(file)
        for dir in dirs:
            if dir.endswith('.SAFE'):
                existing_files.add(f"{dir}.zip")
    
    # Filtra os arquivos que precisam ser baixados
    to_download = [img for img in needed_images if img not in existing_files]
    
    print(f"Total de arquivos necessários: {len(needed_images)}")
    print(f"Arquivos já existentes: {len(needed_images) - len(to_download)}")
    print(f"Arquivos a serem baixados: {len(to_download)}")
    
    if len(to_download) == 0:
        print("Todos os arquivos necessários já estão disponíveis!")
        return
    
    # Precisa buscar os IDs atuais dos produtos na API
    print("\nConsultando API para obter IDs dos produtos...")
    geometry_path = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/MG/PN_PANTANAL_MG.shp"
    wkt_geometry = file_to_wkt(geometry_path)
    
    # Consulta mais ampla para garantir que encontre tudo
    features = query_features(
        "Sentinel2", 
        {
            "startDate": "2024-01-01",
            "completionDate": date(2025, 12, 31),
            "processingLevel": "S2MSI2A",
            "geometry": wkt_geometry,
        },
    )
    
    print(f"Consulta retornou {len(features)} imagens totais.")
    
    # Mapeia nomes de arquivos para features
    features_by_name = {}
    for feature in features:
        title = feature["properties"]["title"]
        # Tenta várias variações do nome do arquivo
        features_by_name[title] = feature
        features_by_name[f"{title}.zip"] = feature
        features_by_name[f"{title}.SAFE"] = feature
        features_by_name[f"{title}.SAFE.zip"] = feature
    
    # Identifica as features para download
    features_to_download = []
    missing_features = []
    
    for img_name in to_download:
        # Tenta várias formas do nome
        base_name = img_name.replace('.zip', '').replace('.SAFE', '')
        
        found = False
        for name_variant in [img_name, base_name, f"{base_name}.SAFE", f"{base_name}.zip", f"{base_name}.SAFE.zip"]:
            if name_variant in features_by_name:
                features_to_download.append(features_by_name[name_variant])
                found = True
                break
        
        if not found:
            missing_features.append(img_name)
    
    if missing_features:
        print(f"\nAVISO: {len(missing_features)} imagens não foram encontradas na API:")
        for missing in missing_features[:10]:  # Mostra os 10 primeiros
            print(f"  - {missing}")
        if len(missing_features) > 10:
            print(f"  - ... e mais {len(missing_features)-10} imagens")
    
    # Confirma o download
    print(f"\nEncontradas {len(features_to_download)} de {len(to_download)} imagens para download.")
    
    if not features_to_download:
        print("Nenhuma imagem para baixar. Verifique se os nomes no JSON correspondem às imagens da API.")
        return
    
    response = input(f"Deseja iniciar o download de {len(features_to_download)} arquivos? (s/n): ")
    if response.lower() == 's':
        # CORREÇÃO AQUI: usar o generator corretamente
        download_count = 0
        print("\nIniciando downloads...")
        
        for download_id in download_features(
            features_to_download,
            download_path,
            {
                "concurrency": 4,
                "monitor": StatusMonitor(),
                "credentials": credentials,
            }
        ):
            download_count += 1
            print(f"✓ Download concluído: {download_id} ({download_count}/{len(features_to_download)})")

        print(f"\nDownload concluído para {download_count} arquivos.")
    else:
        print("Download cancelado.")

if __name__ == "__main__":
    print("=== Script de Download de Imagens Específicas ===")
    download_specific_files()
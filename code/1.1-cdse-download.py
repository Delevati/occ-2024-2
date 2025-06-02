import os
import certifi
import geopandas as gpd
import pandas as pd
from cdsetool.credentials import Credentials
from cdsetool.query import query_features
from cdsetool.download import download_features
from cdsetool.monitor import StatusMonitor
from datetime import date
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv('/Users/luryand/Documents/encode-image/coverage_otimization/code/.env')

# Definições de caminhos
geometry_path = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/MG/PN_PANTANAL_MG.shp"
download_path = "/Volumes/luryand/nova_busca"
relatorio_csv_path = "relatorio_downloads.csv" 

# Configurações SSL
os.environ['SSL_CERT_FILE'] = certifi.where()

# Credenciais
credentials = Credentials(os.getenv('CDSE_USER'), os.getenv('CDSE_PASSWORD'))

# Função para carregar arquivo de geometria e converter para WKT
def file_to_wkt(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.shp':
        gdf = gpd.read_file(file_path)
    elif ext == '.kml':
        gdf = gpd.read_file(file_path, driver='KML')
    else:
        raise ValueError(f"Formato de arquivo '{ext}' não suportado. Use KML ou SHP.")
    
    return gdf.geometry.iloc[0].wkt

# Função para obter os produtos já baixados
def get_downloaded_files(directory):
    downloaded = set()
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".SAFE") or (file.startswith("S2") and file.endswith(".zip")):
                filename = os.path.splitext(file)[0]
                downloaded.add(filename.upper())
        
        for dir_name in dirs:
            if dir_name.endswith(".SAFE"):
                dir_name_no_ext = os.path.splitext(dir_name)[0]
                downloaded.add(dir_name_no_ext.upper())
                
    return downloaded

# Consulta a API
print("Consultando produtos...")
wkt_geometry = file_to_wkt(geometry_path)

features = query_features(
    "Sentinel2", 
    {
        "startDate": "2024-08-13",
        "completionDate": date(2025, 4, 13),
        "processingLevel": "S2MSI2A",
        "geometry": wkt_geometry,
    },
)

print(f"Consulta retornou {len(features)} imagens totais.")

# Carrega arquivos já baixados
downloaded_files = get_downloaded_files(download_path)
print(f"Encontrados {len(downloaded_files)} arquivos já baixados.")

# Prepara relatório e lista para download
relatorio = []
features_to_download = []

for feature in features:
    product_title = feature["properties"]["title"].upper()
    product_id = feature["id"]
    
    status = "BAIXADO" if product_title in downloaded_files else "FALTA"
    relatorio.append({
        "id": product_id,
        "title": product_title,
        "status": status
    })
    
    if status == "FALTA":
        features_to_download.append(feature)

# Mostra resumo
print(f"Faltam baixar {len(features_to_download)} imagens.")

# Gera CSV do relatório
df_relatorio = pd.DataFrame(relatorio)
df_relatorio.to_csv(relatorio_csv_path, index=False)
print(f"Relatório salvo em {relatorio_csv_path}")

# Confirmação antes de baixar
if len(features_to_download) == 0:
    print("Todos os arquivos já foram baixados.")
else:
    response = input(f"Deseja iniciar o download dos {len(features_to_download)} arquivos restantes? (s/n): ")
    if response.lower() == 's':
        downloads = download_features(
            features_to_download,
            download_path,
            {
                "concurrency": 4,
                "monitor": StatusMonitor(),
                "credentials": credentials,
            }
        )

        for id in downloads:
            print(f"Feature {id} baixado com sucesso.")
    else:
        print("Download cancelado.")

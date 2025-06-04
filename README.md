# Otimização de Cobertura de Mosaicos de Imagens de Satélite

## Visão Geral

Este projeto implementa um pipeline para otimização da seleção de mosaicos de imagens de satélite Sentinel-2, visando maximizar a cobertura efetiva de uma Área de Interesse (AOI) com mínima presença de nuvens. O sistema combina processamento geométrico, algoritmos heurísticos e métodos exatos de otimização para resolver este problema.

## Estrutura do Projeto

```text

coverage_otimization/
├── code/                        # Scripts principais de processamento
│   ├── 1.1-cdse-download.py     # Download inicial de imagens Sentinel-2
│   ├── 1.1.2-cdse-recapture-images.py # Recuperação de imagens
│   ├── 2-prepare-and-greedy.py  # Implementação da heurística gulosa
│   ├── 2.2.1-PG-PC.py           # Cálculo de cobertura 2a2
│   ├── 3-CPLEX.py               # Modelo PLIM com IBM CPLEX
│   ├── APA-input/               # Diretório com shapefiles das áreas de interesse (utilizados no projeto .shp's)
│   │   ├── recapture/           # Subdiretório com áreas categorizadas
│   │   └── primarios/           # Shapefiles originais das AOIs
│   └── greedy_utils/            # Módulo de utilitários para o algoritmo
│       ├── __init__.py
│       ├── configuration.py     # Configurações globais e parâmetros
│       ├── file_utils.py        # Utilitários para manipulação de arquivos
│       ├── image_processing.py  # Funções de processamento de imagens
│       ├── metadata_utils.py    # Processamento de metadados
│       └── plotting_utils.py    # Funções de visualização
└── results/                     # Diretório de saída para resultados

````

## Inputs Necessários

### 1. Área de Interesse (AOI)

- __Formato__: Shapefile (.shp)

- __Projeção__: Recomendado usar projeções UTM específicas para a região

- __Localização__: ```code/APA-input/[região]/[nome_shapefile].shp```

### 2. Imagens Sentinel-2

- __Formato__: Arquivos .zip ou diretórios .SAFE

- __Localização__: Configurável em `ZIP_SOURCE_DIR`

- __Padrão de Nomes__: Formato padrão do Sentinel-2 (ex: `S2A_MSIL2A_YYYYMMDD...`)

## Componentes do Fluxo de Trabalho

### 1. Aquisição de Imagens

- __`1.1-cdse-download.py`__: Realiza busca e download inicial de imagens Sentinel-2

- __`1.1.2-cdse-recapture-images.py`__: Recupera imagens específicas da seleção inicial

### 2. Processamento e Análise

- __`2-prepare-and-greedy.py`__: Processa imagens e aplica algoritmo heurístico guloso para encontrar combinações iniciais de mosaicos

- __`2.2.1-PG-PC.py`__: Pós-processa grupos de mosaicos para calcular valores de cobertura par a par

### 3. Otimização

- __`3-CPLEX.py`__: Implementa modelo de Programação Linear Inteira Mista (PLIM) utilizando CPLEX para selecionar o conjunto ótimo de grupos de mosaicos

## Abordagem Metodológica

O sistema utiliza os valores de cobertura para calcular considerando sobreposições entre mosaicos. O modelo de otimização:

- Maximiza a cobertura efetiva
- Minimiza o número de grupos de mosaicos
- Minimiza a cobertura de nuvens
- Garante restrições de exclusividade de imagens
- Impõe requisitos mínimos de cobertura

### Métodos de Cálculo de Cobertura

__Cobertura estimada pelo MILP__: Cobertura = Soma(Áreas_Individuais) - Soma(Interseções_Pares)

### Normalização no Modelo de Otimização

Todos os valores de cobertura são normalizados para o intervalo [0,1] no modelo de otimização, representando a proporção (%) da AOI coberta.

## Personalização para Novas Áreas

Para aplicar o sistema a uma nova área de interesse:

1. __Preparar Shapefile__: Adicione o shapefile da nova AOI em `code/APA-input/[nova_regiao]/`

2. __Ajustar Configurações__:

   - Edite `code/greedy_utils/configuration.py` para apontar para o novo shapefile
   - Ajuste `OUTPUT_BASE_DIR` para um diretório adequado à nova região

3. __Parâmetros Específicos da Região__:

   - Para regiões com mais nuvens, considere ajustar `MAX_CLOUD_COVERAGE_THRESHOLD`
   - Em áreas menores, pode-se reduzir `MOSAIC_MIN_CONTRIBUTION_THRESHOLD`

4. __Limitações Computacionais__:

   - Para áreas muito grandes, considere subdividir em regiões menores
   - Ajuste `TEMP_EXTRACT_DIR` para um volume com espaço suficiente

## Requisitos de Sistema

- __Python 3.9+__
- __Espaço em Disco__: ~500GB para áreas grandes (imagens temporárias + resultados)
- __RAM__: Mínimo 16GB, recomendado 32GB+ para áreas extensas
- __IBM CPLEX Optimizer__: Versão 12.10+
- __Pacotes Python__:
  - docplex
  - geopandas
  - rasterio
  - shapely
  - pyproj
  - numpy
  - matplotlib
  - cdsetool (para download direto do Copernicus)
  
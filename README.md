# Otimização de Cobertura de Mosaicos de Imagens de Satélite

## Visão Geral

Este projeto implementa um pipeline para otimização da seleção de mosaicos de imagens de satélite Sentinel-2, visando maximizar a cobertura efetiva de uma Área de Interesse (AOI) com mínima presença de nuvens. O sistema combina processamento geométrico, algoritmos heurísticos e métodos exatos de otimização para resolver este problema.

## Estrutura do Projeto

```text

coverage_otimization/
├── code/                         # Scripts principais de processamento
│   ├── 1.1-cdse-download.py      # Download inicial de imagens Sentinel-2
│   ├── 1.2-cdse-recapture.py     # Recuperação de imagens
│   ├── 2-compatibility-greedy.py # Implementação da heurística gulosa
│   ├── 2.2-calc-area-2a2.py      # Cálculo de cobertura par a par (método 2a2)
│   ├── 3-CPLEX.py                # Modelo PLIM com IBM CPLEX
│   ├── APA-input/                # Diretório com shapefiles das áreas de interesse
│   │   ├── recapture/            # Subdiretório com áreas categorizadas
│   │   └── primarios/            # Shapefiles originais das AOIs
│   ├── greedy_utils/             # Módulo de utilitários para o algoritmo
│   └── external_utils/           # Utilitários externos / não usados em scripts principais
│       ├── __init__.py
│       ├── configuration.py      # Configurações globais e parâmetros
│       ├── file_utils.py         # Utilitários para manipulação de arquivos
│       ├── image_processing.py   # Funções de processamento de imagens
│       ├── json_utils.py         # Utilitários para serialização JSON
│       ├── metadata_utils.py     # Processamento de metadados
│       ├── plotting_utils.py     # Funções de visualização
│       └── processing_utils.py   # Funções de processamento principal
├── submission/                   # Dir para submissões
│   ├── artigo-sbpo/              # Arquivos .tex, .pdf e img/ 
│   └── resumo-sbpo/              # Arquivos .tex, .pdf e img/
└── results/                      # Diretório de saída para resultados

````

## Inputs Necessários

### 1. Área de Interesse (AOI)

- __Formato__: Shapefile (.shp)

- __Projeção__: Recomendado usar projeções UTM específicas para a região

- __Localização__: ```code/APA-input/recapture/[região]/[nome_shapefile].shp```

### 2. Imagens Sentinel-2

- __Formato__: Arquivos .zip ou diretórios .SAFE

- __Localização__: Configurável em `ZIP_SOURCE_DIR`

- __Padrão de Nomes__: Formato padrão do Sentinel-2 (ex: `S2A_MSIL2A_YYYYMMDD...`)

## Componentes do Fluxo de Trabalho

### 1. Aquisição de Imagens

- __`1.1-cdse-download.py`__: Realiza busca e download inicial de imagens Sentinel-2

- __`1.2-cdse-recapture.py`__: Recupera imagens específicas da seleção inicial

### 2. Processamento e Análise

- __`2-compatibility-greedy.py`__: Processa imagens e aplica algoritmo heurístico guloso para encontrar combinações iniciais de mosaicos

- __`2.2-calc-area-2a2.py.py`__: Pós-processa grupos de mosaicos para calcular valores de cobertura par a par

## Parâmetros de Configuração Detalhados

Os seguintes parâmetros podem ser ajustados no arquivo `code/greedy_utils/configuration.py`:

## Parâmetros do Algoritmo Guloso

- `MOSAIC_MIN_CONTRIBUTION_THRESHOLD`: Limiar de contribuição mínima (5%) para inclusão de imagem em grupo
- `CENTRAL_IMAGE_EFFECTIVE_COVERAGE_THRESHOLD`: Limiar para classificação de imagem como central (30%)
- `COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD`: Cobertura mínima para imagens complementares (2%)
- `MOSAIC_TIME_WINDOW_DAYS`: Janela temporal máxima para agrupar imagens (5 dias)
- `MAX_CLOUD_COVERAGE_THRESHOLD`: Cobertura máxima de nuvens permitida (50%)
- `OVERLAP_QUALITY_WEIGHT`: Peso para qualidade na avaliação de sobreposições (0.3)

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

## Parâmetros de Seleção Heurística Gulosa

Para aplicar o sistema a uma nova área de interesse:

1. __Preparar Shapefile__: Adicione o shapefile da nova AOI em `code/APA-input/recapture/[nova_regiao]/`

2. __Ajustar Configurações__:

   - Edite `code/greedy_utils/configuration.py` para apontar para o novo shapefile
   - Ajuste `OUTPUT_BASE_DIR` para um diretório adequado à nova região

3. __Parâmetros Específicos da Região__:

   - Para regiões com mais nuvens, considere ajustar `MAX_CLOUD_COVERAGE_THRESHOLD`
   - Em áreas menores, pode-se reduzir `MOSAIC_MIN_CONTRIBUTION_THRESHOLD`

4. __Limitações Computacionais__:

   - Para áreas muito grandes, considere subdividir em regiões menores
   - Ajuste `TEMP_EXTRACT_DIR` para um volume com espaço suficiente

## Parâmetros de Otimização Método Exato CPLEX

Para personalizar o comportamento do modelo de otimização CPLEX:

1. __Diretórios e Arquivos__:

   - Ajuste `METADATA_DIR` e `OUTPUT_DIR` para os diretórios de metadados e resultados
   - Configure os caminhos de `OPTIMIZATION_PARAMS_FILE` e `CPLEX_RESULTS_FILE`

2. __Pesos da Função Objetivo__:

   - `alpha = 0.4` - Penalidade por número de grupos (maior valor = menos grupos)
   - `gamma = 0.8` - Penalidade por cobertura de nuvens (maior valor = menos nuvens)

3. __Restrições do Modelo__:

   - `cloud_threshold = 0.40` - Limite máximo de cobertura de nuvens (40%)
   - `min_total_coverage = 0.85` - Cobertura mínima requerida da AOI (85%)

4. __Ajustes Recomendados por Cenário__:

   - Para regiões com muitas nuvens: Reduza `gamma` (0.4-0.6) e aumente `cloud_threshold` (0.60-0.70)
   - Para maximizar qualidade: Aumente `gamma` (0.9-1.0)
   - Para maximizar cobertura: Aumente `min_total_coverage` (0.90-0.95) e reduza `alpha` (0.2-0.3)
   - Para reduzir grupos de mosaicos: Aumente `alpha` (0.5-0.7)

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
  
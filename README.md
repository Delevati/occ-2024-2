
## Componentes do Fluxo de Trabalho

### 1. Aquisição de Imagens

-**`1.1-cdse-download-image-by-range.py`**: Realiza busca e download inicial de imagens Sentinel-2 por intervalo de datas

-**`1.2-cdse-recapture-img-not-downloaded.py`**: Recupera imagens específicas que não foram baixadas na etapa inicial

### 2. Processamento e Análise Heurística

-**`2-heuristica-gulosa.py`**: Processa imagens e aplica algoritmo heurístico guloso para encontrar combinações iniciais de mosaicos candidatos

### 3. Otimização Final

-**`3-cplex.py`**: Implementa modelo de Programação Linear Inteira Mista (PLIM) utilizando CPLEX para selecionar o conjunto ótimo de grupos de mosaicos

## Estrutura do Projeto

```text

coverage_otimization/
├── code/                                    # Scripts principais de processamento
│   ├── 1.1-cdse-download-image-by-range.py # Download inicial de imagens Sentinel-2
│   ├── 1.2-cdse-recapture-img-not-downloaded.py # Recuperação de imagens específicas
│   ├── 2-heuristica-gulosa.py              # Implementação da heurística gulosa
│   ├── 3-cplex.py                          # Modelo PLIM com IBM CPLEX
│   ├── .env                                # Variáveis de ambiente (credenciais CDSE)
│   ├── APA-input/                          # Diretório com shapefiles das áreas de interesse
│   │   ├── primarios/                      # Shapefiles originais das AOIs
│   │   │   ├── AL/                         # Área de Alagoas
│   │   │   ├── BA/                         # Área da Bahia
│   │   │   ├── MG/                         # Área de Minas Gerais
│   │   │   ├── MG-SP-RJ/                   # Área da Mantiqueira (MG-SP-RJ)
│   │   │   ├── PI-PE-CE/                   # Área do Piauí-Pernambuco-Ceará
│   │   │   └── RS/                         # Área do Rio Grande do Sul
│   │   └── recapture/                      # Áreas processadas com dados organizados
│   │       ├── AL/                         # Contém shapefiles, JSONs e resultados para AL
│   │       ├── BA/                         # Contém shapefiles, JSONs e resultados para BA
│   │       ├── MG/                         # Contém shapefiles, JSONs e resultados para MG
│   │       ├── MG-SP-RJ/                   # Contém shapefiles, JSONs e resultados para Mantiqueira
│   │       ├── PI-PE-CE/                   # Contém shapefiles, JSONs e resultados para PI-PE-CE
│   │       └── RS/                         # Contém shapefiles, JSONs e resultados para RS
│   ├── cplex_utils/                        # Módulo de utilitários para o algoritmo CPLEX
│   │   ├── __init__.py
│   │   ├── validation.py                   # Utilitário de validação de resultados do modelo
│   │   └── save_log.py                     # Utilitário para logs de mosaicos selecionados
│   ├── greedy_utils/                       # Módulo de utilitários para o algoritmo Guloso
│   │   ├── __init__.py
│   │   ├── configuration.py                # Configurações globais e parâmetros
│   │   ├── file_utils.py                   # Utilitários para manipulação de arquivos
│   │   ├── image_processing.py             # Funções de processamento de imagens
│   │   ├── json_utils.py                   # Utilitários para manipulação de JSON
│   │   ├── metadata_utils.py               # Funções para metadados de imagens
│   │   ├── plotting_utils.py               # Funções de plotagem e visualização
│   │   └── processing_utils.py             # Funções de processamento de imagens ZIP
│   ├── output_log_cplex/                   # Logs e resultados da otimização CPLEX
│   │   ├── selected_mosaics_log.txt        # Log detalhado dos mosaicos selecionados
│   │   └── *.json                          # Resultados por região (AL, BA, MG, etc.)
│   └── cplex_env/                          # Ambiente virtual Python para CPLEX
├── external_utils/                         # Scripts auxiliares e utilitários externos
├── results/                                # Resultados finais de processamento
├── submission/                             # Artigos e documentos de submissão
└── apresent/                              # Apresentações e slides

```

## Fluxo de Execução

### Pré-requisitos

1. Configure as credenciais do Copernicus Data Space Ecosystem (CDSE) no arquivo [`code/.env`](code/.env)
2. Certifique-se de ter os shapefiles das áreas de interesse em [`code/APA-input/primarios/`](code/APA-input/primarios/)

### Execução Sequencial

#### Etapa 1: Download de Imagens

```bash

cdcode/

python1.1-cdse-download-image-by-range.py

python1.2-cdse-recapture-img-not-downloaded.py# Se necessário recuperar imagens específicas

```

**Saídas da Etapa 1:**

- Imagens Sentinel-2 baixadas no diretório configurado
- Logs de download

#### Etapa 2: Processamento Heurístico Guloso

```bash

cdcode/

python2-heuristica-gulosa.py

```

**Pré-condições:**

- Imagens Sentinel-2 disponíveis da Etapa 1
- Shapefiles das AOIs em [`APA-input/primarios/`](code/APA-input/primarios/)

**Saídas da Etapa 2:**

- Arquivo `optimization_parameters.json` com grupos de mosaicos candidatos
- Metadados de imagens processadas
- Logs de processamento em [`greedy_utils/`](code/greedy_utils/)

#### Etapa 3: Otimização CPLEX

```bash

cdcode/

python3-cplex.py

```

**Pré-condições:**

- Arquivo `optimization_parameters.json` da Etapa 2
- IBM CPLEX instalado e configurado

**Saídas da Etapa 3:**

- Grupos de mosaicos otimizados selecionados
- Logs detalhados em [`output_log_cplex/`](code/output_log_cplex/)
- Resultados por região no formato JSON

## Configuração por Região

Para processar uma região específica, ajuste os caminhos nos scripts:

-**AL**: [`code/APA-input/recapture/AL/`](code/APA-input/recapture/AL/)

-**BA**: [`code/APA-input/recapture/BA/`](code/APA-input/recapture/BA/)

-**MG**: [`code/APA-input/recapture/MG/`](code/APA-input/recapture/MG/)

-**MG-SP-RJ**: [`code/APA-input/recapture/MG-SP-RJ/`](code/APA-input/recapture/MG-SP-RJ/)

-**PI-PE-CE**: [`code/APA-input/recapture/PI-PE-CE/`](code/APA-input/recapture/PI-PE-CE/)

-**RS**: [`code/APA-input/recapture/RS/`](code/APA-input/recapture/RS/)

Cada região deve conter:

- Shapefile da área de interesse
- Diretório `greedy/` para resultados da heurística
- Arquivos de configuração específicos da região

# Otimização de Cobertura de Mosaicos de Imagens de Satélite

## Visão Geral

Este projeto implementa um pipeline completo para otimização da seleção de mosaicos de imagens de satélite Sentinel-2, visando maximizar a cobertura efetiva de uma Área de Interesse (AOI) com mínima presença de nuvens. O sistema combina processamento geométrico, algoritmos heurísticos e métodos exatos de otimização para resolver este problema.

## Componentes do Fluxo de Trabalho

### 1. Aquisição de Imagens

- **`1.1-cdse-download.py`**: Realiza busca e download inicial de imagens Sentinel-2
- **`1.1.2-cdse-recapture-images.py`**: Recupera imagens específicas ausentes após análise inicial

### 2. Processamento e Análise

- **`2-prepare-and-greedy.py`**: Processa imagens e aplica algoritmo heurístico guloso para encontrar combinações iniciais de mosaicos
- **`2.2.1-PG-PC.py`**: Pós-processa grupos de mosaicos para calcular valores de cobertura par a par com precisão

### 3. Otimização

- **`3-CPLEX.py`**: Implementa modelo de Programação Linear Inteira Mista (PLIM) utilizando CPLEX para selecionar o conjunto ótimo de grupos de mosaicos

## Abordagem Metodológica

O sistema utiliza o Princípio da Inclusão-Exclusão (PIE) para calcular com precisão a cobertura considerando sobreposições entre imagens. O modelo de otimização:

- Maximiza a cobertura efetiva
- Minimiza o número de grupos de mosaicos
- Minimiza a cobertura de nuvens
- Garante restrições de exclusividade de imagens
- Impõe requisitos mínimos de cobertura

### Métodos de Cálculo de Cobertura

Duas abordagens distintas para o cálculo PIE são implementadas:

1. **PIE Tradicional** (no modelo CPLEX): obertura = Soma(Áreas_Individuais) - Soma(Interseções_Pares)

### Normalização no Modelo de Otimização

Todos os valores de cobertura são normalizados para o intervalo [0,1] no modelo de otimização, representando a proporção (%) da AOI coberta.

## Utilização

```bash
# 1. Download de imagens
python 1.1-cdse-download.py

# 2. Recaptura de imagens ausentes, se necessário
python 1.1.2-cdse-recapture-images.py

# 3. Processamento de imagens e aplicação de heurística gulosa
python 2-prepare-and-greedy.py

# 4. Cálculo preciso de valores de cobertura
python 2.2.1-PG-PC.py

# 5. Execução da otimização CPLEX
python 3-CPLEX.py 
```

## Requisitos

- Python 3.9+
- IBM CPLEX Optimizer
- Pacotes Python:
  - docplex
  - geopandas
  - rasterio
  - shapely
  - pyproj
  - cdsetool
  
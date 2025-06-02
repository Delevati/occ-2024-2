"""
Utilitários para operações com arquivos e diretórios.

Este módulo fornece funções para extração segura de arquivos de arquivos ZIP
e limpeza de diretórios temporários.
"""

import zipfile
import shutil
import logging
from pathlib import Path

def safe_extract(zip_ref, patterns, extract_path):
    """
    Extrai arquivos de um arquivo ZIP de forma segura, filtrando por padrões específicos.
    
    Parâmetros:
        zip_ref: Referência para o arquivo ZIP aberto
        patterns: Lista de padrões de nomes de arquivos a serem extraídos
        extract_path: Caminho para onde os arquivos serão extraídos
        
    Retorno:
        Dicionário com padrões como chaves e listas de caminhos extraídos como valores,
        ou None em caso de erro
    """
    extracted_files = {pattern: [] for pattern in patterns}
    try:
        for file_info in zip_ref.infolist():
            if file_info.is_dir():
                continue
                
            filename_part = Path(file_info.filename).name
            for pattern in patterns:
                if pattern in filename_part:
                    target_path = extract_path / filename_part
                    try:
                        with zip_ref.open(file_info) as source, open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                        extracted_files[pattern].append(str(target_path))
                        break
                    except Exception as e_extract_single:
                        logging.error(f"Erro ao extrair arquivo {file_info.filename} para {target_path}: {e_extract_single}")
                        if target_path.exists():
                            target_path.unlink()
                        break
    except zipfile.BadZipFile:
        logging.error(f"Arquivo ZIP corrompido: {zip_ref.filename}")
        return None
    except Exception as e:
        logging.error(f"Erro geral na extração de {zip_ref.filename}: {e}")
        return None
    
    return extracted_files

def remove_dir_contents(path: Path):
    """
    Remove recursivamente o conteúdo de um diretório.
    
    Parâmetros:
        path: Caminho do diretório a ser limpo
    """
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                logging.warning(f"Não foi possível remover {item}: {e}")
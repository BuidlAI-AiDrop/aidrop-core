"""
유틸리티 함수
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# 로깅 설정
def setup_logger(name: str = 'ai_deduction', level: int = logging.INFO) -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# 데이터 캐싱
def save_to_cache(data: Any, filename: str, cache_dir: str = 'cache') -> str:
    """데이터를 캐시 파일로 저장"""
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return file_path

def load_from_cache(filename: str, cache_dir: str = 'cache') -> Optional[Any]:
    """캐시 파일에서 데이터 로드"""
    file_path = os.path.join(cache_dir, filename)
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None

# 주소 유효성 검사
def is_valid_eth_address(address: str) -> bool:
    """이더리움 주소 유효성 검사"""
    if not address.startswith('0x'):
        return False
    if len(address) != 42:  # '0x' + 40 hex chars
        return False
    try:
        int(address, 16)
        return True
    except ValueError:
        return False

# 모델 버전 관리
def get_model_path(model_name: str, version: str = 'latest') -> str:
    """모델 파일 경로 반환"""
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    if version == 'latest':
        # 최신 버전 찾기
        versions = [d for d in os.listdir(models_dir) 
                   if os.path.isdir(os.path.join(models_dir, d)) and 
                   os.path.exists(os.path.join(models_dir, d, f"{model_name}.pkl"))]
        
        if not versions:
            raise FileNotFoundError(f"No versions found for model {model_name}")
        
        version = sorted(versions)[-1]
    
    model_path = os.path.join(models_dir, version, f"{model_name}.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return model_path 
"""
클러스터링 모듈용 유틸리티 함수
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# 로깅 설정
def setup_logger(name: str = 'ai_clusturing', level: int = logging.INFO) -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# 데이터 캐싱 및 로딩
def load_addresses_data(data_path: str) -> List[Dict]:
    """주소 데이터 로드"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    
    # JSON 또는 CSV 파일 지원
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {data_path}")

def save_results(results: Dict[str, Any], filename: str, output_dir: str = 'results'):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return file_path

def save_dataframe(df: pd.DataFrame, filename: str, output_dir: str = 'results'):
    """데이터프레임 저장"""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    
    if filename.endswith('.csv'):
        df.to_csv(file_path, index=True)
    elif filename.endswith('.json'):
        df.to_json(file_path, orient='records', indent=2)
    elif filename.endswith('.pkl'):
        df.to_pickle(file_path)
    else:
        # 기본 포맷은 CSV
        df.to_csv(f"{file_path}.csv", index=True)
        file_path = f"{file_path}.csv"
    
    return file_path

# 모델 유틸리티
def save_model(model: Any, model_name: str, version: str = 'v1', 
              metadata: Optional[Dict] = None, output_dir: str = 'models'):
    """모델 저장"""
    version_dir = os.path.join(output_dir, version)
    os.makedirs(version_dir, exist_ok=True)
    
    model_path = os.path.join(version_dir, f"{model_name}.pkl")
    
    model_data = {
        'model': model,
        'metadata': metadata or {}
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_path

def load_model(model_name: str, version: str = 'latest', output_dir: str = 'models') -> Optional[Dict]:
    """모델 로드"""
    if version == 'latest':
        # 최신 버전 찾기
        versions = [d for d in os.listdir(output_dir) 
                  if os.path.isdir(os.path.join(output_dir, d)) and 
                  os.path.exists(os.path.join(output_dir, d, f"{model_name}.pkl"))]
        
        if not versions:
            return None
        
        version = sorted(versions)[-1]
    
    model_path = os.path.join(output_dir, version, f"{model_name}.pkl")
    
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# 데이터 검증
def validate_features(features: Dict[str, float], required_features: List[str]) -> bool:
    """특성 벡터 검증"""
    # 필수 특성이 모두 있는지 확인
    missing_features = [f for f in required_features if f not in features]
    
    if missing_features:
        logging.warning(f"누락된 필수 특성: {missing_features}")
        return False
    
    return True

# 차원 축소 및 시각화 유틸리티
def reduce_dimensions(features_df: pd.DataFrame, n_components: int = 2, method: str = 'pca'):
    """차원 축소"""
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"지원하지 않는 차원 축소 방법: {method}")
    
    # 숫자형 특성만 선택
    numeric_features = features_df.select_dtypes(include=['float64', 'int64']).columns
    
    # 차원 축소 적용
    reduced_data = reducer.fit_transform(features_df[numeric_features])
    
    # 결과 데이터프레임 생성
    result_df = pd.DataFrame(
        reduced_data,
        index=features_df.index,
        columns=[f'{method}{i+1}' for i in range(n_components)]
    )
    
    return result_df, reducer 
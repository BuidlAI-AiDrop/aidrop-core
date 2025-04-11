"""
사용자 분류를 위한 AI 모델 모듈
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from utils import setup_logger

# 상대 경로 임포트를 절대 경로 임포트로 변경
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# get_model_path 함수 직접 정의
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

logger = setup_logger(__name__)

class UserClassificationModel:
    """사용자 분류를 위한 모델 클래스"""
    
    def __init__(self, model_name: str = 'user_classifier', version: str = 'latest'):
        """
        Args:
            model_name: 모델 이름
            version: 모델 버전 ('latest'는 최신 버전)
        """
        self.logger = logger
        self.model_name = model_name
        self.version = version
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoder = None
        self.clustering_model = None
    
    def train(self, X, y):
        """
        모델 훈련
        
        Args:
            X: 특성 행렬
            y: 타겟 레이블
        
        Returns:
            훈련된 모델 객체
        """
        self.logger.info(f"모델 훈련 시작: {self.model_name}")
        
        # 특성 이름 저장
        self.feature_names = X.columns.tolist()
        
        # 스케일러 적용
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 모델 초기화 및 훈련
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        self.logger.info(f"모델 훈련 완료: {self.model_name}")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        모델 평가
        
        Args:
            X_test: 테스트 특성 행렬
            y_test: 테스트 레이블
            
        Returns:
            정확도 (float)
        """
        if self.model is None:
            self.logger.error("모델이 훈련되지 않았습니다.")
            return 0.0
            
        # 테스트 데이터 스케일링
        X_test_scaled = self.scaler.transform(X_test)
        
        # 예측
        y_pred = self.model.predict(X_test_scaled)
        
        # 정확도 계산
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info(f"모델 평가 완료, 정확도: {accuracy:.4f}")
        return accuracy
    
    def get_feature_importance(self):
        """
        특성 중요도 반환
        
        Returns:
            특성 중요도 딕셔너리
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            self.logger.error("모델이 훈련되지 않았거나 특성 중요도를 제공하지 않습니다.")
            return {}
            
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            feature_importance[feature] = float(self.model.feature_importances_[i])
            
        return feature_importance
    
    def save(self, model_dir='./models'):
        """
        모델 저장
        
        Args:
            model_dir: 모델 저장 디렉토리
            
        Returns:
            저장된 모델 경로
        """
        if self.model is None:
            self.logger.error("저장할 모델이 없습니다.")
            return None
            
        # 버전 디렉토리 생성
        version = datetime.now().strftime('%Y%m%d')
        save_path = os.path.join(model_dir, version)
        os.makedirs(save_path, exist_ok=True)
        
        # 모델 데이터 구성
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'clustering_model': self.clustering_model,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'model_type': type(self.model).__name__,
                'feature_count': len(self.feature_names)
            }
        }
        
        # 모델 저장
        model_path = os.path.join(save_path, f"{self.model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        self.logger.info(f"모델 저장 완료: {model_path}")
        return model_path
    
    def load(self) -> bool:
        """저장된 모델 로드"""
        try:
            model_path = get_model_path(self.model_name, self.version)
            self.logger.info(f"모델 로딩: {model_path}")
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names')
            self.label_encoder = model_data.get('label_encoder')
            self.clustering_model = model_data.get('clustering_model')
            
            return True
        except Exception as e:
            self.logger.error(f"모델 로딩 실패: {str(e)}")
            return False
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        사용자 특성을 기반으로 분류 예측
        
        Args:
            features: 특성 벡터 (딕셔너리)
            
        Returns:
            예측 결과 (딕셔너리)
        """
        if self.model is None:
            loaded = self.load()
            if not loaded:
                return {'error': '모델 로딩 실패'}
        
        # 특성 벡터 준비
        feature_vector = self._prepare_feature_vector(features)
        
        # 분류 예측
        result = {}
        if self.model is not None:
            # 지도학습 모델 예측
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            
            # 클래스 레이블 디코딩
            if self.label_encoder is not None:
                class_name = self.label_encoder.inverse_transform([prediction])[0]
                class_names = self.label_encoder.classes_
                
                result['predicted_class'] = class_name
                result['prediction_confidence'] = float(probabilities[prediction])
                
                # 클래스별 확률
                class_probabilities = {}
                for i, class_name in enumerate(class_names):
                    class_probabilities[class_name] = float(probabilities[i])
                
                result['class_probabilities'] = class_probabilities
            else:
                result['predicted_class'] = str(prediction)
                result['prediction_confidence'] = float(max(probabilities))
        
        # 클러스터링 모델이 있으면 클러스터링 결과도 제공
        if self.clustering_model is not None:
            cluster = self.clustering_model.predict(feature_vector)[0]
            result['cluster'] = int(cluster)
            
            # 클러스터 중심까지의 거리 계산
            if hasattr(self.clustering_model, 'cluster_centers_'):
                distances = {}
                for i, center in enumerate(self.clustering_model.cluster_centers_):
                    dist = np.linalg.norm(feature_vector - center)
                    distances[f'cluster_{i}'] = float(dist)
                
                result['cluster_distances'] = distances
        
        # 중요 특성 추출
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                feature_importance[feature] = float(importances[i])
            
            # 상위 5개 중요 특성
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            result['top_features'] = dict(top_features)
        
        return result
    
    def _prepare_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """특성 벡터 준비"""
        # 모델이 학습된 특성 순서대로 벡터 생성
        feature_vector = np.zeros(len(self.feature_names))
        
        for i, feature_name in enumerate(self.feature_names):
            feature_vector[i] = features.get(feature_name, 0.0)
        
        # 스케일링 적용
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        else:
            feature_vector = feature_vector.reshape(1, -1)
        
        return feature_vector


class ClusteringModel:
    """비지도학습 클러스터링 모델 클래스"""
    
    def __init__(self, model_name: str = 'user_clustering', version: str = 'latest'):
        self.logger = logger
        self.model_name = model_name
        self.version = version
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def load(self) -> bool:
        """저장된 모델 로딩"""
        try:
            model_path = get_model_path(self.model_name, self.version)
            self.logger.info(f"클러스터링 모델 로딩: {model_path}")
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names')
            
            return True
        except Exception as e:
            self.logger.error(f"클러스터링 모델 로딩 실패: {str(e)}")
            return False
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        특성 벡터에 대해 클러스터 예측
        
        Args:
            features: 특성 벡터 (딕셔너리)
            
        Returns:
            클러스터링 결과 (딕셔너리)
        """
        if self.model is None:
            loaded = self.load()
            if not loaded:
                return {'error': '클러스터링 모델 로딩 실패'}
        
        # 특성 벡터 준비
        feature_vector = self._prepare_feature_vector(features)
        
        # 클러스터 예측
        cluster = self.model.predict(feature_vector)[0]
        
        result = {
            'cluster': int(cluster)
        }
        
        # 클러스터 중심까지의 거리
        if hasattr(self.model, 'cluster_centers_'):
            distances = {}
            for i, center in enumerate(self.model.cluster_centers_):
                dist = np.linalg.norm(feature_vector - center)
                distances[f'cluster_{i}'] = float(dist)
            
            result['cluster_distances'] = distances
            
            # 가장 가까운 클러스터
            closest_cluster = min(distances.items(), key=lambda x: x[1])[0]
            result['closest_cluster'] = closest_cluster
        
        return result
    
    def _prepare_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """특성 벡터 준비"""
        feature_vector = np.zeros(len(self.feature_names))
        
        for i, feature_name in enumerate(self.feature_names):
            feature_vector[i] = features.get(feature_name, 0.0)
        
        # 스케일링 적용
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        else:
            feature_vector = feature_vector.reshape(1, -1)
        
        return feature_vector 
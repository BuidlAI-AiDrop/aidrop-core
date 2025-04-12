"""
AI 파이프라인 메인 모듈
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime

from .utils import setup_logger

logger = setup_logger('pipeline')

class BlockchainDataCollector:
    """블록체인 데이터 수집기"""
    
    def __init__(self, api_key=None, chain='ethereum'):
        self.api_key = api_key
        self.chain = chain
        self.logger = setup_logger('data_collector')
    
    def collect_address_data(self, address):
        """주소 데이터 수집"""
        self.logger.info(f"수집 시작: {address}")
        # 실제 구현에서는 블록체인 API 호출
        # 목 데이터 반환
        return {
            "address": address,
            "transactions": [],
            "token_transfers": [],
            "internal_transactions": []
        }
    
    def collect_wallet_data(self, address):
        """지갑 데이터 수집 API"""
        try:
            data = self.collect_address_data(address)
            return {
                "status": "success",
                "data": data
            }
        except Exception as e:
            self.logger.error(f"데이터 수집 오류: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }


class InferenceService:
    """추론 서비스"""
    
    def __init__(self, model_path, scaler_path=None):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.logger = setup_logger('inference')
        
        # 모델 로드
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        try:
            self.model = joblib.load(self.model_path)
            if self.scaler_path and os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            else:
                self.scaler = None
                
            self.logger.info(f"모델 로드 완료: {self.model_path}")
        except Exception as e:
            self.logger.error(f"모델 로드 오류: {str(e)}")
            raise
    
    def predict(self, features):
        """예측 수행"""
        try:
            if self.scaler:
                features = self.scaler.transform(features)
            
            return self.model.predict(features)
        except Exception as e:
            self.logger.error(f"예측 오류: {str(e)}")
            raise


class AIPipeline:
    """AI 파이프라인 메인 클래스"""
    
    def __init__(self, cluster_model_dir, profile_dir, deduction_model_dir, data_dir, output_dir, version="v1.0"):
        self.cluster_model_dir = cluster_model_dir
        self.profile_dir = profile_dir
        self.deduction_model_dir = deduction_model_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.version = version
        
        # 디렉토리 생성
        os.makedirs(cluster_model_dir, exist_ok=True)
        os.makedirs(profile_dir, exist_ok=True)
        os.makedirs(deduction_model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger = setup_logger('ai_pipeline')
    
    def train(self, data_file, num_clusters=5, force_retrain=False):
        """모델 학습"""
        self.logger.info(f"훈련 시작: {data_file}, 클러스터 수: {num_clusters}")
        
        # 파일 경로 설정
        cluster_model_path = os.path.join(self.cluster_model_dir, f"cluster_models_{self.version}.pkl")
        profile_path = os.path.join(self.profile_dir, f"cluster_profiles_{self.version}.json")
        deduction_model_path = os.path.join(self.deduction_model_dir, f"deduction_model_{self.version}.pkl")
        
        # 이미 존재하는지 확인
        if not force_retrain and all(os.path.exists(p) for p in [cluster_model_path, profile_path, deduction_model_path]):
            self.logger.info("이미 훈련된 모델이 존재합니다. force_retrain=True로 설정하여 재훈련하세요.")
            return {
                "status": "skipped", 
                "message": "모델이 이미 존재합니다."
            }
        
        try:
            # 데이터 로드
            df = pd.read_csv(data_file)
            self.logger.info(f"데이터 로드 완료: {len(df)} 행")
            
            # 주소 열 제거 (특성에서 제외)
            addresses = df['address'].values
            features = df.drop('address', axis=1)
            
            # 학습 시뮬레이션
            # 1. 클러스터링
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(features)
            
            # 2. 클러스터 프로필 생성
            profiles = self._create_cluster_profiles(features, clusters, num_clusters)
            
            # 3. 추론 모델 (실제로는 Random Forest 등 사용 가능)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            # 임의의 라벨 생성 (목 데이터)
            mock_labels = np.random.randint(0, 2, size=len(features))
            rf.fit(features, mock_labels)
            
            # 모델 저장
            joblib.dump(kmeans, cluster_model_path)
            joblib.dump(rf, deduction_model_path)
            
            # 프로필 저장
            with open(profile_path, 'w') as f:
                json.dump(profiles, f, indent=2)
            
            self.logger.info(f"모델 학습 및 저장 완료: {self.version}")
            
            return {
                "status": "success",
                "version": self.version,
                "num_samples": len(df),
                "clustering": {
                    "num_clusters": num_clusters,
                    "model_path": cluster_model_path
                },
                "profiles": {
                    "path": profile_path
                },
                "deduction": {
                    "model_path": deduction_model_path
                }
            }
            
        except Exception as e:
            self.logger.error(f"훈련 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _create_cluster_profiles(self, features, clusters, num_clusters):
        """클러스터 프로필 생성"""
        profiles = {}
        
        for i in range(num_clusters):
            cluster_samples = features[clusters == i]
            
            if len(cluster_samples) == 0:
                continue
                
            # 각 특성의 평균 계산
            profile = {
                "cluster_id": i,
                "size": len(cluster_samples),
                "features": {}
            }
            
            # 각 특성의 평균/중앙값 계산
            for col in features.columns:
                profile["features"][col] = {
                    "mean": float(cluster_samples[col].mean()),
                    "median": float(cluster_samples[col].median())
                }
            
            # 가상의 클러스터 특성 추가
            profile["cluster_traits"] = [
                f"특성_{i}_1",
                f"특성_{i}_2",
                f"특성_{i}_3"
            ]
            
            profiles[str(i)] = profile
        
        return profiles 
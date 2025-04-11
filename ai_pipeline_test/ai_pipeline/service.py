"""
통합 분석 서비스
"""

import os
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from .pipeline import BlockchainDataCollector, InferenceService
from .utils import setup_logger

logger = setup_logger('service')

class DataStorage:
    """데이터 저장소"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.logger = setup_logger('storage')
    
    def save_data(self, address, data):
        """데이터 저장"""
        file_path = os.path.join(self.data_dir, f"{address.lower()}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return file_path
    
    def load_data(self, address):
        """데이터 로드"""
        file_path = os.path.join(self.data_dir, f"{address.lower()}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def save_processed_data(self, address, data):
        """처리된 데이터 저장"""
        file_path = os.path.join(self.data_dir, f"{address.lower()}_processed.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return file_path
    
    def load_processed_data(self, address):
        """처리된 데이터 로드"""
        file_path = os.path.join(self.data_dir, f"{address.lower()}_processed.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None


class FeatureExtractor:
    """특성 추출기"""
    
    def __init__(self):
        self.logger = setup_logger('extractor')
    
    def extract_features(self, data):
        """원시 블록체인 데이터에서 특성 추출"""
        address = data.get('address', '')
        txs = data.get('transactions', [])
        tokens = data.get('token_transfers', [])
        internal_txs = data.get('internal_transactions', [])
        
        # 기본 특성 추출
        features = {
            "address": address,
            "total_tx_count": len(txs),
            "token_transfer_count": len(tokens),
            "internal_tx_count": len(internal_txs),
            # 추가 특성 (실제로는 더 많은 특성 추출)
            "eth_balance_eth": 0.0,  # 실제 계산 필요
            "unique_tokens_count": len(set(t.get('tokenSymbol', '') for t in tokens)),
            "wallet_age_days": 0,  # 실제 계산 필요
            "total_eth_sent": 0.0,  # 실제 계산 필요
            "total_eth_received": 0.0,  # 실제 계산 필요
            "avg_tx_value": 0.0,  # 실제 계산 필요
            "max_tx_value": 0.0,  # 실제 계산 필요
            "min_tx_value": 0.0,  # 실제 계산 필요
            "tx_frequency_per_day": 0.0,  # 실제 계산 필요
            "unique_counterparties": 0,  # 실제 계산 필요
            "contract_interaction_count": 0,  # 실제 계산 필요
            "token_diversity_index": 0.0,  # 실제 계산 필요
            "defi_interaction": False,  # 실제 계산 필요
            "nft_interaction": False,  # 실제 계산 필요
            "exchange_interaction": False  # 실제 계산 필요
        }
        
        # 예시 데이터로 몇 가지 특성 계산
        if txs:
            # 총 ETH 전송량 계산
            eth_sent = sum(float(tx.get('value', '0')) / 1e18 for tx in txs if tx.get('from', '').lower() == address.lower())
            eth_received = sum(float(tx.get('value', '0')) / 1e18 for tx in txs if tx.get('to', '').lower() == address.lower())
            
            features['total_eth_sent'] = eth_sent
            features['total_eth_received'] = eth_received
            
            # 거래 상대방 수 계산
            counterparties = set()
            for tx in txs:
                if tx.get('from', '').lower() != address.lower():
                    counterparties.add(tx.get('from', '').lower())
                if tx.get('to', '').lower() != address.lower():
                    counterparties.add(tx.get('to', '').lower())
            
            features['unique_counterparties'] = len(counterparties)
            
            # 트랜잭션 값 통계
            tx_values = [float(tx.get('value', '0')) / 1e18 for tx in txs]
            if tx_values:
                features['avg_tx_value'] = sum(tx_values) / len(tx_values)
                features['max_tx_value'] = max(tx_values)
                features['min_tx_value'] = min([v for v in tx_values if v > 0] or [0])
        
        return features


class IntegratedAnalysisService:
    """통합 분석 서비스"""
    
    def __init__(self, cluster_model_dir, profile_dir, deduction_model_dir, output_dir, version="v1.0"):
        self.version = version
        self.cluster_model_dir = cluster_model_dir
        self.profile_dir = profile_dir
        self.deduction_model_dir = deduction_model_dir
        self.output_dir = output_dir
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 내부 컴포넌트 초기화
        self.collector = BlockchainDataCollector()
        self.extractor = FeatureExtractor()
        self.storage = DataStorage(os.path.join(output_dir, 'data'))
        self.logger = setup_logger('analysis_service')
        
        # 모델 파일 경로
        self.cluster_model_path = os.path.join(cluster_model_dir, f"cluster_models_{version}.pkl")
        self.profile_path = os.path.join(profile_dir, f"cluster_profiles_{version}.json")
        self.deduction_model_path = os.path.join(deduction_model_dir, f"deduction_model_{version}.pkl")
        
        # 모델 로드
        self._load_models()
    
    def _load_models(self):
        """모델 로드"""
        try:
            if os.path.exists(self.cluster_model_path):
                self.cluster_model = joblib.load(self.cluster_model_path)
                self.logger.info(f"클러스터 모델 로드 완료: {self.cluster_model_path}")
            else:
                self.cluster_model = None
                self.logger.warning(f"클러스터 모델을 찾을 수 없음: {self.cluster_model_path}")
            
            if os.path.exists(self.profile_path):
                with open(self.profile_path, 'r') as f:
                    self.profiles = json.load(f)
                self.logger.info(f"프로필 로드 완료: {self.profile_path}")
            else:
                self.profiles = {}
                self.logger.warning(f"프로필을 찾을 수 없음: {self.profile_path}")
            
            if os.path.exists(self.deduction_model_path):
                self.deduction_model = joblib.load(self.deduction_model_path)
                self.logger.info(f"추론 모델 로드 완료: {self.deduction_model_path}")
            else:
                self.deduction_model = None
                self.logger.warning(f"추론 모델을 찾을 수 없음: {self.deduction_model_path}")
        
        except Exception as e:
            self.logger.error(f"모델 로드 오류: {str(e)}")
            raise
    
    def analyze_address(self, address, force_refresh=False):
        """주소 분석"""
        start_time = time.time()
        self.logger.info(f"주소 분석 시작: {address}")
        
        # 1. 데이터 수집 또는 로드
        data = self.storage.load_data(address)
        if data is None or force_refresh:
            result = self.collector.collect_wallet_data(address)
            if result['status'] != 'success':
                return {
                    "status": "error",
                    "message": f"데이터 수집 실패: {result.get('message', '알 수 없는 오류')}"
                }
            data = result['data']
            self.storage.save_data(address, data)
        
        # 2. 특성 추출
        processed_data = self.storage.load_processed_data(address)
        if processed_data is None or force_refresh:
            features = self.extractor.extract_features(data)
            self.storage.save_processed_data(address, features)
        else:
            features = processed_data
        
        # 3. 분석 (클러스터링 & 분류)
        analysis_result = self._analyze_features(features)
        
        # 4. 결과 반환
        elapsed_time = time.time() - start_time
        
        return {
            "address": address,
            "timestamp": datetime.now().isoformat(),
            "version": self.version,
            "analysis_time": elapsed_time,
            "clustering": analysis_result.get("clustering", {}),
            "classification": analysis_result.get("classification", {}),
            "data_summary": {
                "transaction_count": features.get("total_tx_count", 0),
                "token_transfer_count": features.get("token_transfer_count", 0),
                "eth_sent": features.get("total_eth_sent", 0),
                "eth_received": features.get("total_eth_received", 0),
                "unique_counterparties": features.get("unique_counterparties", 0)
            }
        }
    
    def _analyze_features(self, features):
        """특성 분석 (클러스터링 & 분류)"""
        result = {}
        
        # 주소 필드 제외
        feature_vector = {k: v for k, v in features.items() if k != 'address'}
        
        # 벡터 변환
        feature_df = pd.DataFrame([feature_vector])
        
        try:
            # 클러스터링
            if self.cluster_model is not None:
                cluster_id = int(self.cluster_model.predict(feature_df)[0])
                
                # 클러스터 프로필 정보 추가
                cluster_profile = self.profiles.get(str(cluster_id), {})
                
                result["clustering"] = {
                    "cluster": cluster_id,
                    "cluster_size": cluster_profile.get("size", 0),
                    "primary_traits": cluster_profile.get("cluster_traits", [])[:3]
                }
            
            # 분류
            if self.deduction_model is not None:
                user_type = int(self.deduction_model.predict(feature_df)[0])
                confidence = max(self.deduction_model.predict_proba(feature_df)[0])
                
                result["classification"] = {
                    "user_type": "human" if user_type == 0 else "bot",
                    "confidence": float(confidence)
                }
        
        except Exception as e:
            self.logger.error(f"특성 분석 오류: {str(e)}")
            result["error"] = str(e)
        
        return result 
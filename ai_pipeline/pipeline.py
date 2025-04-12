#!/usr/bin/env python3
"""
AI 파이프라인 관리 모듈
클러스터링 및 분류 모델을 통합 관리
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 모듈 임포트
from ..ai_deduction import (
    FeatureExtractor, Model, InferenceService
)
from ..ai_clustering import clustering_utils, feature_engineering
from ..data_process.blockchain_data import BlockchainDataCollector
from ..data_process.data_processor import BlockchainDataProcessor
from ..data_process.data_storage import DataStorage

from .utils import setup_logger, validate_data, save_results

logger = setup_logger('pipeline')

class AIPipeline:
    """AI 파이프라인 클래스 - 클러스터링 및 분류 모델 통합 관리"""
    
    def __init__(self, 
                cluster_model_dir: str = 'models/clustering',
                profile_dir: str = 'models/profiles',
                deduction_model_dir: str = 'models/deduction',
                data_dir: str = 'data',
                output_dir: str = 'results',
                version: str = None,
                api_key: str = None):
        """
        초기화
        
        Args:
            cluster_model_dir: 클러스터링 모델 저장 디렉토리
            profile_dir: 클러스터 프로필 저장 디렉토리
            deduction_model_dir: 분류 모델 저장 디렉토리
            data_dir: 데이터 저장 디렉토리
            output_dir: 결과 저장 디렉토리
            version: 모델 버전 (기본값: 현재 날짜)
            api_key: Etherscan API 키
        """
        self.cluster_model_dir = cluster_model_dir
        self.profile_dir = profile_dir
        self.deduction_model_dir = deduction_model_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.version = version or datetime.now().strftime('%Y%m%d')
        self.api_key = api_key
        
        # 디렉토리 생성
        os.makedirs(cluster_model_dir, exist_ok=True)
        os.makedirs(profile_dir, exist_ok=True)
        os.makedirs(deduction_model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # 데이터 컬렉션 및 처리 클래스 초기화
        self.data_collector = BlockchainDataCollector(api_key=api_key) if api_key else None
        self.data_processor = BlockchainDataProcessor()
        self.data_storage = DataStorage(base_dir=data_dir)
        
        # 모델 관련 변수
        self.cluster_models = {}
        self.cluster_profiles = {}
        self.deduction_model = None
        self.feature_scaler = None
        
        logger.info(f"AI 파이프라인 초기화 완료 (버전: {self.version})")
    
    def train(self, data_file: str, num_clusters: int = 5, 
             force_retrain: bool = False) -> Dict[str, Any]:
        """
        클러스터링 및 분류 모델을 훈련
        
        Args:
            data_file: 훈련 데이터 파일 경로 (CSV 또는 JSON)
            num_clusters: 생성할 클러스터 수
            force_retrain: 기존 모델 덮어쓰기
            
        Returns:
            훈련 결과 정보
        """
        start_time = time.time()
        logger.info(f"파이프라인 훈련 시작: {data_file}")
        
        try:
            # 데이터 로드 및 검증
            df = self._load_data(data_file)
            if df is None or df.empty:
                return {'status': 'error', 'error': f"데이터 로드 실패: {data_file}"}
            
            # 특성 추출 및 전처리
            features_df, addresses = self._prepare_features(df)
            if features_df is None or features_df.empty:
                return {'status': 'error', 'error': "특성 추출 실패"}
            
            # 클러스터링 모델 훈련
            cluster_result = self._train_clustering_models(
                features_df, 
                num_clusters,
                force_retrain
            )
            
            if 'error' in cluster_result:
                return {'status': 'error', 'error': cluster_result['error']}
            
            # 클러스터 분석 및 프로필 생성
            profile_result = self._analyze_clusters(
                features_df, 
                cluster_result['labels'], 
                addresses
            )
            
            # 분류 모델 훈련 (지도 학습)
            classification_result = self._train_classification_model(
                features_df,
                cluster_result['labels'],
                force_retrain
            )
            
            # 결과 저장
            training_time = time.time() - start_time
            result = {
                'status': 'success',
                'message': f"파이프라인 훈련 완료 (소요 시간: {training_time:.2f}초)",
                'version': self.version,
                'num_samples': len(df),
                'clustering': {
                    'num_clusters': num_clusters,
                    'models': list(self.cluster_models.keys()),
                    'metrics': cluster_result.get('metrics', {})
                },
                'classification': classification_result,
                'cluster_profiles': profile_result,
                'training_time': training_time
            }
            
            # 결과 파일 저장
            results_file = os.path.join(
                self.output_dir, 
                f"pipeline_training_{self.version}.json"
            )
            save_results(result, results_file)
            
            return result
            
        except Exception as e:
            logger.error(f"훈련 중 오류 발생: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _load_data(self, data_file: str) -> pd.DataFrame:
        """
        데이터 파일 로드
        
        Args:
            data_file: 데이터 파일 경로 (CSV 또는 JSON)
        
        Returns:
            데이터프레임
        """
        try:
            if not os.path.exists(data_file):
                logger.error(f"파일을 찾을 수 없음: {data_file}")
                return None
            
            logger.info(f"데이터 로드 중: {data_file}")
            
            # 파일 형식에 따라 로드
            if data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
            elif data_file.endswith('.json'):
                df = pd.read_json(data_file)
            else:
                logger.error(f"지원되지 않는 파일 형식: {data_file}")
                return None
            
            # 데이터 검증
            if not validate_data(df):
                logger.error("데이터 검증 실패")
                return None
            
            logger.info(f"데이터 로드 완료: {len(df)}개 레코드")
            return df
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            return None
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        특성 추출 및 전처리
        
        Args:
            df: 원본 데이터프레임
        
        Returns:
            특성 데이터프레임, 주소 목록
        """
        try:
            logger.info("특성 추출 및 전처리 중...")
            
            # 주소가 있는지 확인
            if 'address' not in df.columns:
                logger.error("필수 열 'address'가 누락됨")
                return None, []
            
            addresses = df['address'].tolist()
            
            # data-process 모듈 사용
            processed_features = []
            
            # 주소별 특성 추출
            for idx, row in df.iterrows():
                address = row['address']
                
                # 기존 저장된 처리 데이터 확인
                wallet_data = self.data_storage.load_processed_data(address)
                
                if wallet_data is None and self.data_collector:
                    # 블록체인 데이터 수집
                    raw_data = self.data_collector.collect_address_data(address)
                    if raw_data:
                        # 데이터 처리 및 저장
                        wallet_data = self.data_processor.process_wallet_data(raw_data)
                        self.data_storage.save_processed_data(address, wallet_data)
                
                if wallet_data:
                    # 특성 추출
                    features = self.data_processor.extract_wallet_features(wallet_data)
                    
                    # 원본 데이터의 다른 특성과 병합
                    for col in df.columns:
                        if col != 'address' and col not in features:
                            features[col] = row[col]
                            
                    processed_features.append(features)
                else:
                    # 데이터 없는 경우 원본 행 사용
                    features = row.to_dict()
                    processed_features.append(features)
            
            # 특성 데이터프레임 생성
            features_df = pd.DataFrame(processed_features)
            
            # address 열 제거
            if 'address' in features_df.columns:
                features_df = features_df.drop('address', axis=1)
            
            # 누락된 값 처리
            features_df = features_df.fillna(0)
            
            # 범주형 특성 처리 (원-핫 인코딩)
            obj_cols = features_df.select_dtypes(include=['object']).columns
            features_df = pd.get_dummies(features_df, columns=obj_cols)
            
            logger.info(f"특성 추출 완료: {features_df.shape[1]}개 특성")
            return features_df, addresses
            
        except Exception as e:
            logger.error(f"특성 준비 중 오류 발생: {str(e)}")
            return None, []
    
    def _train_clustering_models(self, 
                               features_df: pd.DataFrame, 
                               num_clusters: int,
                               force_retrain: bool) -> Dict[str, Any]:
        """
        클러스터링 모델 훈련
        
        Args:
            features_df: 특성 데이터프레임
            num_clusters: 클러스터 수
            force_retrain: 기존 모델 덮어쓰기
            
        Returns:
            훈련 결과
        """
        try:
            logger.info(f"클러스터링 모델 훈련 중 (클러스터 수: {num_clusters})...")
            
            # 모델 파일 확인
            model_file = os.path.join(
                self.cluster_model_dir, 
                f"cluster_models_{self.version}.pkl"
            )
            
            if not force_retrain and os.path.exists(model_file):
                # 기존 모델 로드
                logger.info(f"기존 클러스터링 모델 로드: {model_file}")
                with open(model_file, 'rb') as f:
                    self.cluster_models = pickle.load(f)
                
                # 특성에 모델 적용하여 레이블 생성
                labels = self._apply_clustering_models(features_df)
                
                return {
                    'labels': labels,
                    'metrics': {'reused_existing_model': True}
                }
            
            # 특성 스케일링
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_df)
            self.feature_scaler = scaler
            
            # 차원 축소 (PCA)
            pca = PCA(n_components=min(50, features_df.shape[1]))
            reduced_features = pca.fit_transform(scaled_features)
            
            # KMeans 모델 훈련
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(reduced_features)
            
            # DBSCAN 모델 훈련
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(reduced_features)
            
            # 모델 저장
            self.cluster_models = {
                'kmeans': kmeans,
                'dbscan': dbscan,
                'pca': pca,
                'scaler': scaler
            }
            
            # 모델 파일 저장
            with open(model_file, 'wb') as f:
                pickle.dump(self.cluster_models, f)
            
            logger.info(f"클러스터링 모델 저장 완료: {model_file}")
            
            # KMeans 레이블 사용 (기본)
            return {
                'labels': kmeans_labels,
                'metrics': {
                    'kmeans_inertia': kmeans.inertia_,
                    'dbscan_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                }
            }
            
        except Exception as e:
            logger.error(f"클러스터링 모델 훈련 중 오류 발생: {str(e)}")
            return {'error': str(e)}
    
    def _apply_clustering_models(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        특성에 클러스터링 모델 적용
        
        Args:
            features_df: 특성 데이터프레임
            
        Returns:
            클러스터 레이블
        """
        # 모델 유무 확인
        if not self.cluster_models or 'kmeans' not in self.cluster_models:
            logger.error("클러스터링 모델이 없음")
            return np.zeros(len(features_df))
        
        try:
            # 특성 스케일링
            scaler = self.cluster_models['scaler']
            scaled_features = scaler.transform(features_df)
            
            # 차원 축소
            pca = self.cluster_models['pca']
            reduced_features = pca.transform(scaled_features)
            
            # KMeans 모델 적용
            kmeans = self.cluster_models['kmeans']
            labels = kmeans.predict(reduced_features)
            
            return labels
            
        except Exception as e:
            logger.error(f"클러스터링 모델 적용 중 오류 발생: {str(e)}")
            return np.zeros(len(features_df))
    
    def _analyze_clusters(self, 
                        features_df: pd.DataFrame, 
                        labels: np.ndarray,
                        addresses: List[str]) -> Dict[str, Any]:
        """
        클러스터 분석 및 프로필 생성
        
        Args:
            features_df: 특성 데이터프레임
            labels: 클러스터 레이블
            addresses: 주소 목록
            
        Returns:
            클러스터 프로필 정보
        """
        try:
            logger.info("클러스터 분석 및 프로필 생성 중...")
            
            # 레이블 열 추가
            df_with_labels = features_df.copy()
            df_with_labels['cluster'] = labels
            df_with_labels['address'] = addresses
            
            # 클러스터별 프로필 생성
            profiles = {}
            unique_clusters = np.unique(labels)
            
            for cluster in unique_clusters:
                cluster_data = df_with_labels[df_with_labels['cluster'] == cluster]
                cluster_size = len(cluster_data)
                
                # 클러스터 대표 특성 추출
                if cluster_size > 0:
                    # 수치형 특성의 평균
                    numeric_cols = cluster_data.select_dtypes(include=['number'])
                    numeric_cols = numeric_cols.drop('cluster', axis=1)
                    mean_values = numeric_cols.mean().to_dict()
                    
                    # 전체 데이터셋과 비교하여 차별화되는 특성 찾기
                    all_numeric = features_df.select_dtypes(include=['number'])
                    all_mean = all_numeric.mean()
                    
                    # 특성 중요도 계산 (평균과의 차이 기준)
                    feature_importance = {}
                    for col in numeric_cols.columns:
                        if col in all_mean:
                            # 전체 평균과의 차이 (표준화)
                            diff = (mean_values[col] - all_mean[col]) / all_mean[col] if all_mean[col] != 0 else 0
                            feature_importance[col] = abs(diff)
                    
                    # 상위 특성 선택
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    top_feature_names = [f[0] for f in top_features]
                    
                    # 클러스터 주소 목록
                    cluster_addresses = cluster_data['address'].tolist()
                    
                    # 프로필 저장
                    profiles[str(cluster)] = {
                        'size': cluster_size,
                        'mean_values': mean_values,
                        'top_traits': top_feature_names,
                        'trait_importance': {k: float(v) for k, v in feature_importance.items()},
                        'addresses': cluster_addresses[:100]  # 최대 100개만 저장
                    }
            
            # 프로필 저장
            profile_file = os.path.join(
                self.profile_dir, 
                f"cluster_profiles_{self.version}.json"
            )
            
            # 프로필 클래스 변수에 저장
            self.cluster_profiles = profiles
            
            # JSON 직렬화 가능한 형태로 변환
            serializable_profiles = {}
            for cluster, profile in profiles.items():
                serializable_profile = {
                    'size': profile['size'],
                    'top_traits': profile['top_traits'],
                    'trait_importance': profile['trait_importance'],
                    'addresses': profile['addresses']
                }
                serializable_profiles[cluster] = serializable_profile
            
            # 파일 저장
            with open(profile_file, 'w') as f:
                json.dump(serializable_profiles, f, indent=2)
            
            logger.info(f"클러스터 프로필 저장 완료: {profile_file}")
            
            return serializable_profiles
            
        except Exception as e:
            logger.error(f"클러스터 분석 중 오류 발생: {str(e)}")
            return {}
    
    def _train_classification_model(self, 
                                  features_df: pd.DataFrame,
                                  labels: np.ndarray,
                                  force_retrain: bool) -> Dict[str, Any]:
        """
        분류 모델 훈련
        
        Args:
            features_df: 특성 데이터프레임
            labels: 클러스터 레이블 (지도 학습용 타깃)
            force_retrain: 기존 모델 덮어쓰기
            
        Returns:
            훈련 결과
        """
        try:
            logger.info("분류 모델 훈련 중...")
            
            # 모델 파일 확인
            model_file = os.path.join(
                self.deduction_model_dir, 
                f"deduction_model_{self.version}.pkl"
            )
            
            if not force_retrain and os.path.exists(model_file):
                # 기존 모델 로드
                logger.info(f"기존 분류 모델 로드: {model_file}")
                with open(model_file, 'rb') as f:
                    self.deduction_model = pickle.load(f)
                
                return {
                    'status': 'success',
                    'reused_existing_model': True
                }
            
            # 훈련/테스트 데이터 분리
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, labels, test_size=0.2, random_state=42
            )
            
            # 모델 훈련
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 모델 평가
            accuracy = model.score(X_test, y_test)
            
            # 모델 저장
            self.deduction_model = model
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"분류 모델 저장 완료: {model_file} (정확도: {accuracy:.4f})")
            
            return {
                'status': 'success',
                'accuracy': accuracy,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"분류 모델 훈련 중 오류 발생: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def predict_cluster(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        클러스터 예측
        
        Args:
            features: 특성 데이터프레임
            
        Returns:
            클러스터 예측 결과
        """
        try:
            if not self.cluster_models or 'kmeans' not in self.cluster_models:
                logger.error("클러스터링 모델이 로드되지 않음")
                return {'error': "클러스터링 모델이 로드되지 않음"}
            
            # 클러스터링 모델 적용
            labels = self._apply_clustering_models(features)
            
            # 첫 번째 예측 결과 사용 (단일 데이터 예측 가정)
            if len(labels) > 0:
                cluster = int(labels[0])
                
                # 클러스터 프로필 정보 가져오기
                profile = self.cluster_profiles.get(str(cluster), {})
                
                return {
                    'cluster': cluster,
                    'primary_traits': profile.get('top_traits', [])[:5],
                    'cluster_size': profile.get('size', 0)
                }
            else:
                return {'error': "예측 결과 없음"}
            
        except Exception as e:
            logger.error(f"클러스터 예측 중 오류 발생: {str(e)}")
            return {'error': str(e)}
    
    def collect_data(self, addresses: List[str]) -> List[Dict[str, Any]]:
        """
        블록체인 데이터 수집
        
        Args:
            addresses: 주소 목록
            
        Returns:
            수집 결과 목록
        """
        if not self.data_collector:
            return [{'status': 'error', 'error': 'API 키가 설정되지 않음'}]
        
        results = []
        for address in addresses:
            try:
                # 블록체인 데이터 수집
                raw_data = self.data_collector.collect_address_data(address)
                
                if raw_data:
                    # 데이터 처리 및 저장
                    wallet_data = self.data_processor.process_wallet_data(raw_data)
                    self.data_storage.save_processed_data(address, wallet_data)
                    
                    results.append({
                        'status': 'success',
                        'address': address,
                        'transactions': len(raw_data.get('transactions', [])),
                        'tokens': len(raw_data.get('tokens', []))
                    })
                else:
                    results.append({
                        'status': 'error',
                        'address': address,
                        'error': '데이터 수집 실패'
                    })
            except Exception as e:
                results.append({
                    'status': 'error',
                    'address': address,
                    'error': str(e)
                })
        
        return results
    
    def export_features(self, addresses: List[str], output_file: str) -> Dict[str, Any]:
        """
        특성 데이터 내보내기
        
        Args:
            addresses: 주소 목록
            output_file: 출력 파일 경로
            
        Returns:
            내보내기 결과
        """
        try:
            all_features = []
            processed_count = 0
            
            for address in addresses:
                # 처리된 데이터 불러오기
                wallet_data = self.data_storage.load_processed_data(address)
                
                if wallet_data is None and self.data_collector:
                    # 데이터 수집
                    raw_data = self.data_collector.collect_address_data(address)
                    if raw_data:
                        wallet_data = self.data_processor.process_wallet_data(raw_data)
                        self.data_storage.save_processed_data(address, wallet_data)
                
                if wallet_data:
                    # 특성 추출
                    features = self.data_processor.extract_wallet_features(wallet_data)
                    features['address'] = address
                    all_features.append(features)
                    processed_count += 1
            
            if not all_features:
                return {
                    'status': 'error',
                    'error': '처리된 특성 없음'
                }
            
            # 데이터프레임 생성 및 저장
            features_df = pd.DataFrame(all_features)
            
            # 파일 형식에 따라 저장
            if output_file.endswith('.csv'):
                features_df.to_csv(output_file, index=False)
            elif output_file.endswith('.json'):
                features_df.to_json(output_file, orient='records')
            else:
                # 기본값으로 CSV 사용
                output_file = output_file + '.csv'
                features_df.to_csv(output_file, index=False)
            
            return {
                'status': 'success',
                'output_file': output_file,
                'processed_count': processed_count,
                'columns': features_df.columns.tolist()
            }
            
        except Exception as e:
            logger.error(f"특성 내보내기 중 오류 발생: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def load_models(self, version: str = None) -> bool:
        """
        저장된 모델 로드
        
        Args:
            version: 모델 버전 (기본값: self.version)
            
        Returns:
            로드 성공 여부
        """
        version = version or self.version
        
        try:
            # 클러스터링 모델 로드
            cluster_model_file = os.path.join(
                self.cluster_model_dir, 
                f"cluster_models_{version}.pkl"
            )
            
            if os.path.exists(cluster_model_file):
                with open(cluster_model_file, 'rb') as f:
                    self.cluster_models = pickle.load(f)
                logger.info(f"클러스터링 모델 로드 완료: {cluster_model_file}")
            else:
                logger.warning(f"클러스터링 모델 파일 없음: {cluster_model_file}")
                return False
            
            # 분류 모델 로드
            deduction_model_file = os.path.join(
                self.deduction_model_dir, 
                f"deduction_model_{version}.pkl"
            )
            
            if os.path.exists(deduction_model_file):
                with open(deduction_model_file, 'rb') as f:
                    self.deduction_model = pickle.load(f)
                logger.info(f"분류 모델 로드 완료: {deduction_model_file}")
            else:
                logger.warning(f"분류 모델 파일 없음: {deduction_model_file}")
            
            # 프로필 로드
            profile_file = os.path.join(
                self.profile_dir, 
                f"cluster_profiles_{version}.json"
            )
            
            if os.path.exists(profile_file):
                with open(profile_file, 'r') as f:
                    self.cluster_profiles = json.load(f)
                logger.info(f"클러스터 프로필 로드 완료: {profile_file}")
            else:
                logger.warning(f"클러스터 프로필 파일 없음: {profile_file}")
            
            # 특성 스케일러 확인
            if 'scaler' in self.cluster_models:
                self.feature_scaler = self.cluster_models['scaler']
            
            return True
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            return False 
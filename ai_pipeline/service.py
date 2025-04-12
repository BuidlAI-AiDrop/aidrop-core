"""
통합 AI 분석 서비스 - 클러스터링과 추론 시스템 통합
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

# ai_clusturing 모듈 임포트
from ai_clusturing.feature_extraction import FeatureExtractor as ClusterFeatureExtractor
from ai_clusturing.cluster_analyzer import ClusterFeatureAnalyzer
from ai_clusturing.clustering import ClusteringModel

# ai_deduction 모듈 임포트
from ai_deduction.feature_engineering import FeatureExtractor as DeductionFeatureExtractor
from ai_deduction.model import UserClassificationModel, ClusteringModel as DeductionClusteringModel
from ai_deduction.inference_service import InferenceService

# data-process 모듈 임포트
from data_process.blockchain_data import BlockchainDataCollector
from data_process.data_processor import BlockchainDataProcessor
from data_process.data_storage import DataStorage

# 로컬 모듈 임포트
from .utils import setup_logger, is_valid_eth_address, save_results, save_features

logger = setup_logger(__name__)

class IntegratedAnalysisService:
    """통합 분석 서비스 - 클러스터링과 추론 기능 통합"""
    
    def __init__(self, 
                 cluster_model_dir: str = 'models/clustering',
                 profile_dir: str = 'models/profiles',
                 deduction_model_dir: str = 'models/deduction',
                 cache_dir: str = 'cache',
                 output_dir: str = 'results',
                 version: str = 'v1',
                 api_key: Optional[str] = None):
        """
        Args:
            cluster_model_dir: 클러스터링 모델 디렉토리
            profile_dir: 클러스터 프로필 디렉토리
            deduction_model_dir: 분류 모델 디렉토리
            cache_dir: 캐시 디렉토리
            output_dir: 결과 저장 디렉토리
            version: 모델 버전
            api_key: Etherscan API 키 (선택사항)
        """
        self.logger = logging.getLogger(__name__)
        
        # 디렉토리 설정
        self.cluster_model_dir = cluster_model_dir
        self.profile_dir = profile_dir
        self.deduction_model_dir = deduction_model_dir
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.version = version
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        
        # 디렉토리 생성
        os.makedirs(self.cluster_model_dir, exist_ok=True)
        os.makedirs(self.profile_dir, exist_ok=True)
        os.makedirs(self.deduction_model_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 클러스터링 구성요소
        self.cluster_feature_extractor = ClusterFeatureExtractor()
        self.cluster_model = ClusteringModel(output_dir=cluster_model_dir)
        self.cluster_analyzer = ClusterFeatureAnalyzer(output_dir=profile_dir)
        
        # 분류 구성요소
        self.deduction_feature_extractor = DeductionFeatureExtractor()
        self.deduction_model = UserClassificationModel(model_name='user_classifier')
        self.inference_service = InferenceService(db_path=os.path.join(cache_dir, 'ai_results.db'))
        
        # data-process 구성요소
        self.data_collector = BlockchainDataCollector(
            api_key=self.api_key, 
            data_dir=os.path.join(cache_dir, 'raw')
        )
        self.data_processor = BlockchainDataProcessor(
            data_dir=os.path.join(cache_dir, 'raw'),
            processed_dir=os.path.join(cache_dir, 'processed')
        )
        self.data_storage = DataStorage(
            data_dir=cache_dir,
            db_path=os.path.join(cache_dir, 'data_storage.db')
        )
        
        # 모델 로드
        self.load_models()
        
        # 프로필 통합 모듈
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from profile_generator import AIProfileGenerator
            self.profile_generator = AIProfileGenerator(output_dir=os.path.join(output_dir, 'profiles'))
            self.logger.info("프로필 생성기 초기화 성공")
            self.profile_generation_enabled = True
        except Exception as e:
            self.logger.warning(f"프로필 생성기 초기화 실패: {e}")
            self.profile_generation_enabled = False
    
    def load_models(self) -> bool:
        """
        모델 로드
        
        Returns:
            로드 성공 여부
        """
        try:
            # 클러스터링 모델 로드
            kmeans_loaded = self.cluster_model.load_model('kmeans', self.version)
            profiles_loaded = self.cluster_analyzer.load_cluster_profiles(self.version)
            
            if not kmeans_loaded or not profiles_loaded:
                self.logger.warning("클러스터링 모델 또는 프로필 로드 실패 - 기본값 사용")
            else:
                self.logger.info("클러스터링 모델 및 프로필 로드 완료")
            
            # 분류 모델은 필요 시 자동 로드됨
            
            return True
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            return False
    
    def analyze_address(self, address: str, address_data: Optional[Dict] = None,
                       force_refresh: bool = False) -> Dict[str, Any]:
        """
        주소 통합 분석 (클러스터링 + 분류)
        
        Args:
            address: 분석할 이더리움 주소
            address_data: 주소 데이터 (없으면 자동으로 가져옴)
            force_refresh: 기존 결과가 있어도 강제로 새로 분석
            
        Returns:
            통합 분석 결과
        """
        start_time = time.time()
        
        # 주소 유효성 검사
        if not is_valid_eth_address(address):
            return {'error': '유효하지 않은 이더리움 주소'}
        
        self.logger.info(f"주소 분석 시작: {address}")
        
        try:
            # 0. 캐시된 분석 결과 확인 (force_refresh가 False인 경우)
            if not force_refresh:
                cached_result = self.data_storage.load_data(address, "vector")
                if cached_result and 'analysis_result' in cached_result:
                    self.logger.info(f"캐시된 분석 결과 사용: {address}")
                    return cached_result['analysis_result']
            
            # 1. 블록체인 데이터 가져오기
            if address_data is None:
                # data-process 모듈을 사용하여 블록체인 데이터 수집
                # 기존 처리된 데이터가 있는지 확인
                processed_data = self.data_storage.load_data(address, "processed")
                
                if processed_data and not force_refresh:
                    self.logger.info(f"기존 처리된 데이터 사용: {address}")
                    
                    # 필요한 데이터 구조 추출
                    basic_metrics = processed_data.get('basic_metrics', {})
                    temporal_metrics = processed_data.get('temporal_metrics', {})
                    token_metrics = processed_data.get('token_metrics', {})
                    network_metrics = processed_data.get('network_metrics', {})
                    
                    # 트랜잭션 데이터가 필요하면 raw 데이터에서 가져오기
                    raw_data = self.data_storage.load_data(address, "raw")
                    transactions = raw_data.get('transactions', [])
                    token_holdings = raw_data.get('token_transfers', [])
                    contract_interactions = raw_data.get('internal_transactions', [])
                else:
                    # 새로 데이터 수집 및 처리
                    self.logger.info(f"새로운 블록체인 데이터 수집 중: {address}")
                    collection_result = self.data_collector.collect_wallet_data(address)
                    
                    if collection_result.get('status') != 'success':
                        return {'error': '블록체인 데이터 수집 실패'}
                    
                    raw_data = collection_result.get('data', {})
                    
                    # 데이터 처리 및 특성 추출
                    self.logger.info(f"데이터 처리 및 특성 추출 중: {address}")
                    processed_data = self.data_processor.process_wallet_data(raw_data)
                    
                    if not processed_data:
                        return {'error': '데이터 처리 실패'}
                    
                    # 필요한 데이터 구조 추출
                    basic_metrics = processed_data.get('basic_metrics', {})
                    temporal_metrics = processed_data.get('temporal_metrics', {})
                    token_metrics = processed_data.get('token_metrics', {})
                    network_metrics = processed_data.get('network_metrics', {})
                    
                    # 트랜잭션 데이터 추출
                    transactions = raw_data.get('transactions', [])
                    token_holdings = raw_data.get('token_transfers', [])
                    contract_interactions = raw_data.get('internal_transactions', [])
                
                # 데이터 검사
                if not transactions:
                    return {'error': '트랜잭션 데이터 없음'}
                
                # ai_deduction 및 ai_clusturing 모듈에서 사용할 형식으로 데이터 구성
                address_data = {
                    'address': address,
                    'transactions': transactions,
                    'token_holdings': token_holdings,
                    'contract_interactions': contract_interactions
                }
            else:
                # 주어진 데이터 사용
                transactions = address_data.get('transactions', [])
                token_holdings = address_data.get('token_holdings', [])
                contract_interactions = address_data.get('contract_interactions', [])
                
                # data-process로 처리된 데이터가 없으면 생성
                processed_data = self.data_processor.process_wallet_data({
                    'address': address,
                    'transactions': transactions,
                    'token_transfers': token_holdings,
                    'internal_transactions': contract_interactions
                })
                
                # 필요한 데이터 구조 추출
                basic_metrics = processed_data.get('basic_metrics', {})
                temporal_metrics = processed_data.get('temporal_metrics', {})
                token_metrics = processed_data.get('token_metrics', {})
                network_metrics = processed_data.get('network_metrics', {})
            
            # 2. 클러스터링 특성 생성
            self.logger.info("클러스터링 특성 추출 중...")
            
            # data-process에서 추출한 특성 활용
            combined_features = {
                **basic_metrics,
                **temporal_metrics,
                **token_metrics,
                **network_metrics
            }
            
            # AI-클러스터링 특성 추출 (추가 특성이 필요한 경우)
            cluster_features = self.cluster_feature_extractor.extract_features(
                address, transactions, token_holdings, contract_interactions
            )
            
            # 두 특성 세트 병합 (data-process 특성 우선)
            for key, value in cluster_features.items():
                if key not in combined_features:
                    combined_features[key] = value
            
            # 특성 벡터를 DataFrame으로 변환 (클러스터링용)
            cluster_features_df = pd.DataFrame([combined_features])
            cluster_features_df.index = [address]
            
            # 디버깅용 특성 저장
            save_features(combined_features, address, os.path.join(self.output_dir, 'debug'))
            
            # 3. 클러스터 및 사용자 특성 예측
            self.logger.info("클러스터 및 사용자 특성 예측 중...")
            cluster_result = self._predict_cluster_and_traits(combined_features, cluster_features_df)
            
            # 4. 클러스터링 특성을 활용한 분류 특성 보강
            self.logger.info("분류 특성 보강 중...")
            enhanced_features = self._enhance_deduction_features(
                address, transactions, token_holdings, contract_interactions, 
                cluster_result.get('user_traits', {})
            )
            
            # 5. 추론 서비스로 분류 수행
            self.logger.info("추론 서비스 분류 수행 중...")
            # 기존 InferenceService 사용하되, 보강된 특성 정보 추가
            classification_result = self.inference_service.analyze_address(address, force_refresh)
            
            # 클러스터링 특성 정보 추가
            classification_result['user_traits'] = cluster_result.get('user_traits', {})
            classification_result['primary_traits'] = cluster_result.get('primary_traits', [])
            
            # 6. 결과 통합
            result = {
                'address': address,
                'clustering': cluster_result,
                'classification': classification_result,
                'analysis_time': time.time() - start_time,
                'data_summary': {
                    'transaction_count': basic_metrics.get('total_tx_count', 0),
                    'eth_balance': basic_metrics.get('eth_balance_eth', 0),
                    'unique_tokens': token_metrics.get('unique_tokens_count', 0),
                    'wallet_age_days': temporal_metrics.get('wallet_age_days', 0),
                    'total_tx_volume': network_metrics.get('total_eth_sent', 0) + network_metrics.get('total_eth_received', 0)
                }
            }
            
            # 7. 결과 캐싱
            # 벡터 저장소에 분석 결과 저장
            vector_data = {
                'address': address,
                'feature_vector': combined_features,
                'analysis_result': result,
                'timestamp': time.time()
            }
            self.data_storage.store_feature_vector(address, vector_data)
            
            # 8. 결과 저장
            result_path = save_results(result, address, self.output_dir)
            
            # 9. 프로필 이미지 생성 (활성화된 경우)
            if self.profile_generation_enabled:
                try:
                    self.logger.info(f"프로필 이미지 생성 중: {address}")
                    image_path = self.profile_generator.generate_profile(
                        address,
                        save=True,
                        show_prompt=False
                    )
                    
                    # 결과에 프로필 이미지 경로 추가
                    if image_path:
                        result['profile_image'] = image_path
                        self.logger.info(f"프로필 이미지 생성 완료: {image_path}")
                    
                    # 클러스터 및 특성 정보 출력
                    self.logger.info(f"[분석 결과] 주소: {address[:8]}...")
                    self.logger.info(f"클러스터: {result['clustering']['cluster']}")
                    self.logger.info(f"주요 특성: {', '.join(result['clustering']['primary_traits'])}")
                    
                    if 'classification' in result:
                        self.logger.info(f"분류: {result['classification'].get('user_type', 'unknown')}")
                        
                except Exception as e:
                    self.logger.warning(f"프로필 이미지 생성 실패: {e}")
            
            self.logger.info(f"통합 분석 완료: {address}, 소요 시간: {result['analysis_time']:.2f}초")
            
            return result
            
        except Exception as e:
            self.logger.error(f"주소 분석 중 오류 발생: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'error': f'분석 중 오류 발생: {str(e)}'}
    
    def _predict_cluster_and_traits(self, features: Dict[str, float], 
                                   features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        클러스터 및 사용자 특성 예측
        
        Args:
            features: 특성 딕셔너리
            features_df: 특성 DataFrame
            
        Returns:
            클러스터 및 사용자 특성 정보
        """
        # 클러스터 예측
        try:
            cluster = int(self.cluster_model.predict(features_df, 'kmeans')[0])
        except Exception as e:
            self.logger.warning(f"클러스터 예측 실패, 기본값 사용: {str(e)}")
            cluster = 0
        
        # 사용자 특성 추출
        try:
            user_traits = self.cluster_analyzer.get_user_traits(features)
            primary_traits = self.cluster_analyzer.get_primary_traits(user_traits)
        except Exception as e:
            self.logger.warning(f"사용자 특성 추출 실패, 기본값 사용: {str(e)}")
            user_traits = {}
            primary_traits = []
        
        # 결과 생성
        result = {
            'cluster': cluster,
            'user_traits': user_traits,
            'primary_traits': primary_traits,
            'cluster_profile': self.cluster_analyzer.cluster_profiles.get(f'cluster_{cluster}', {})
        }
        
        return result
    
    def _enhance_deduction_features(self, address: str, transactions: List[Dict],
                                   token_holdings: List[Dict], contract_interactions: List[Dict],
                                   user_traits: Dict[str, float]) -> Dict[str, float]:
        """
        클러스터링 특성을 활용한 분류 특성 보강
        
        Args:
            address: 주소
            transactions: 트랜잭션 목록
            token_holdings: 토큰 보유 목록
            contract_interactions: 컨트랙트 상호작용 목록
            user_traits: 사용자 특성 점수
            
        Returns:
            보강된 특성 벡터
        """
        # 기본 분류 특성 추출
        deduction_features = self.deduction_feature_extractor.extract_features(
            address, transactions, token_holdings, contract_interactions
        )
        
        # 사용자 특성으로 보강
        for trait, score in user_traits.items():
            deduction_features[f'trait_{trait}'] = score
        
        return deduction_features 
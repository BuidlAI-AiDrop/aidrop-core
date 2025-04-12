"""
사용자 분류 추론 서비스 API
"""

import time
import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from .utils import setup_logger, is_valid_eth_address
from .feature_engineering import FeatureExtractor
from .model import UserClassificationModel, ClusteringModel

logger = setup_logger(__name__)

class InferenceService:
    """사용자 분류 추론 서비스"""
    
    def __init__(self, db_path: str = 'ai_results.db'):
        """
        Args:
            db_path: 결과 저장 데이터베이스 경로
        """
        self.logger = logger
        self.db_path = db_path
        self.feature_extractor = FeatureExtractor()
        self.classifier = UserClassificationModel()
        self.clustering = ClusteringModel()
        
        # 결과 저장용 DB 초기화
        self._init_db()
    
    def _init_db(self):
        """결과 저장용 데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 분류 결과 테이블
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS classification_results (
                address TEXT PRIMARY KEY,
                result_json TEXT,
                timestamp INTEGER,
                model_version TEXT
            )
            ''')
            
            # 클러스터링 결과 테이블
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS clustering_results (
                address TEXT PRIMARY KEY,
                result_json TEXT,
                timestamp INTEGER,
                model_version TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"데이터베이스 초기화 완료: {self.db_path}")
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {str(e)}")
    
    def get_data_from_blockchain(self, address: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        블록체인에서 주소 데이터 가져오기
        
        참고: 실제 구현에서는 데이터 수집 모듈과 연동해야 함
        여기서는 이미 데이터가 수집되어 있다고 가정
        
        Args:
            address: 지갑 주소
            
        Returns:
            트랜잭션, 토큰 보유, 컨트랙트 상호작용 데이터
        """
        from .utils import load_from_cache
        
        # 캐시에서 데이터 로드 (실제로는 데이터 수집 모듈에서 가져와야 함)
        transactions = load_from_cache(f"{address}_transactions.json") or []
        token_holdings = load_from_cache(f"{address}_tokens.json") or []
        contract_interactions = load_from_cache(f"{address}_contracts.json") or []
        
        self.logger.info(f"주소 데이터 로드: {address}, 트랜잭션 {len(transactions)}개")
        
        return transactions, token_holdings, contract_interactions
    
    def analyze_address(self, address: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        지갑 주소 분석
        
        Args:
            address: 분석할 지갑 주소
            force_refresh: 기존 결과가 있어도 강제로 새로 분석
            
        Returns:
            분석 결과 (딕셔너리)
        """
        # 주소 유효성 검사
        if not is_valid_eth_address(address):
            return {'error': '유효하지 않은 이더리움 주소'}
        
        # 기존 결과 확인 (캐시 활용)
        if not force_refresh:
            existing_result = self.get_cached_result(address)
            if existing_result:
                self.logger.info(f"캐시된 분석 결과 사용: {address}")
                return existing_result
        
        start_time = time.time()
        
        # 1. 블록체인 데이터 가져오기
        transactions, token_holdings, contract_interactions = self.get_data_from_blockchain(address)
        
        # 데이터가 없으면 오류 반환
        if not transactions:
            return {'error': '트랜잭션 데이터 없음'}
        
        # 2. 특성 추출
        features = self.feature_extractor.extract_features(
            address, transactions, token_holdings, contract_interactions
        )
        
        # 3. 모델 예측
        # 3.1 지도학습 분류
        classification_result = self.classifier.predict(features)
        
        # 3.2 비지도학습 클러스터링
        clustering_result = self.clustering.predict(features)
        
        # 4. 결과 저장
        result = {
            'address': address,
            'classification': classification_result,
            'clustering': clustering_result,
            'features': features,  # 디버깅용, 실제 API에서는 제외 가능
            'analysis_time': time.time() - start_time
        }
        
        # 결과 DB에 저장
        self.save_result(address, result)
        
        self.logger.info(f"주소 분석 완료: {address}, 소요 시간: {result['analysis_time']:.2f}초")
        
        return result
    
    def get_cached_result(self, address: str) -> Optional[Dict[str, Any]]:
        """캐시된 분석 결과 가져오기"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 분류 결과 조회
            cursor.execute(
                "SELECT result_json, timestamp FROM classification_results WHERE address = ?", 
                (address,)
            )
            classification_row = cursor.fetchone()
            
            # 클러스터링 결과 조회
            cursor.execute(
                "SELECT result_json FROM clustering_results WHERE address = ?", 
                (address,)
            )
            clustering_row = cursor.fetchone()
            
            conn.close()
            
            if classification_row:
                # 결과가 너무 오래된 경우 새로 분석 (1일 기준)
                result_timestamp = classification_row[1]
                if time.time() - result_timestamp > 86400:
                    return None
                
                # 저장된 결과 반환
                classification_result = json.loads(classification_row[0])
                
                result = {
                    'address': address,
                    'classification': classification_result,
                }
                
                if clustering_row:
                    result['clustering'] = json.loads(clustering_row[0])
                
                return result
            
        except Exception as e:
            self.logger.error(f"캐시된 결과 조회 실패: {str(e)}")
        
        return None
    
    def save_result(self, address: str, result: Dict[str, Any]):
        """분석 결과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = int(time.time())
            
            # 분류 결과 저장
            cursor.execute(
                """
                INSERT OR REPLACE INTO classification_results 
                (address, result_json, timestamp, model_version) 
                VALUES (?, ?, ?, ?)
                """,
                (
                    address,
                    json.dumps(result['classification']),
                    current_time,
                    self.classifier.version
                )
            )
            
            # 클러스터링 결과 저장
            cursor.execute(
                """
                INSERT OR REPLACE INTO clustering_results 
                (address, result_json, timestamp, model_version) 
                VALUES (?, ?, ?, ?)
                """,
                (
                    address,
                    json.dumps(result['clustering']),
                    current_time,
                    self.clustering.version
                )
            )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"분석 결과 저장 완료: {address}")
        except Exception as e:
            self.logger.error(f"분석 결과 저장 실패: {str(e)}")
    
    def get_category_distribution(self) -> Dict[str, int]:
        """분류된 사용자 카테고리 분포 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT result_json FROM classification_results")
            rows = cursor.fetchall()
            
            conn.close()
            
            # 카테고리별 카운트
            category_counts = {}
            for row in rows:
                result = json.loads(row[0])
                category = result.get('predicted_class')
                if category:
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            return category_counts
        except Exception as e:
            self.logger.error(f"카테고리 분포 조회 실패: {str(e)}")
            return {}
    
    def get_cluster_distribution(self) -> Dict[str, int]:
        """클러스터 분포 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT result_json FROM clustering_results")
            rows = cursor.fetchall()
            
            conn.close()
            
            # 클러스터별 카운트
            cluster_counts = {}
            for row in rows:
                result = json.loads(row[0])
                cluster = result.get('cluster')
                if cluster is not None:
                    cluster_key = f'cluster_{cluster}'
                    cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1
            
            return cluster_counts
        except Exception as e:
            self.logger.error(f"클러스터 분포 조회 실패: {str(e)}")
            return {} 
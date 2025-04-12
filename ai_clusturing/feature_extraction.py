"""
블록체인 데이터에서 클러스터링을 위한 특성을 추출하는 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
import sys
import time
from datetime import datetime

# 상대 경로 임포트를 절대 경로 임포트로 변경
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import setup_logger

logger = setup_logger(__name__)

class FeatureExtractor:
    """클러스터링을 위한 특성 추출 클래스"""
    
    def __init__(self):
        self.logger = logger
    
    def extract_features_batch(self, addresses_data: List[Dict]) -> pd.DataFrame:
        """
        여러 주소의 데이터에서 특성 추출
        
        Args:
            addresses_data: 각 주소별 데이터 목록
            
        Returns:
            특성 DataFrame (주소별 특성 벡터)
        """
        features_list = []
        
        for addr_data in addresses_data:
            address = addr_data.get('address')
            transactions = addr_data.get('transactions', [])
            token_holdings = addr_data.get('token_holdings', [])
            contract_interactions = addr_data.get('contract_interactions', [])
            
            features = self.extract_features(
                address, transactions, token_holdings, contract_interactions
            )
            features['address'] = address
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        features_df.set_index('address', inplace=True)
        
        self.logger.info(f"총 {len(features_df)} 개 주소의 특성 추출 완료")
        return features_df
    
    def extract_features(self, address: str, transactions: List[Dict], 
                         token_holdings: List[Dict], 
                         contract_interactions: List[Dict]) -> Dict[str, float]:
        """
        단일 주소의 데이터에서 특성 추출
        
        Args:
            address: 지갑 주소
            transactions: 트랜잭션 목록
            token_holdings: 토큰 보유 목록
            contract_interactions: 컨트랙트 상호작용 목록
            
        Returns:
            특성 벡터 (딕셔너리)
        """
        features = {}
        
        # 1. 기본 활동 지표
        features['total_txn_count'] = len(transactions)
        
        if transactions:
            # 트랜잭션 타임스탬프 추출 및 정렬
            timestamps = sorted([tx.get('timestamp', 0) for tx in transactions])
            
            if len(timestamps) > 1:
                # 활동 기간
                features['account_age_days'] = (timestamps[-1] - timestamps[0]) / 86400
                
                # 트랜잭션 빈도 (일평균)
                time_span = (timestamps[-1] - timestamps[0]) / 86400  # 초를 일로 변환
                if time_span > 0:
                    features['txn_per_day'] = len(transactions) / time_span
                else:
                    features['txn_per_day'] = float(len(transactions))
                
                # 트랜잭션 간격
                intervals = np.diff(timestamps)
                features['avg_interval'] = float(intervals.mean()) if len(intervals) > 0 else 0
                features['interval_std'] = float(intervals.std()) if len(intervals) > 1 else 0
                
                # 최근성 지표
                current_time = int(time.time())
                features['days_since_last_activity'] = (current_time - timestamps[-1]) / 86400
            
            # 트랜잭션 값 분석
            values = [float(tx.get('value', 0)) for tx in transactions]
            if values:
                features['avg_txn_value'] = np.mean(values)
                features['median_txn_value'] = np.median(values)
                features['max_txn_value'] = max(values)
                features['total_value'] = sum(values)
                
                # 값의 변동성
                if len(values) > 1:
                    features['value_std'] = np.std(values)
                    features['value_range'] = max(values) - min(values)
        
        # 2. 토큰 관련 지표
        if token_holdings:
            features['token_count'] = len(token_holdings)
            
            # ERC-20, NFT 구분
            erc20_tokens = [t for t in token_holdings if t.get('type') == 'ERC20']
            nfts = [t for t in token_holdings if t.get('type') in ['ERC721', 'ERC1155']]
            
            features['erc20_count'] = len(erc20_tokens)
            features['nft_count'] = len(nfts)
            
            # 토큰 가치 (USD 기준)
            token_values = [float(t.get('value_usd', 0)) for t in token_holdings]
            if token_values:
                features['total_token_value_usd'] = sum(token_values)
                features['avg_token_value_usd'] = np.mean(token_values)
            
            # NFT 비율
            if features['token_count'] > 0:
                features['nft_ratio'] = features['nft_count'] / features['token_count']
            else:
                features['nft_ratio'] = 0.0
        else:
            features['token_count'] = 0
            features['erc20_count'] = 0
            features['nft_count'] = 0
            features['nft_ratio'] = 0.0
        
        # 3. 네트워크 관련 지표
        if transactions:
            # 고유 상대방 수
            counterparties = set()
            for tx in transactions:
                if tx.get('from') and tx.get('from').lower() != address.lower():
                    counterparties.add(tx.get('from').lower())
                if tx.get('to') and tx.get('to').lower() != address.lower():
                    counterparties.add(tx.get('to').lower())
            
            features['unique_counterparties'] = len(counterparties)
            
            # 상대방 다양성 (트랜잭션 수 대비)
            if len(transactions) > 0:
                features['counterparty_diversity'] = features['unique_counterparties'] / len(transactions)
            else:
                features['counterparty_diversity'] = 0.0
        
        # 4. 컨트랙트 상호작용 지표
        if contract_interactions:
            # 고유 컨트랙트 수
            contracts = set([c.get('address') for c in contract_interactions if c.get('address')])
            features['unique_contracts'] = len(contracts)
            
            # 직접 호출과 토큰 관련 구분
            token_interactions = [c for c in contract_interactions 
                                if any(method in c.get('method', '') 
                                      for method in ['transfer', 'approve', 'mint'])]
            
            features['token_interaction_count'] = len(token_interactions)
            
            # DeFi 관련 지표
            defi_methods = ['swap', 'addLiquidity', 'stake', 'deposit', 'borrow', 'lend', 'farm']
            defi_interactions = [c for c in contract_interactions 
                               if any(method in c.get('method', '') for method in defi_methods)]
            
            features['defi_interaction_count'] = len(defi_interactions)
            
            # NFT 마켓플레이스 지표
            nft_methods = ['mint', 'createSale', 'bid', 'list']
            nft_interactions = [c for c in contract_interactions 
                              if any(method in c.get('method', '') for method in nft_methods)]
            
            features['nft_marketplace_count'] = len(nft_interactions)
            
            # 상호작용 다양성 (컨트랙트 수 / 상호작용 수)
            if len(contract_interactions) > 0:
                features['contract_diversity'] = features['unique_contracts'] / len(contract_interactions)
            else:
                features['contract_diversity'] = 0.0
        
        # 5. 활동 유형 지표
        # 출금/입금 비율
        if transactions:
            outgoing = len([tx for tx in transactions if tx.get('from', '').lower() == address.lower()])
            incoming = len([tx for tx in transactions if tx.get('to', '').lower() == address.lower()])
            
            features['outgoing_count'] = outgoing
            features['incoming_count'] = incoming
            
            if incoming + outgoing > 0:
                features['outgoing_ratio'] = outgoing / (incoming + outgoing)
            else:
                features['outgoing_ratio'] = 0.0
        
        # 6. 가스 사용 지표
        if transactions:
            gas_used = [float(tx.get('gasUsed', 0)) for tx in transactions]
            gas_price = [float(tx.get('gasPrice', 0)) for tx in transactions]
            
            if gas_used:
                features['avg_gas_used'] = np.mean(gas_used)
                features['total_gas_used'] = sum(gas_used)
            
            if gas_price:
                features['avg_gas_price'] = np.mean(gas_price)
                features['max_gas_price'] = max(gas_price) if gas_price else 0
                
            # 가스 효율성 (값 대비 가스 비용)
            if values and sum(values) > 0 and gas_used and gas_price:
                gas_fees = [g_used * g_price for g_used, g_price in zip(gas_used, gas_price)]
                features['gas_to_value_ratio'] = sum(gas_fees) / sum(values)
            else:
                features['gas_to_value_ratio'] = 0.0
        
        return features
    
    def preprocess_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        특성 전처리 (정규화, 스케일링, 결측치 처리 등)
        
        Args:
            features_df: 원본 특성 DataFrame
            
        Returns:
            전처리된 특성 DataFrame
        """
        # 결측치 처리
        features_df = features_df.fillna(0)
        
        # 로그 스케일링 (큰 값을 가진 특성)
        log_scale_features = [
            'total_txn_count', 'txn_per_day', 'avg_txn_value', 'max_txn_value',
            'total_value', 'token_count', 'total_token_value_usd'
        ]
        
        for feature in log_scale_features:
            if feature in features_df.columns:
                # 0 또는 음수 값 처리
                features_df[f'{feature}_log'] = np.log1p(features_df[feature].clip(lower=0.0001))
        
        # 표준화 (StandardScaler) - 클러스터링 전 필요
        from sklearn.preprocessing import StandardScaler
        
        # 모든 숫자형 특성 선택
        numeric_features = features_df.select_dtypes(include=['float64', 'int64']).columns
        
        # 스케일러 적용
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df[numeric_features])
        
        # 스케일링된 데이터프레임 생성
        scaled_df = pd.DataFrame(
            scaled_features, 
            index=features_df.index, 
            columns=[f'{col}_scaled' for col in numeric_features]
        )
        
        # 원본 특성과 스케일링된 특성 합치기
        result_df = pd.concat([features_df, scaled_df], axis=1)
        
        self.logger.info(f"특성 전처리 완료: {len(result_df.columns)} 개 특성")
        # 스케일러는 내부 속성으로 저장하고 DataFrame만 반환
        self._scaler = scaler
        return result_df 
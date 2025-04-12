"""
블록체인 데이터에서 특성을 추출하는 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import sys
import os

# 상대 경로 임포트를 절대 경로 임포트로 변경
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import setup_logger

logger = setup_logger(__name__)

class FeatureExtractor:
    """블록체인 데이터에서 사용자 특성을 추출하는 클래스"""
    
    def __init__(self):
        self.logger = logger
        
    def extract_features(self, address: str, transactions: List[Dict], 
                         token_holdings: List[Dict], 
                         contract_interactions: List[Dict]) -> Dict[str, float]:
        """
        지갑 주소의 트랜잭션 및 토큰 데이터에서 특성 추출
        
        Args:
            address: 지갑 주소
            transactions: 트랜잭션 목록
            token_holdings: 토큰 보유 목록
            contract_interactions: 컨트랙트 상호작용 목록
            
        Returns:
            특성 벡터 (딕셔너리)
        """
        self.logger.info(f"특성 추출 시작: {address}")
        
        features = {}
        
        # 1. 활동 지표
        features['total_txn_count'] = len(transactions)
        
        if transactions:
            # 트랜잭션 타임스탬프 추출 및 정렬
            timestamps = sorted([tx.get('timestamp', 0) for tx in transactions])
            
            if len(timestamps) > 1:
                # 트랜잭션 빈도 (일평균)
                time_span = (timestamps[-1] - timestamps[0]) / 86400  # 초를 일로 변환
                if time_span > 0:
                    features['txn_per_day'] = len(transactions) / time_span
                else:
                    features['txn_per_day'] = float(len(transactions))
                
                # 트랜잭션 간격
                intervals = np.diff(timestamps)
                features['min_interval'] = float(intervals.min()) if len(intervals) > 0 else 0
                features['avg_interval'] = float(intervals.mean()) if len(intervals) > 0 else 0
                features['max_interval'] = float(intervals.max()) if len(intervals) > 0 else 0
        
        # 2. 트랜잭션 특성
        if transactions:
            values = [float(tx.get('value', 0)) for tx in transactions]
            features['avg_txn_value'] = np.mean(values) if values else 0
            features['median_txn_value'] = np.median(values) if values else 0
            features['total_value'] = np.sum(values) if values else 0
            
            gas_fees = [float(tx.get('gasPrice', 0)) * float(tx.get('gasUsed', 0)) for tx in transactions]
            features['avg_gas_fee'] = np.mean(gas_fees) if gas_fees else 0
            features['total_gas_fee'] = np.sum(gas_fees) if gas_fees else 0
        
        # 3. 상대방 분석
        if transactions:
            counterparties = set()
            for tx in transactions:
                if tx.get('from') and tx.get('from').lower() != address.lower():
                    counterparties.add(tx.get('from').lower())
                if tx.get('to') and tx.get('to').lower() != address.lower():
                    counterparties.add(tx.get('to').lower())
            
            features['unique_counterparties'] = len(counterparties)
        
        # 4. 컨트랙트 상호작용
        if contract_interactions:
            contract_addresses = set([c.get('address') for c in contract_interactions if c.get('address')])
            features['unique_contracts'] = len(contract_addresses)
            
            # 주요 프로토콜 상호작용 확인 (예: Uniswap, OpenSea 등)
            protocol_flags = self._check_protocol_interactions(contract_interactions)
            features.update(protocol_flags)
        
        # 5. 토큰 보유량
        if token_holdings:
            features['token_count'] = len(token_holdings)
            
            # ERC-20 토큰과 NFT 구분
            erc20_tokens = [t for t in token_holdings if t.get('type') == 'ERC20']
            nfts = [t for t in token_holdings if t.get('type') == 'ERC721' or t.get('type') == 'ERC1155']
            
            features['erc20_count'] = len(erc20_tokens)
            features['nft_count'] = len(nfts)
        
        # 6. 계정 나이 및 수명
        if transactions and timestamps:
            features['account_age_days'] = (timestamps[-1] - timestamps[0]) / 86400
            
            # 최근 활동 (마지막 트랜잭션으로부터 지난 일수)
            import time
            current_time = int(time.time())
            features['days_since_last_activity'] = (current_time - timestamps[-1]) / 86400
        
        self.logger.info(f"특성 추출 완료: {len(features)} 개 특성")
        return features
    
    def _check_protocol_interactions(self, contract_interactions: List[Dict]) -> Dict[str, int]:
        """주요 프로토콜과의 상호작용 여부 확인"""
        # 주요 프로토콜 컨트랙트 주소 (실제로는 DB나 설정에서 가져올 것)
        protocols = {
            'uniswap': ['0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984'.lower()],  # Uniswap 예시 주소
            'opensea': ['0x7Be8076f4EA4A4AD08075C2508e481d6C946D12b'.lower()],  # OpenSea 예시 주소
            'aave': ['0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9'.lower()],     # Aave 예시 주소
        }
        
        results = {}
        contract_addresses = [c.get('address', '').lower() for c in contract_interactions]
        
        for protocol_name, addresses in protocols.items():
            # 해당 프로토콜 컨트랙트와의 상호작용 여부
            interaction = any(addr in contract_addresses for addr in addresses)
            results[f'used_{protocol_name}'] = 1 if interaction else 0
        
        return results
    
    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """특성 정규화"""
        # 여기서는 간단한 MinMax 스케일링을 적용할 수 있음
        # 실제로는 학습 시 계산된 스케일링 파라미터를 사용해야 함
        
        # 예시: txn_per_day를 로그 스케일링
        if 'txn_per_day' in features and features['txn_per_day'] > 0:
            features['txn_per_day_log'] = np.log1p(features['txn_per_day'])
        
        return features 
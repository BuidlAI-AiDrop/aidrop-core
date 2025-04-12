#!/usr/bin/env python3
"""
블록체인 데이터 처리 모듈
원시 블록체인 데이터를 분석에 필요한 형태로 변환
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

class BlockchainDataProcessor:
    """블록체인 데이터를 처리하는 클래스"""
    
    def __init__(self, data_dir="./cache/raw", processed_dir="./cache/processed"):
        """
        초기화 함수
        
        Args:
            data_dir: 원시 데이터 디렉토리
            processed_dir: 처리된 데이터 저장 디렉토리
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
    
    def process_address_data(self, address_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        주소 데이터 처리
        
        Args:
            address_data: 원시 주소 데이터
            
        Returns:
            처리된 특성 데이터
        """
        address = address_data.get("address", "")
        chain_id = address_data.get("chain_id", "1")
        
        # 트랜잭션 데이터 변환
        transactions = address_data.get("transactions", [])
        token_transfers = address_data.get("token_transfers", [])
        contract_interactions = address_data.get("contract_interactions", [])
        
        # 특성 추출
        features = self._extract_features(
            address, 
            transactions, 
            token_transfers, 
            contract_interactions
        )
        
        # 결과 저장
        result = {
            "address": address,
            "chain_id": chain_id,
            "features": features,
            "processed_at": datetime.now().isoformat()
        }
        
        # 처리된 데이터 저장
        output_file = os.path.join(self.processed_dir, f"{address.lower()}_{chain_id}_processed.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def _extract_features(
        self, 
        address: str, 
        transactions: List[Dict[str, Any]], 
        token_transfers: List[Dict[str, Any]], 
        contract_interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        블록체인 데이터에서 특성 추출
        
        Args:
            address: 블록체인 주소
            transactions: 트랜잭션 데이터
            token_transfers: 토큰 전송 데이터
            contract_interactions: 컨트랙트 상호작용 데이터
            
        Returns:
            추출된 특성 데이터
        """
        # 1. 기본 활동 지표
        tx_count = len(transactions)
        token_tx_count = len(token_transfers)
        contract_int_count = len(contract_interactions)
        
        # 빈 데이터 처리
        if tx_count == 0:
            return {
                "tx_count": 0,
                "avg_tx_value": 0,
                "unique_counterparties": 0,
                "defi_ratio": 0,
                "nft_ratio": 0,
                "gaming_ratio": 0,
                "social_ratio": 0,
                "tx_frequency": 0,
                "holding_period": 0,
                "risk_score": 0,
                "is_active": False
            }
        
        # 2. 트랜잭션 특성
        tx_values = [float(tx.get("value", 0)) for tx in transactions]
        avg_tx_value = sum(tx_values) / max(1, len(tx_values))
        
        # 3. 고유 상대방 수
        counterparties = set()
        for tx in transactions:
            sender = tx.get("sender", "")
            receiver = tx.get("receiver", "")
            if sender and sender.lower() != address.lower():
                counterparties.add(sender.lower())
            if receiver and receiver.lower() != address.lower():
                counterparties.add(receiver.lower())
        
        unique_counterparties = len(counterparties)
        
        # 4. 컨트랙트 상호작용 분석
        # 컨트랙트 유형 카운트
        contract_types = {
            "defi": 0,
            "nft": 0,
            "gaming": 0,
            "social": 0,
            "other": 0
        }
        
        for interaction in contract_interactions:
            contract_type = interaction.get("contract_type", "other").lower()
            if contract_type in contract_types:
                contract_types[contract_type] += 1
            else:
                contract_types["other"] += 1
        
        total_interactions = max(1, sum(contract_types.values()))
        
        # 각 유형 비율 계산
        defi_ratio = contract_types.get("defi", 0) / total_interactions
        nft_ratio = contract_types.get("nft", 0) / total_interactions
        gaming_ratio = contract_types.get("gaming", 0) / total_interactions
        social_ratio = contract_types.get("social", 0) / total_interactions
        
        # 5. 활동 빈도 및 패턴
        # 트랜잭션 타임스탬프 정렬
        timestamps = []
        for tx in transactions:
            tx_time = tx.get("timestamp")
            if tx_time:
                try:
                    if isinstance(tx_time, str):
                        dt = datetime.fromisoformat(tx_time.replace('Z', '+00:00'))
                        timestamps.append(dt.timestamp())
                    else:
                        timestamps.append(float(tx_time))
                except:
                    pass
        
        timestamps.sort()
        
        # 트랜잭션 빈도 (하루당 트랜잭션 수)
        tx_frequency = 0
        holding_period = 0
        
        if len(timestamps) >= 2:
            first_tx = timestamps[0]
            last_tx = timestamps[-1]
            time_diff_days = (last_tx - first_tx) / (60 * 60 * 24)  # 초를 일로 변환
            
            if time_diff_days > 0:
                tx_frequency = len(timestamps) / time_diff_days
                holding_period = time_diff_days
        
        # 6. 리스크 점수 (간단한 휴리스틱)
        risk_score = 0
        
        # 높은 빈도의 트랜잭션은 더 공격적인 경향
        if tx_frequency > 5:  # 하루에 5회 이상 트랜잭션
            risk_score += 0.4
        elif tx_frequency > 1:  # 하루에 1회 이상 트랜잭션
            risk_score += 0.2
        
        # DeFi 비율이 높으면 리스크 증가
        risk_score += defi_ratio * 0.3
        
        # 최근 활동 여부 (마지막 트랜잭션이 30일 이내)
        is_active = False
        if timestamps:
            last_tx_time = timestamps[-1]
            current_time = datetime.now().timestamp()
            days_since_last_tx = (current_time - last_tx_time) / (60 * 60 * 24)
            is_active = days_since_last_tx <= 30
        
        # 결과 반환
        return {
            "tx_count": tx_count,
            "avg_tx_value": avg_tx_value,
            "unique_counterparties": unique_counterparties,
            "defi_ratio": defi_ratio,
            "nft_ratio": nft_ratio,
            "gaming_ratio": gaming_ratio,
            "social_ratio": social_ratio,
            "tx_frequency": tx_frequency,
            "holding_period": holding_period,
            "risk_score": risk_score,
            "is_active": is_active
        } 
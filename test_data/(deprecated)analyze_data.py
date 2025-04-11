#!/usr/bin/env python3
import json
import os
import collections
from typing import Dict, List, Any

def analyze_blockchain_data(filename: str = "blockchain_test_data.json"):
    """블록체인 테스트 데이터 분석"""
    
    # 파일 로드
    print(f"파일 '{filename}' 로딩 중...")
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"총 {len(data)} 개의 주소 데이터 로드됨")
    
    # 이벤트 통계
    total_events = sum(len(events) for events in data.values())
    print(f"총 {total_events} 개의 이벤트 분석 중...")
    
    # 주소별 이벤트 수 분석
    event_counts = [len(events) for events in data.values()]
    avg_events = sum(event_counts) / len(event_counts)
    print(f"주소당 평균 이벤트 수: {avg_events:.2f}")
    print(f"주소당 최소 이벤트 수: {min(event_counts)}")
    print(f"주소당 최대 이벤트 수: {max(event_counts)}")
    
    # 계약 타입별 이벤트 분석
    contract_type_counts = collections.Counter()
    event_type_counts = collections.Counter()
    
    for address, events in data.items():
        for event in events:
            contract_type = event.get('type', 'unknown')
            event_name = event.get('event', 'unknown')
            contract_type_counts[contract_type] += 1
            event_type_counts[event_name] += 1
    
    print("\n--- 계약 타입별 이벤트 분포 ---")
    for contract_type, count in contract_type_counts.most_common():
        percentage = count / total_events * 100
        print(f"{contract_type}: {count} ({percentage:.2f}%)")
    
    print("\n--- 상위 10개 이벤트 타입 ---")
    for event_type, count in event_type_counts.most_common(10):
        percentage = count / total_events * 100
        print(f"{event_type}: {count} ({percentage:.2f}%)")
    
    # 사용자 유형 추론 분석
    print("\n--- 사용자 유형 추론 분석 ---")
    user_type_inferences = classify_users(data)
    
    # 유형별 주소 수 집계
    inferred_types = collections.Counter()
    for address, inferred_type in user_type_inferences.items():
        inferred_types[inferred_type] += 1
    
    for user_type, count in inferred_types.most_common():
        percentage = count / len(data) * 100
        print(f"{user_type}: {count} 주소 ({percentage:.2f}%)")

def classify_users(data: Dict[str, List[Dict]]) -> Dict[str, str]:
    """사용자 유형 추론"""
    user_types = {}
    
    for address, events in data.items():
        # 1. DeFi vs NFT (D-N)
        contract_types = [event.get('type') for event in events]
        type_counter = collections.Counter(contract_types)
        
        defi_ratio = type_counter.get('defi', 0) / len(events) if events else 0
        nft_ratio = type_counter.get('nft', 0) / len(events) if events else 0
        
        finance_vs_creation = 'D' if defi_ratio > nft_ratio else 'N'
        
        # 2. 트레이딩 vs 홀딩 (T-H)
        # 이벤트 타임스탬프 간격 분석
        if len(events) >= 2:
            timestamps = sorted([event.get('time') for event in events])
            # 간단한 근사: 첫 이벤트와 마지막 이벤트 간의 기간이 길고 이벤트 수가 적으면 홀더
            timespan = (pd.to_datetime(timestamps[-1]) - pd.to_datetime(timestamps[0])).days
            if timespan > 0:
                event_frequency = len(events) / timespan  # 일당 이벤트 수
                trading_vs_holding = 'T' if event_frequency > 0.3 else 'H'  # 임의의 임계값
            else:
                trading_vs_holding = 'T'  # 짧은 기간에 여러 이벤트 = 트레이더
        else:
            trading_vs_holding = 'H'  # 단일 이벤트만 있으면 홀더로 간주
        
        # 3. 위험 vs 안정성 (A-S)
        # 다양한 컨트랙트와 상호작용하는 계정은 위험 감수 성향 있음
        contracts = set([event.get('contract') for event in events])
        contract_diversity = len(contracts) / len(events) if events else 0
        risk_vs_security = 'A' if contract_diversity > 0.5 else 'S'  # 임의의 임계값
        
        # 4. 커뮤니티 vs 독립성 (C-I)
        # DAO 활동 및 다양한 컨트랙트와 상호작용하는 계정은 커뮤니티 중심
        dao_ratio = type_counter.get('dao', 0) / len(events) if events else 0
        social_vs_solo = 'C' if dao_ratio > 0.1 or contract_diversity > 0.7 else 'I'
        
        # 최종 유형 결정
        user_type = f"{finance_vs_creation}-{trading_vs_holding}-{risk_vs_security}-{social_vs_solo}"
        user_types[address] = user_type
    
    return user_types

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("pandas 패키지가 설치되어 있지 않습니다.")
        print("필요한 경우: pip install pandas")
        exit(1)
    
    analyze_blockchain_data() 
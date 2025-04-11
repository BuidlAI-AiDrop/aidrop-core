#!/usr/bin/env python3
"""
간단한 블록체인 데이터 처리 테스트
"""

import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def get_mock_data(address):
    """모의 블록체인 데이터 생성"""
    return {
        "address": address,
        "transactions": [
            {
                "hash": f"0x{i}{'0'*63}", 
                "from": address if i % 2 == 0 else f"0x{'a'*40}",
                "to": f"0x{'a'*40}" if i % 2 == 0 else address,
                "value": str(i * 1000000000000000000),  # i ETH
                "gasPrice": "20000000000",
                "gasUsed": "21000",
                "timestamp": str(1600000000 + i * 86400)  # 하루 간격
            } for i in range(1, 6)  # 5개 트랜잭션
        ]
    }

def extract_features(data):
    """특성 추출"""
    address = data["address"]
    txs = data["transactions"]
    
    # 기본 특성 추출
    features = {
        "address": address,
        "total_tx_count": len(txs),
        "total_eth_sent": 0.0,
        "total_eth_received": 0.0
    }
    
    # ETH 전송량 계산
    for tx in txs:
        value_eth = float(tx["value"]) / 1e18
        if tx["from"].lower() == address.lower():
            features["total_eth_sent"] += value_eth
        else:
            features["total_eth_received"] += value_eth
    
    return features

def main():
    """메인 함수"""
    # 테스트 주소
    addresses = [
        "0x1111111111111111111111111111111111111111",
        "0x2222222222222222222222222222222222222222",
        "0x3333333333333333333333333333333333333333"
    ]
    
    # 데이터 컬렉션
    data_collection = {}
    for addr in addresses:
        data_collection[addr] = get_mock_data(addr)
    
    # 특성 추출
    features_list = []
    for addr, data in data_collection.items():
        features = extract_features(data)
        features_list.append(features)
    
    # 데이터프레임 변환
    df = pd.DataFrame(features_list)
    print("특성 데이터프레임:")
    print(df)
    
    # 간단한 클러스터링 (주소 제외)
    X = df.drop("address", axis=1)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # 결과 출력
    df["cluster"] = clusters
    print("\n클러스터링 결과:")
    print(df)
    
    print("\n테스트 완료: 블록체인 데이터 처리 및 클러스터링 성공")

if __name__ == "__main__":
    main() 
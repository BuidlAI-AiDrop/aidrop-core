#!/usr/bin/env python3
"""
테스트 데이터를 사용해 ai-clusturing(비지도 학습)과 ai-deduction(지도 학습) 모듈 테스트
"""

import json
import sys
import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 임시 패키지 모듈 생성
import importlib.util
import types

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not find module {module_name} at {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# ai-clusturing 모듈 임포트
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feature_extraction = import_module_from_file('feature_extraction', 
                                            os.path.join(base_dir, 'ai-clusturing', 'feature_extraction.py'))
clustering = import_module_from_file('clustering', 
                                    os.path.join(base_dir, 'ai-clusturing', 'clustering.py'))
cluster_analyzer = import_module_from_file('cluster_analyzer', 
                                          os.path.join(base_dir, 'ai-clusturing', 'cluster_analyzer.py'))

# ai-deduction 모듈 임포트
feature_engineering = import_module_from_file('feature_engineering', 
                                             os.path.join(base_dir, 'ai-deduction', 'feature_engineering.py'))
model = import_module_from_file('model', 
                               os.path.join(base_dir, 'ai-deduction', 'model.py'))

# 기능 추출기 및 모델 클래스 할당
ClusterFeatureExtractor = feature_extraction.FeatureExtractor
ClusteringModel = clustering.ClusteringModel
ClusterFeatureAnalyzer = cluster_analyzer.ClusterFeatureAnalyzer

DeductionFeatureExtractor = feature_engineering.FeatureExtractor
UserClassificationModel = model.UserClassificationModel

def load_test_data(filename="blockchain_test_data.json"):
    """
    테스트 데이터 로드
    """
    print(f"테스트 데이터 로드 중: {filename}")
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"총 {len(data)} 개의 주소 데이터 로드됨")
    return data

def extract_feature_vectors(data):
    """
    테스트 데이터에서 특성 벡터 추출
    """
    # 데이터 구조화
    addresses_data = []
    true_labels = {}  # 생성된 데이터의 실제 라벨 (분석용)
    
    for address, events in data.items():
        # 실제 라벨 추출
        # 생성된 이벤트의 type 필드를 분석하여 라벨 결정
        contract_types = [event.get('type') for event in events]
        type_counter = Counter(contract_types)
        
        # 1차 축: D-N (DeFi vs NFT)
        defi_ratio = type_counter.get('defi', 0) / len(events) if events else 0
        nft_ratio = type_counter.get('nft', 0) / len(events) if events else 0
        gaming_ratio = type_counter.get('gaming', 0) / len(events) if events else 0
        social_ratio = type_counter.get('social', 0) / len(events) if events else 0
        undefined_ratio = type_counter.get('undefined', 0) / len(events) if events else 0
        
        # 주요 활동 영역 결정
        if gaming_ratio > max(defi_ratio, nft_ratio, social_ratio, undefined_ratio):
            primary_focus = 'G'  # Gaming
        elif social_ratio > max(defi_ratio, nft_ratio, gaming_ratio, undefined_ratio):
            primary_focus = 'S'  # Social
        elif undefined_ratio > max(defi_ratio, nft_ratio, gaming_ratio, social_ratio):
            primary_focus = 'U'  # Undefined
        elif defi_ratio > nft_ratio:
            primary_focus = 'D'  # DeFi
        else:
            primary_focus = 'N'  # NFT
            
        # 2차 축: T-H (Trading vs Holding)
        timestamps = sorted([event.get('time') for event in events])
        if len(timestamps) >= 2:
            timespan = (pd.to_datetime(timestamps[-1]) - pd.to_datetime(timestamps[0])).days
            event_frequency = len(events) / max(timespan, 1)  # 0으로 나누기 방지
            trading_vs_holding = 'T' if event_frequency > 0.3 else 'H'
        else:
            trading_vs_holding = 'H'
        
        # 3차 축: A-S (Aggressive vs Safe)
        # 큰 금액의 거래 비율로 판단
        large_txs = [event for event in events if 
                    isinstance(event['data'], dict) and 
                    event['data'].get('amount', 0) > 5.0]
        large_tx_ratio = len(large_txs) / len(events) if events else 0
        risk_preference = 'A' if large_tx_ratio > 0.2 else 'S'
        
        # 4차 축: C-I (Community vs Individual)
        # DAO 활동 및 소셜 참여로 판단
        dao_ratio = type_counter.get('dao', 0) / len(events) if events else 0
        social_events = [e for e in events if e['type'] == 'social' or 
                        (e['type'] == 'dao' and e['event'] in ['vote', 'comment', 'join'])]
        community_ratio = (len(social_events) + dao_ratio * len(events)) / len(events) if events else 0
        community_preference = 'C' if community_ratio > 0.1 else 'I'
        
        # 완전한 16가지(+추가 유형) 라벨 조합
        full_label = f"{primary_focus}-{trading_vs_holding}-{risk_preference}"
        
        # 커뮤니티 축은 선택적으로 추가 (G/S/U 유형은 제외)
        if primary_focus in ['D', 'N']:
            full_label = f"{full_label}-{community_preference}"
        
        # 무작위 패턴 특별 처리
        if all(ratio < 0.3 for ratio in [defi_ratio, nft_ratio, gaming_ratio, social_ratio, undefined_ratio]):
            full_label = "R-X-X"  # 무작위 패턴
            
        # 라벨 저장
        true_labels[address] = full_label
        
        # 트랜잭션 데이터 변환
        txs = []
        for event in events:
            tx = {
                'hash': f"0x{hash(event['time'] + event['contract'])}", # 가상 트랜잭션 해시
                'timestamp': int(pd.to_datetime(event['time']).timestamp()),
                'from': event['sender'],
                'to': event['contract'],
                'value': event['data'].get('amount', 0) if isinstance(event['data'], dict) else 0,
                'gas': 21000,  # 기본 가스값
                'gas_price': 50 * 10**9,  # 기본 가스 가격 (50 Gwei)
                'method': event['event'],
                'contract_name': event['name'],
                'contract_type': event['type']
            }
            txs.append(tx)
        
        # 토큰 거래 데이터 변환
        token_transfers = []
        for event in events:
            if event['type'] == 'nft' or (event['type'] == 'defi' and 'token' in event['data']):
                token = {
                    'token_address': f"0x{hash(event['name'])}", # 가상 토큰 주소
                    'token_symbol': event['data'].get('token', 'ETH') if isinstance(event['data'], dict) else 'NFT',
                    'token_name': event['name'],
                    'token_type': 'ERC20' if event['type'] == 'defi' else 'ERC721',
                    'amount': event['data'].get('amount', 1) if isinstance(event['data'], dict) else 1,
                    'timestamp': int(pd.to_datetime(event['time']).timestamp()),
                    'from': event['sender'],
                    'to': event['contract']
                }
                token_transfers.append(token)
        
        # 컨트랙트 상호작용
        contract_interactions = []
        for event in events:
            interaction = {
                'contract_address': event['contract'],
                'contract_name': event['name'],
                'contract_type': event['type'],
                'method': event['event'],
                'timestamp': int(pd.to_datetime(event['time']).timestamp()),
                'success': True
            }
            contract_interactions.append(interaction)
        
        # 주소 데이터 구조화
        address_data = {
            'address': address,
            'transactions': txs,
            'token_holdings': token_transfers,
            'contract_interactions': contract_interactions
        }
        
        addresses_data.append(address_data)
    
    return addresses_data, true_labels

def run_clustering(feature_vectors, true_labels):
    """
    비지도 학습 (클러스터링) 수행
    """
    print("\n=== 비지도 학습 (클러스터링) 실행 ===")
    
    # 특성 추출기 초기화
    feature_extractor = ClusterFeatureExtractor()
    
    # 특성 추출
    print("특성 추출 중...")
    features_df = feature_extractor.extract_features_batch(feature_vectors)
    
    # 디버깅 코드 추가
    print(f"features_df 타입: {type(features_df)}")
    
    # DataFrame 확인 및 변환
    if not isinstance(features_df, pd.DataFrame):
        raise TypeError("특성 데이터가 DataFrame 형식이 아닙니다.")
    
    # 특성 전처리
    processed_df = feature_extractor.preprocess_features(features_df)
    print(f"processed_df 타입: {type(processed_df)}")
    
    # 다시 확인
    if not isinstance(processed_df, pd.DataFrame):
        raise TypeError("전처리된 특성 데이터가 DataFrame 형식이 아닙니다.")
        
    print(f"특성 추출 완료: {processed_df.shape[0]} 주소, {processed_df.shape[1]} 특성")
    
    # 클러스터링 모델 초기화
    clustering_model = ClusteringModel(output_dir='./results/cluster_models')
    
    # 클러스터링 수행
    print("클러스터링 수행 중...")
    clustering_results = clustering_model.fit_models(processed_df)
    
    # 클러스터링 모델 저장
    version = datetime.now().strftime('%Y%m%d')
    clustering_model.save_models(version)
    
    # 클러스터 분석
    print("클러스터 분석 중...")
    cluster_analyzer = ClusterFeatureAnalyzer(output_dir='./results/cluster_profiles')
    cluster_profiles = cluster_analyzer.analyze_clusters(
        processed_df,
        clustering_results['kmeans']['labels']
    )
    
    # 결과 저장
    cluster_analyzer.save_cluster_profiles(version)
    
    # 클러스터와 실제 라벨 비교
    cluster_labels = clustering_results['kmeans']['labels']
    addresses = [data['address'] for data in feature_vectors]
    
    # 클러스터별 실제 라벨 분포 분석
    print("\n--- 클러스터와 실제 라벨 비교 ---\n")
    cluster_distribution = {}
    
    for cluster_id in range(np.max(cluster_labels) + 1):
        # 해당 클러스터에 속한 주소들의 인덱스
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_size = len(indices)
        
        if cluster_size == 0:
            continue
            
        print(f"클러스터 {cluster_id} (크기: {cluster_size}):")
        
        # 해당 클러스터 내 라벨 카운트
        labels_in_cluster = [true_labels[addresses[i]] for i in indices]
        label_counts = Counter(labels_in_cluster)
        
        # 상세 라벨 분포 저장
        cluster_distribution[str(cluster_id)] = {
            "size": cluster_size,
            "labels": {label: {"count": count, "percentage": round(count / cluster_size * 100, 1)} 
                     for label, count in label_counts.items()}
        }
        
        # 라벨 분포 출력 - 상위 5개만
        for label, count in label_counts.most_common(10):
            percentage = round(count / cluster_size * 100, 1)
            print(f"  {label}: {count} ({percentage}%)")
        
        print()
    
    # 클러스터 분포 저장
    os.makedirs('./results/cluster_analysis', exist_ok=True)
    with open('./results/cluster_analysis/cluster_distribution.json', 'w') as f:
        json.dump(cluster_distribution, f, indent=2)
    print(f"클러스터 분포가 './results/cluster_analysis/cluster_distribution.json'에 저장되었습니다.")
    
    # t-SNE 시각화 생성
    from sklearn.manifold import TSNE
    import matplotlib.cm as cm
    
    # 차원 축소
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(processed_df)
    
    # 시각화
    plt.figure(figsize=(12, 10))
    
    # 클러스터별 색상 설정
    unique_clusters = np.unique(cluster_labels)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    
    # 클러스터별 플롯
    for i, cluster_id in enumerate(unique_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        plt.scatter(
            reduced_data[indices, 0], 
            reduced_data[indices, 1],
            s=80, 
            c=[colors[i]],
            label=f'클러스터 {cluster_id} (n={len(indices)})'
        )
    
    # 주요 유형 라벨 추가
    # 각 클러스터의 주요 특성 표시
    for cluster_id in range(np.max(cluster_labels) + 1):
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) == 0:
            continue
            
        # 해당 클러스터 중심점 계산
        center_x = np.mean(reduced_data[indices, 0])
        center_y = np.mean(reduced_data[indices, 1])
        
        # 클러스터에서 가장 많은 라벨 2개 찾기
        labels_in_cluster = [true_labels[addresses[i]] for i in indices]
        most_common = Counter(labels_in_cluster).most_common(2)
        
        # 클러스터의 대표 라벨 텍스트 생성
        if most_common:
            label_text = "\n".join([f"{label}: {count}" for label, count in most_common])
            plt.annotate(
                label_text,
                (center_x, center_y),
                fontsize=8,
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
    
    plt.title('클러스터링 시각화 (t-SNE)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # 시각화 저장
    os.makedirs('./results/visualizations', exist_ok=True)
    plt.savefig('./results/visualizations/clustering_visualization.png', dpi=300, bbox_inches='tight')
    print(f"클러스터 시각화가 './results/visualizations/clustering_visualization.png'에 저장되었습니다.")
    
    # 클러스터링 결과 반환
    return {
        'kmeans': clustering_results['kmeans'],
        'processed_df': processed_df,
        'cluster_profiles': cluster_profiles
    }

def run_classification(feature_vectors, true_labels):
    """
    지도 학습 (분류) 수행
    """
    print("\n=== 지도 학습 (분류) 실행 ===")
    
    # 특성 추출기 초기화
    feature_extractor = DeductionFeatureExtractor()
    
    # 특성 추출
    print("특성 추출 중...")
    features_list = []
    labels_list = []
    addresses = []
    
    for addr_data in feature_vectors:
        address = addr_data['address']
        addresses.append(address)
        
        # 특성 추출
        features = feature_extractor.extract_features(
            address=address,
            transactions=addr_data['transactions'],
            token_holdings=addr_data['token_holdings'],
            contract_interactions=addr_data['contract_interactions']
        )
        
        # 유형 라벨 (처음 두 글자만 사용 - D/N, T/H)
        label = true_labels.get(address, 'unknown')[:3]  # D-T, D-H, N-T, N-H
        
        features_list.append(features)
        labels_list.append(label)
    
    # 데이터프레임 변환
    features_df = pd.DataFrame(features_list)
    
    # 라벨 인코딩
    label_mapping = {label: idx for idx, label in enumerate(set(labels_list))}
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}
    labels = np.array([label_mapping[label] for label in labels_list])
    
    print(f"데이터 준비 완료: {len(features_df)} 샘플, {len(features_df.columns)} 특성")
    print(f"라벨 분포: {Counter(labels_list)}")
    
    # 데이터와 라벨 저장
    os.makedirs('./results/classification_data', exist_ok=True)
    features_df['address'] = addresses
    features_df['label'] = labels_list
    features_df.to_csv('./results/classification_data/prepared_data.csv', index=False)
    print("분류 데이터가 './results/classification_data/prepared_data.csv'에 저장되었습니다.")
    
    # 라벨 분포 시각화
    plt.figure(figsize=(10, 6))
    sns.countplot(x=labels_list)
    plt.title('User Type Distribution', fontsize=14)
    plt.xlabel('User Type')
    plt.ylabel('Count')
    plt.savefig('./results/visualizations/label_distribution.png', dpi=300, bbox_inches='tight')
    print("라벨 분포가 './results/visualizations/label_distribution.png'에 저장되었습니다.")
    
    # 훈련/테스트 데이터 분할
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features_df.drop(['address', 'label'], axis=1), labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # 모델 초기화
    model = UserClassificationModel(model_name="test_model")
    
    # 모델 훈련
    print("모델 훈련 중...")
    model.train(X_train, y_train)
    
    # 모델 평가
    print("모델 평가 중...")
    accuracy = model.evaluate(X_test, y_test)
    print(f"모델 정확도: {accuracy:.2f}")
    
    # 모델 성능 평가 및 결과 저장
    y_pred = model.model.predict(model.scaler.transform(X_test))
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[reverse_mapping[i] for i in range(len(reverse_mapping))], 
               yticklabels=[reverse_mapping[i] for i in range(len(reverse_mapping))])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('./results/visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("혼동 행렬이 './results/visualizations/confusion_matrix.png'에 저장되었습니다.")
    
    # 분류 보고서 생성 및 저장
    report = classification_report(y_test, y_pred, target_names=[reverse_mapping[i] for i in range(len(reverse_mapping))], output_dict=True)
    with open('./results/classification_data/classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("분류 보고서가 './results/classification_data/classification_report.json'에 저장되었습니다.")
    
    # 특성 중요도
    feature_importance = model.get_feature_importance()
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n상위 10개 중요 특성:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")
    
    # 특성 중요도 시각화
    plt.figure(figsize=(12, 8))
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    sns.barplot(x=importances, y=features)
    plt.title('Top Feature Importance', fontsize=14)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('./results/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    print("특성 중요도가 './results/visualizations/feature_importance.png'에 저장되었습니다.")
    
    # 모델 저장
    model_path = model.save("./results/deduction_models/test_model")
    print(f"모델이 '{model_path}'에 저장되었습니다.")
    
    # 모델 요약 정보 저장
    model_summary = {
        "model_type": type(model.model).__name__,
        "accuracy": float(accuracy),
        "feature_importance": dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)),
        "label_mapping": label_mapping,
        "reverse_mapping": reverse_mapping,
        "timestamp": datetime.now().isoformat()
    }
    
    with open('./results/classification_data/model_summary.json', 'w') as f:
        json.dump(model_summary, f, indent=2)
    print("모델 요약 정보가 './results/classification_data/model_summary.json'에 저장되었습니다.")
    
    return model, label_mapping

def generate_combined_report(clustering_results, classification_model, true_labels):
    """
    비지도 학습과 지도 학습 결과를 통합한 종합 보고서 생성
    """
    print("\n=== 종합 보고서 생성 중 ===")
    
    # numpy 타입을 Python 기본 타입으로 변환하는 헬퍼 함수
    def convert_numpy_types(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    # 클러스터 크기 계산 및 변환 (numpy 타입을 일반 타입으로 변환)
    cluster_sizes = {}
    for cluster_id in set(clustering_results['kmeans']['labels']):
        cluster_sizes[int(cluster_id)] = int(sum(clustering_results['kmeans']['labels'] == cluster_id))
    
    # 종합 보고서 구조
    combined_report = {
        "timestamp": datetime.now().isoformat(),
        "data_summary": {
            "total_addresses": len(true_labels),
            "label_distribution": dict(Counter(true_labels.values()))
        },
        "clustering_summary": {
            "num_clusters": len(set(clustering_results['kmeans']['labels'])),
            "cluster_sizes": cluster_sizes,
            "silhouette_score": clustering_results.get('kmeans', {}).get('silhouette_score', None)
        },
        "classification_summary": {
            "model_type": type(classification_model.model).__name__,
            "accuracy": classification_model.model.score(
                classification_model.scaler.transform(classification_model._X_test),
                classification_model._y_test
            ) if hasattr(classification_model, '_X_test') else None,
            "top_features": dict(sorted(
                classification_model.get_feature_importance().items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]) if hasattr(classification_model, 'get_feature_importance') else {}
        }
    }
    
    # NumPy 타입을 Python 기본 타입으로 변환
    combined_report = convert_numpy_types(combined_report)
    
    # 클러스터와 분류 결과 비교 분석
    # 이 부분은 데이터 구조에 따라 달라질 수 있음
    try:
        # 두 모델의 결과를 연결하기 위한 코드
        # 실제 구현에서는 데이터 구조에 맞게 조정 필요
        combined_report["combined_analysis"] = {
            "description": "분류 모델과 클러스터링 모델의 결과를 비교 분석하여 인사이트 도출"
        }
    except Exception as e:
        combined_report["combined_analysis"] = {
            "error": str(e)
        }
    
    # 종합 보고서 저장
    os.makedirs('./results', exist_ok=True)
    with open('./results/combined_analysis_report.json', 'w') as f:
        json.dump(combined_report, f, indent=2)
    
    print("종합 분석 보고서가 './results/combined_analysis_report.json'에 저장되었습니다.")
    
    return combined_report

def main():
    """
    메인 실행 함수
    """
    # 테스트 데이터 로드
    try:
        test_data = load_test_data("blockchain_test_data.json")
    except Exception as e:
        print(f"Error: {e}")
        test_data = load_test_data("test_data/blockchain_test_data.json")
    
    # 특성 벡터 추출
    feature_vectors, true_labels = extract_feature_vectors(test_data)
    
    # 비지도 학습 (클러스터링) 실행
    clustering_results = run_clustering(feature_vectors, true_labels)
    
    # 지도 학습 (분류) 실행
    classification_model, test_results = run_classification(feature_vectors, true_labels)
    
    # 종합 보고서 생성
    generate_combined_report(clustering_results, classification_model, true_labels)
    
    print("\n=== 테스트 완료 ===")
    print("클러스터링 결과와 분류 모델이 성공적으로 생성되었습니다.")
    print("모든 분석 결과와 시각화가 './results/' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 
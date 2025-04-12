"""
비지도 학습 클러스터링을 통한 사용자 특성 추출 메인 모듈
"""

import os
import sys
import argparse
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from .utils import setup_logger, load_addresses_data, save_results, save_dataframe
from .feature_extraction import FeatureExtractor
from .clustering import ClusteringModel
from .cluster_analyzer import ClusterFeatureAnalyzer

logger = setup_logger('main')

def run_clustering(data_path: str, output_dir: str = 'results', 
                  model_dir: str = 'models', version: str = 'v1') -> Dict[str, Any]:
    """
    클러스터링 실행 및 사용자 특성 추출
    
    Args:
        data_path: 주소 데이터 파일 경로
        output_dir: 결과 저장 디렉토리
        model_dir: 모델 저장 디렉토리
        version: 저장할 모델 버전
        
    Returns:
        클러스터링 결과 딕셔너리
    """
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 데이터 로드
    logger.info(f"데이터 로드: {data_path}")
    addresses_data = load_addresses_data(data_path)
    
    # 2. 특성 추출
    logger.info("특성 추출 시작")
    feature_extractor = FeatureExtractor()
    features_df = feature_extractor.extract_features_batch(addresses_data)
    
    # 특성 전처리
    logger.info("특성 전처리 진행")
    processed_df, scaler = feature_extractor.preprocess_features(features_df)
    
    # 특성 저장
    features_file = save_dataframe(processed_df, "features.csv", output_dir)
    logger.info(f"특성 저장 완료: {features_file}")
    
    # 3. 클러스터링
    logger.info("클러스터링 모델 학습 시작")
    clustering = ClusteringModel(output_dir=model_dir)
    clustering_results = clustering.fit_models(processed_df)
    
    # 모델 저장
    clustering.save_models(version=version)
    logger.info(f"클러스터링 모델 저장 완료: {model_dir}/{version}")
    
    # K-Means 결과 시각화
    if 'kmeans' in clustering_results:
        kmeans_labels = clustering_results['kmeans']['labels']
        viz_file = clustering.visualize_clusters(
            processed_df, kmeans_labels, "K-Means Clusters"
        )
        logger.info(f"클러스터 시각화 저장: {viz_file}")
    
    # 4. 클러스터 분석 및 사용자 특성 추출
    logger.info("클러스터 분석 및 사용자 특성 추출 시작")
    analyzer = ClusterFeatureAnalyzer(output_dir=os.path.join(output_dir, 'profiles'))
    
    # K-Means 클러스터 분석
    cluster_profiles = analyzer.analyze_clusters(features_df, kmeans_labels)
    
    # 프로필 저장
    analyzer.save_cluster_profiles(version=version)
    logger.info("클러스터 프로필 저장 완료")
    
    # 5. 결과 요약 저장
    results = {
        'num_addresses': len(addresses_data),
        'num_features': len(processed_df.columns),
        'clustering': {
            'kmeans': {
                'num_clusters': clustering_results['kmeans']['num_clusters'],
                'cluster_sizes': {
                    f"cluster_{i}": int(sum(kmeans_labels == i)) 
                    for i in range(clustering_results['kmeans']['num_clusters'])
                }
            }
        },
        'user_traits': list(analyzer.USER_TRAITS.keys()),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    results_file = save_results(results, "clustering_summary.json", output_dir)
    logger.info(f"결과 요약 저장 완료: {results_file}")
    
    return results

def analyze_new_address(address_data: Dict, model_dir: str = 'models', 
                       profile_dir: str = 'results/profiles', 
                       version: str = 'v1') -> Dict[str, Any]:
    """
    새 주소 분석 및 사용자 특성 추출
    
    Args:
        address_data: 주소 데이터 (특성이 추출된 딕셔너리)
        model_dir: 모델 저장 디렉토리
        profile_dir: 프로필 저장 디렉토리
        version: 모델 버전
        
    Returns:
        분석 결과 딕셔너리
    """
    # 1. 특성 추출
    feature_extractor = FeatureExtractor()
    
    # 단일 주소 특성 추출
    address = address_data.get('address')
    transactions = address_data.get('transactions', [])
    token_holdings = address_data.get('token_holdings', [])
    contract_interactions = address_data.get('contract_interactions', [])
    
    features = feature_extractor.extract_features(
        address, transactions, token_holdings, contract_interactions
    )
    
    # 특성 DataFrame 생성
    features_df = pd.DataFrame([features])
    features_df.index = [address]
    
    # 2. 클러스터링 모델 로드
    clustering = ClusteringModel(output_dir=model_dir)
    model_loaded = clustering.load_model('kmeans', version)
    
    if not model_loaded:
        logger.error("클러스터링 모델 로드 실패")
        return {'error': '모델 로드 실패'}
    
    # 3. 클러스터 예측
    cluster = int(clustering.predict(features_df, 'kmeans')[0])
    
    # 4. 사용자 특성 분석기 로드
    analyzer = ClusterFeatureAnalyzer(output_dir=profile_dir)
    profiles_loaded = analyzer.load_cluster_profiles(version)
    
    if not profiles_loaded:
        logger.error("클러스터 프로필 로드 실패")
        return {'error': '프로필 로드 실패'}
    
    # 5. 사용자 특성 추출
    user_traits = analyzer.get_user_traits(features)
    
    # 주요 특성 선택
    primary_traits = analyzer.get_primary_traits(user_traits)
    
    # 6. 결과 반환
    result = {
        'address': address,
        'cluster': cluster,
        'user_traits': user_traits,
        'primary_traits': primary_traits,
        'cluster_profile': analyzer.cluster_profiles.get(f'cluster_{cluster}')
    }
    
    return result

def print_help():
    """사용법 출력"""
    print("사용법: python -m ai_clusturing.main [command] [options]")
    print("\n명령어:")
    print("  cluster <data_path>  - 주소 데이터에 대한 클러스터링 실행")
    print("  analyze <data_path>  - 단일 주소 분석 (지갑 특성 추출)")
    print("\n옵션:")
    print("  --output <dir>      - 결과 저장 디렉토리 (기본값: results)")
    print("  --model <dir>       - 모델 저장 디렉토리 (기본값: models)")
    print("  --version <version> - 모델 버전 (기본값: v1)")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="비지도 학습 클러스터링을 통한 사용자 특성 추출")
    parser.add_argument('command', choices=['cluster', 'analyze', 'help'],
                       help="실행할 명령어 (cluster, analyze, help)")
    parser.add_argument('data_path', nargs='?', help="주소 데이터 파일 경로")
    parser.add_argument('--output', default='results', help="결과 저장 디렉토리")
    parser.add_argument('--model', default='models', help="모델 저장 디렉토리")
    parser.add_argument('--version', default='v1', help="모델 버전")
    parser.add_argument('--address', help="분석할 지갑 주소 (analyze 명령어용)")
    
    args = parser.parse_args()
    
    if args.command == 'help':
        print_help()
        return
    
    if not args.data_path:
        print("오류: 데이터 파일 경로가 필요합니다.")
        print_help()
        return
    
    if args.command == 'cluster':
        logger.info(f"클러스터링 실행 시작: {args.data_path}")
        results = run_clustering(
            args.data_path, args.output, args.model, args.version
        )
        logger.info("클러스터링 완료")
        
    elif args.command == 'analyze':
        if args.address:
            # 단일 주소 분석은 별도의 API로 제공됨
            logger.info(f"주소 분석: {args.address}")
            print("개별 주소 분석 기능은 API를 통해 제공됩니다.")
        else:
            # 데이터 파일에 있는 모든 주소 분석
            logger.info(f"데이터 파일 내 주소 분석: {args.data_path}")
            addresses_data = load_addresses_data(args.data_path)
            
            results = []
            for address_data in addresses_data[:10]:  # 예시로 처음 10개만 분석
                result = analyze_new_address(
                    address_data, args.model, os.path.join(args.output, 'profiles'), args.version
                )
                results.append(result)
            
            # 결과 저장
            save_results({'addresses': results}, "address_analysis.json", args.output)
            logger.info(f"주소 분석 완료: {len(results)}개")

if __name__ == '__main__':
    main() 
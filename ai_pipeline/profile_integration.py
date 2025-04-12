#!/usr/bin/env python3
"""
AI 분석 결과와 프로필 이미지 생성기를 통합하는 모듈
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import importlib.util
import types

# 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# 프로필 생성기 임포트
from profile_generator import AIProfileGenerator

class ProfileIntegration:
    """AI 모델과 프로필 생성기를 통합하는 클래스"""
    
    def __init__(self, models_dir=None, results_dir=None, output_dir=None):
        """
        초기화 함수
        
        Args:
            models_dir: 학습된 모델 디렉토리
            results_dir: 분석 결과 저장 디렉토리
            output_dir: 프로필 이미지 저장 디렉토리
        """
        self.models_dir = models_dir or os.path.join(base_dir, 'results/deduction_models')
        self.results_dir = results_dir or os.path.join(base_dir, 'results')
        self.output_dir = output_dir or os.path.join(self.results_dir, 'profiles')
        
        # 결과 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 프로필 생성기 초기화
        self.profile_generator = AIProfileGenerator(output_dir=self.output_dir)
        
        # 모델 로드
        self.models = self._load_models()
        
        # 결과 데이터 로드
        self.results = self._load_results()
    
    def _load_models(self):
        """
        학습된 모델 로드
        
        Returns:
            모델 객체 딕셔너리
        """
        models = {}
        
        # ai_deduction 분류 모델 확인
        deduction_model_path = os.path.join(self.models_dir, 'test_model')
        if os.path.exists(deduction_model_path):
            try:
                # 임포트 모듈
                if os.path.exists(os.path.join(base_dir, 'ai_deduction', 'model.py')):
                    model_module = self._import_module('model', 
                                    os.path.join(base_dir, 'ai_deduction', 'model.py'))
                    
                    # 모델 로드
                    model_class = getattr(model_module, 'UserClassificationModel', None)
                    if model_class:
                        models['classification'] = model_class(model_name="test_model")
                        models['classification'].load(deduction_model_path)
                        print(f"분류 모델을 로드했습니다: {deduction_model_path}")
            except Exception as e:
                print(f"분류 모델 로드 실패: {e}")
        
        # ai_clusturing 클러스터링 모델 확인
        clustering_model_path = os.path.join(base_dir, 'results/cluster_models')
        if os.path.exists(clustering_model_path):
            try:
                # 임포트 모듈
                if os.path.exists(os.path.join(base_dir, 'ai_clusturing', 'clustering.py')):
                    cluster_module = self._import_module('clustering', 
                                    os.path.join(base_dir, 'ai_clusturing', 'clustering.py'))
                    
                    # 모델 로드
                    model_class = getattr(cluster_module, 'ClusteringModel', None)
                    if model_class:
                        models['clustering'] = model_class(output_dir=clustering_model_path)
                        # 최신 버전 찾기
                        versions = [d for d in os.listdir(clustering_model_path) 
                                   if os.path.isdir(os.path.join(clustering_model_path, d))]
                        if versions:
                            latest_version = max(versions)
                            models['clustering'].load_models(latest_version)
                            print(f"클러스터링 모델을 로드했습니다: {clustering_model_path}/{latest_version}")
            except Exception as e:
                print(f"클러스터링 모델 로드 실패: {e}")
        
        return models
    
    def _load_results(self):
        """
        분석 결과 로드
        
        Returns:
            결과 데이터 딕셔너리
        """
        results = {}
        
        # 클러스터링 결과
        cluster_file = os.path.join(self.results_dir, "cluster_analysis/cluster_distribution.json")
        if os.path.exists(cluster_file):
            try:
                with open(cluster_file, 'r') as f:
                    results['cluster_distribution'] = json.load(f)
                print(f"클러스터 분포를 로드했습니다: {cluster_file}")
            except Exception as e:
                print(f"클러스터 분포 로드 실패: {e}")
        
        # 분류 결과
        classification_file = os.path.join(self.results_dir, "classification_data/model_summary.json")
        if os.path.exists(classification_file):
            try:
                with open(classification_file, 'r') as f:
                    results['classification_summary'] = json.load(f)
                print(f"분류 요약을 로드했습니다: {classification_file}")
            except Exception as e:
                print(f"분류 요약 로드 실패: {e}")
        
        # 종합 보고서
        combined_file = os.path.join(self.results_dir, "combined_analysis_report.json")
        if os.path.exists(combined_file):
            try:
                with open(combined_file, 'r') as f:
                    results['combined_report'] = json.load(f)
                print(f"종합 보고서를 로드했습니다: {combined_file}")
            except Exception as e:
                print(f"종합 보고서 로드 실패: {e}")
        
        # 원본 데이터 로드
        data_file = os.path.join(base_dir, "test_data/blockchain_test_data.json")
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r') as f:
                    results['raw_data'] = json.load(f)
                print(f"원본 데이터를 로드했습니다: {data_file}")
            except Exception as e:
                print(f"원본 데이터 로드 실패: {e}")
        
        return results
    
    def _import_module(self, module_name, file_path):
        """
        파일 경로에서 모듈 동적 임포트
        
        Args:
            module_name: 모듈 이름
            file_path: 파일 경로
            
        Returns:
            임포트된 모듈
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"모듈을 찾을 수 없습니다: {module_name} ({file_path})")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    
    def _extract_user_features(self, address, address_data):
        """
        사용자 주소에서 특성 추출
        
        Args:
            address: 사용자 주소
            address_data: 주소 이벤트 데이터
            
        Returns:
            추출된 특성
        """
        from collections import Counter
        import pandas as pd
        
        # 이벤트 유형 카운트
        contract_types = [event.get('type') for event in address_data]
        type_counter = Counter(contract_types)
        
        # 총 이벤트 수
        total_events = len(address_data)
        
        # 유형별 비율 계산
        defi_ratio = type_counter.get('defi', 0) / total_events if total_events else 0
        nft_ratio = type_counter.get('nft', 0) / total_events if total_events else 0
        gaming_ratio = type_counter.get('gaming', 0) / total_events if total_events else 0
        social_ratio = type_counter.get('social', 0) / total_events if total_events else 0
        
        # 시간 패턴 분석
        timestamps = sorted([event.get('time') for event in address_data])
        
        # 거래 빈도 계산
        if len(timestamps) >= 2:
            try:
                timespan = (pd.to_datetime(timestamps[-1]) - pd.to_datetime(timestamps[0])).days
                event_frequency = len(address_data) / max(timespan, 1)  # 0으로 나누기 방지
            except:
                event_frequency = 0.1
        else:
            event_frequency = 0.1
        
        # 평균 거래 금액 계산
        amounts = []
        for event in address_data:
            if isinstance(event.get('data'), dict):
                amount = event['data'].get('amount', 0)
                if isinstance(amount, (int, float)):
                    amounts.append(amount)
        
        avg_amount = sum(amounts) / len(amounts) if amounts else 0
        
        # 특성 반환
        return {
            'transaction_count': total_events,
            'transaction_frequency': event_frequency,
            'avg_amount': avg_amount,
            'distinct_contracts': len(set(event['contract'] for event in address_data)),
            'token_diversity': len(set(event['data'].get('token') for event in address_data 
                                    if isinstance(event.get('data'), dict) and 'token' in event.get('data', {}))),
            'defi_ratio': defi_ratio,
            'nft_ratio': nft_ratio,
            'gaming_ratio': gaming_ratio,
            'social_ratio': social_ratio,
        }
    
    def _determine_user_type(self, address, features):
        """
        사용자 유형 결정
        
        Args:
            address: 사용자 주소
            features: 사용자 특성
            
        Returns:
            사용자 유형 문자열
        """
        # 1차 축: D-N (DeFi vs NFT)
        if features['gaming_ratio'] > max(features['defi_ratio'], features['nft_ratio'], features['social_ratio']):
            primary_focus = 'G'  # Gaming
        elif features['social_ratio'] > max(features['defi_ratio'], features['nft_ratio'], features['gaming_ratio']):
            primary_focus = 'S'  # Social
        elif features['defi_ratio'] > features['nft_ratio']:
            primary_focus = 'D'  # DeFi
        else:
            primary_focus = 'N'  # NFT
            
        # 2차 축: T-H (Trading vs Holding)
        trading_vs_holding = 'T' if features['transaction_frequency'] > 0.3 else 'H'
        
        # 3차 축: A-S (Aggressive vs Safe)
        risk_preference = 'A' if features['avg_amount'] > 2.0 else 'S'
        
        # 4차 축: C-I (Community vs Individual) - D, N 유형만 적용
        if primary_focus in ['D', 'N']:
            community_preference = 'C' if features['social_ratio'] > 0.1 else 'I'
            return f"{primary_focus}-{trading_vs_holding}-{risk_preference}-{community_preference}"
        else:
            return f"{primary_focus}-{trading_vs_holding}-{risk_preference}"
    
    def analyze_address(self, address):
        """
        주소 분석 및 프로필 생성
        
        Args:
            address: 사용자 지갑 주소
            
        Returns:
            분석 결과 및 프로필 경로
        """
        # 주소 데이터 확인
        if 'raw_data' not in self.results or address not in self.results['raw_data']:
            print(f"주소에 대한 데이터를 찾을 수 없습니다: {address}")
            return None
        
        # 원본 이벤트 데이터
        address_data = self.results['raw_data'][address]
        
        # 특성 추출
        features = self._extract_user_features(address, address_data)
        
        # 사용자 유형 결정
        user_type = self._determine_user_type(address, features)
        
        # 분석 결과 준비
        analysis_result = {
            'address': address,
            'user_type': user_type,
            'features': features,
            'timestamp': datetime.now().isoformat()
        }
        
        # 분석 결과 저장
        result_path = os.path.join(self.results_dir, 'address_analysis')
        os.makedirs(result_path, exist_ok=True)
        
        with open(os.path.join(result_path, f"{address[:8]}_analysis.json"), 'w') as f:
            json.dump(analysis_result, f, indent=2)
        
        # 프로필 이미지 생성
        image_path = self.profile_generator.generate_profile(
            address,
            save=True,
            show_prompt=True
        )
        
        print(f"\n[분석 결과] 주소: {address[:8]}...")
        print(f"사용자 유형: {user_type}")
        print(f"주요 특성: DeFi {features['defi_ratio']:.2f}, NFT {features['nft_ratio']:.2f}, 게임 {features['gaming_ratio']:.2f}, 소셜 {features['social_ratio']:.2f}")
        print(f"거래 빈도: {features['transaction_frequency']:.2f}, 평균 금액: {features['avg_amount']:.2f}")
        
        # 결과 반환
        return {
            'analysis': analysis_result,
            'profile_image': image_path
        }
    
    def batch_process(self, addresses=None, limit=10):
        """
        여러 주소 일괄 처리
        
        Args:
            addresses: 처리할 주소 목록 (없으면 원본 데이터의 주소 사용)
            limit: 처리할 최대 주소 수
            
        Returns:
            처리된 주소 및 결과
        """
        results = {}
        
        # 주소 목록 결정
        if addresses is None and 'raw_data' in self.results:
            addresses = list(self.results['raw_data'].keys())[:limit]
        
        if not addresses:
            print("처리할 주소가 없습니다.")
            return results
        
        # 각 주소 처리
        for address in addresses:
            print(f"주소 처리 중: {address}")
            result = self.analyze_address(address)
            if result:
                results[address] = result
        
        # 처리 결과 요약
        print(f"\n총 {len(results)}개 주소 처리 완료")
        print(f"프로필 이미지가 {self.output_dir}에 저장되었습니다.")
        
        return results

# 독립 실행을 위한 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI 분석 결과를 사용한 사용자 프로필 생성")
    parser.add_argument("--address", type=str, help="분석할 사용자 주소")
    parser.add_argument("--models-dir", type=str, help="모델 디렉토리")
    parser.add_argument("--results-dir", type=str, help="결과 디렉토리")
    parser.add_argument("--output-dir", type=str, help="프로필 이미지 저장 디렉토리")
    parser.add_argument("--batch", action="store_true", help="일괄 처리 모드")
    parser.add_argument("--limit", type=int, default=10, help="일괄 처리 시 최대 주소 수")
    
    args = parser.parse_args()
    
    # 통합 객체 생성
    integration = ProfileIntegration(
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    # 실행 모드 결정
    if args.batch:
        # 일괄 처리
        integration.batch_process(limit=args.limit)
    elif args.address:
        # 단일 주소 처리
        result = integration.analyze_address(args.address)
        if result:
            print(f"분석 결과: {result['analysis']['user_type']}")
            print(f"프로필 이미지: {result['profile_image']}")
    else:
        # 기본 실행 (일부 샘플 주소 처리)
        if 'raw_data' in integration.results:
            sample_addresses = list(integration.results['raw_data'].keys())[:3]
            if sample_addresses:
                integration.batch_process(addresses=sample_addresses)
            else:
                print("샘플 주소를 찾을 수 없습니다.")
        else:
            print("원본 데이터가 로드되지 않았습니다. --address 옵션을 사용하여 특정 주소를 분석하세요.") 
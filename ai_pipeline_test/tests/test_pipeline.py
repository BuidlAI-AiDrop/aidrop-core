#!/usr/bin/env python3
"""
AI 파이프라인 테스트 모듈
목 데이터를 사용한 파이프라인 통합 테스트
"""

import os
import sys
import unittest
import json
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# 테스트 대상 모듈 임포트
from ai_pipeline.pipeline import AIPipeline
from ai_pipeline.service import IntegratedAnalysisService
from ai_pipeline.utils import setup_logger

logger = setup_logger('test_pipeline')

class MockBlockchainData:
    """블록체인 데이터 목 클래스"""
    
    @staticmethod
    def get_mock_transactions(address):
        """모의 트랜잭션 데이터 생성"""
        return [
            {
                "hash": f"0x{i}{'0'*63}", 
                "from": address if i % 2 == 0 else f"0x{'a'*40}",
                "to": f"0x{'a'*40}" if i % 2 == 0 else address,
                "value": str(i * 1000000000000000000),  # i ETH
                "gasPrice": "20000000000",
                "gasUsed": "21000",
                "timestamp": str(1600000000 + i * 86400)  # 하루 간격
            } for i in range(1, 11)  # 10개 트랜잭션
        ]
    
    @staticmethod
    def get_mock_token_transfers(address):
        """모의 토큰 전송 데이터 생성"""
        return [
            {
                "hash": f"0x{i}{'0'*63}",
                "tokenSymbol": f"TOKEN{i}",
                "tokenName": f"Test Token {i}",
                "tokenDecimal": "18",
                "from": address if i % 2 == 0 else f"0x{'b'*40}",
                "to": f"0x{'b'*40}" if i % 2 == 0 else address,
                "value": str(i * 1000000000000000000),
                "timestamp": str(1600000000 + i * 86400)
            } for i in range(1, 6)  # 5개 토큰 전송
        ]
    
    @staticmethod
    def get_mock_internal_transactions(address):
        """모의 내부 트랜잭션 데이터 생성"""
        return [
            {
                "hash": f"0x{i}{'0'*63}",
                "from": address if i % 2 == 0 else f"0x{'c'*40}",
                "to": f"0x{'c'*40}" if i % 2 == 0 else address,
                "value": str(i * 500000000000000000),
                "timestamp": str(1600000000 + i * 86400)
            } for i in range(1, 4)  # 3개 내부 트랜잭션
        ]
    
    @staticmethod
    def get_mock_wallet_data(address):
        """모의 지갑 데이터 생성"""
        return {
            "address": address,
            "transactions": MockBlockchainData.get_mock_transactions(address),
            "token_transfers": MockBlockchainData.get_mock_token_transfers(address),
            "internal_transactions": MockBlockchainData.get_mock_internal_transactions(address)
        }

class TestAIPipeline(unittest.TestCase):
    """AI 파이프라인 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        # 임시 디렉토리 생성
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.test_dir, 'models')
        self.data_dir = os.path.join(self.test_dir, 'data')
        self.results_dir = os.path.join(self.test_dir, 'results')
        
        # 테스트 주소
        self.test_addresses = [
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "0x3333333333333333333333333333333333333333",
            "0x4444444444444444444444444444444444444444",
            "0x5555555555555555555555555555555555555555"
        ]
        
        # 모의 학습 데이터 생성
        self.create_mock_training_data()
        
        # 파이프라인 패치 설정
        self.patches = []
        
        # 데이터 수집 모의 패치
        collector_patch = patch('ai_pipeline.pipeline.BlockchainDataCollector')
        self.mock_collector = collector_patch.start()
        self.patches.append(collector_patch)
        
        # collect_address_data 모의 설정
        mock_collector_instance = MagicMock()
        mock_collector_instance.collect_address_data.side_effect = \
            lambda addr: MockBlockchainData.get_mock_wallet_data(addr)
        mock_collector_instance.collect_wallet_data.side_effect = \
            lambda addr: {
                "status": "success", 
                "data": MockBlockchainData.get_mock_wallet_data(addr)
            }
        self.mock_collector.return_value = mock_collector_instance
        
        # 서비스 패치
        service_patch = patch('ai_pipeline.pipeline.InferenceService')
        self.mock_service = service_patch.start()
        self.patches.append(service_patch)
        
        # 기타 필요한 패치 추가
        
    def tearDown(self):
        """테스트 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.test_dir)
        
        # 패치 종료
        for p in self.patches:
            p.stop()
    
    def create_mock_training_data(self):
        """모의 학습 데이터 생성"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 특성 데이터 생성
        features = []
        for i, address in enumerate(self.test_addresses):
            # 기본 특성
            feature = {
                "address": address,
                "total_tx_count": 10 + i,
                "eth_balance_eth": float(i * 2.5),
                "unique_tokens_count": 5 + i,
                "wallet_age_days": 100 + i * 10,
                "total_eth_sent": float(i * 3.5),
                "total_eth_received": float(i * 4.2),
                "avg_tx_value": float(i * 1.3),
                "max_tx_value": float(i * 5),
                "min_tx_value": 0.01,
                "tx_frequency_per_day": 0.5 + i * 0.2,
                "unique_counterparties": 7 + i,
                "contract_interaction_count": 3 + i,
                "token_diversity_index": 0.3 + i * 0.1,
                "defi_interaction": i % 2 == 0,
                "nft_interaction": i % 3 == 0,
                "exchange_interaction": i % 4 == 0
            }
            features.append(feature)
        
        # CSV 형식으로 저장
        df = pd.DataFrame(features)
        training_file = os.path.join(self.data_dir, "training_data.csv")
        df.to_csv(training_file, index=False)
        
        self.training_file = training_file
    
    def test_pipeline_initialization(self):
        """파이프라인 초기화 테스트"""
        pipeline = AIPipeline(
            cluster_model_dir=os.path.join(self.models_dir, 'clustering'),
            profile_dir=os.path.join(self.models_dir, 'profiles'),
            deduction_model_dir=os.path.join(self.models_dir, 'deduction'),
            data_dir=self.data_dir,
            output_dir=self.results_dir,
            version="test"
        )
        
        self.assertEqual(pipeline.version, "test")
        self.assertEqual(pipeline.data_dir, self.data_dir)
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, 'clustering')))
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, 'profiles')))
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, 'deduction')))
    
    def test_pipeline_training(self):
        """파이프라인 학습 테스트"""
        pipeline = AIPipeline(
            cluster_model_dir=os.path.join(self.models_dir, 'clustering'),
            profile_dir=os.path.join(self.models_dir, 'profiles'),
            deduction_model_dir=os.path.join(self.models_dir, 'deduction'),
            data_dir=self.data_dir,
            output_dir=self.results_dir,
            version="test"
        )
        
        # 훈련 실행
        result = pipeline.train(
            data_file=self.training_file,
            num_clusters=3,
            force_retrain=True
        )
        
        # 결과 검증
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['version'], 'test')
        self.assertEqual(result['num_samples'], len(self.test_addresses))
        self.assertEqual(result['clustering']['num_clusters'], 3)
        
        # 모델 파일 확인
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, 'clustering', 'cluster_models_test.pkl')))
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, 'profiles', 'cluster_profiles_test.json')))
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, 'deduction', 'deduction_model_test.pkl')))
    
    def test_address_analysis(self):
        """주소 분석 테스트"""
        # 먼저 모델 훈련
        pipeline = AIPipeline(
            cluster_model_dir=os.path.join(self.models_dir, 'clustering'),
            profile_dir=os.path.join(self.models_dir, 'profiles'),
            deduction_model_dir=os.path.join(self.models_dir, 'deduction'),
            data_dir=self.data_dir,
            output_dir=self.results_dir,
            version="test"
        )
        
        pipeline.train(
            data_file=self.training_file,
            num_clusters=3,
            force_retrain=True
        )
        
        # 통합 서비스 패치
        with patch('ai_pipeline.service.BlockchainDataCollector') as mock_collector, \
             patch('ai_pipeline.service.DataStorage') as mock_storage:
             
            # 모의 수집기 설정
            mock_collector_instance = MagicMock()
            mock_collector_instance.collect_wallet_data.side_effect = \
                lambda addr: {
                    "status": "success", 
                    "data": MockBlockchainData.get_mock_wallet_data(addr)
                }
            mock_collector.return_value = mock_collector_instance
            
            # 모의 스토리지 설정
            mock_storage_instance = MagicMock()
            mock_storage_instance.load_data.return_value = None  # 기존 데이터 없음
            mock_storage_instance.load_processed_data.return_value = None
            mock_storage.return_value = mock_storage_instance
            
            # 서비스 초기화
            service = IntegratedAnalysisService(
                cluster_model_dir=os.path.join(self.models_dir, 'clustering'),
                profile_dir=os.path.join(self.models_dir, 'profiles'),
                deduction_model_dir=os.path.join(self.models_dir, 'deduction'),
                output_dir=self.results_dir,
                version="test"
            )
            
            # 분석 실행
            test_address = self.test_addresses[0]
            result = service.analyze_address(test_address, force_refresh=True)
            
            # 결과 검증
            self.assertIn('address', result)
            self.assertEqual(result['address'], test_address)
            self.assertIn('clustering', result)
            self.assertIn('data_summary', result)
            
            # 모의 객체 호출 확인
            mock_collector_instance.collect_wallet_data.assert_called_with(test_address)
    
    def test_batch_analysis(self):
        """배치 분석 테스트"""
        # 먼저 모델 훈련
        pipeline = AIPipeline(
            cluster_model_dir=os.path.join(self.models_dir, 'clustering'),
            profile_dir=os.path.join(self.models_dir, 'profiles'),
            deduction_model_dir=os.path.join(self.models_dir, 'deduction'),
            data_dir=self.data_dir,
            output_dir=self.results_dir,
            version="test"
        )
        
        pipeline.train(
            data_file=self.training_file,
            num_clusters=3,
            force_retrain=True
        )
        
        # 모의 분석 함수 생성
        def mock_analyze_address(address, force_refresh=False):
            return {
                'address': address,
                'clustering': {
                    'cluster': hash(address) % 3,
                    'primary_traits': ['trait1', 'trait2'],
                },
                'classification': {
                    'user_type': 'human' if hash(address) % 2 == 0 else 'bot',
                    'confidence': 0.8 + (hash(address) % 20) / 100
                },
                'data_summary': {
                    'transaction_count': 10 + hash(address) % 10,
                    'eth_balance': hash(address) % 5
                },
                'analysis_time': 0.5
            }
        
        # IntegratedAnalysisService.analyze_address 패치
        with patch('ai_pipeline.service.IntegratedAnalysisService.analyze_address', 
                   side_effect=mock_analyze_address):
            
            # 배치 파일 생성
            batch_file = os.path.join(self.data_dir, 'batch_addresses.txt')
            with open(batch_file, 'w') as f:
                for addr in self.test_addresses:
                    f.write(f"{addr}\n")
            
            # CLI 인자 모의
            args = MagicMock()
            args.addresses = batch_file
            args.output = self.results_dir
            args.force = False
            args.api_key = None
            
            # 배치 실행 모듈
            from ai_pipeline.main import batch_analyze
            
            # 실행
            batch_analyze(args)
            
            # 결과 파일 확인
            summary_file = os.path.join(self.results_dir, 'batch_summary.json')
            self.assertTrue(os.path.exists(summary_file))
            
            # 결과 내용 확인
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                
            self.assertEqual(len(summary), len(self.test_addresses))
            for item in summary:
                self.assertEqual(item['status'], 'success')
                self.assertIn(item['address'], self.test_addresses)

if __name__ == '__main__':
    unittest.main() 
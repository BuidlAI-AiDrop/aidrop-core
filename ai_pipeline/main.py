#!/usr/bin/env python3
"""
AI 파이프라인 - 메인 실행 파일
클러스터링과 분류 분석을 통합한 파이프라인
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional
import datetime
import logging

# 파이프라인 및 서비스 클래스 임포트
from .pipeline import AIPipeline
from .service import IntegratedAnalysisService
from .utils import setup_logger

# CLI 로깅 설정
logger = setup_logger('ai_pipeline')

def parse_args():
    """
    명령줄 인수 파싱
    """
    parser = argparse.ArgumentParser(description='AI 파이프라인 - 통합 블록체인 분석 도구')
    
    # 명령 서브파서 생성
    subparsers = parser.add_subparsers(dest='command', help='실행할 명령')
    
    # 학습 명령
    train_parser = subparsers.add_parser('train', help='AI 모델 학습')
    train_parser.add_argument('--data', type=str, required=True, 
                             help='학습 데이터 경로 (CSV 또는 JSON)')
    train_parser.add_argument('--output', type=str, default='models',
                             help='모델 저장 디렉토리')
    train_parser.add_argument('--clusters', type=int, default=5,
                             help='클러스터 수')
    train_parser.add_argument('--version', type=str, default=None,
                             help='모델 버전 (기본값: 현재 날짜)')
    train_parser.add_argument('--force', action='store_true',
                             help='기존 모델 덮어쓰기')
    train_parser.add_argument('--api-key', type=str, default=None,
                             help='Etherscan API 키 (옵션)')
    
    # 주소 분석 명령
    analyze_parser = subparsers.add_parser('analyze', help='주소 분석')
    analyze_parser.add_argument('--address', type=str, required=True,
                               help='분석할 이더리움 주소')
    analyze_parser.add_argument('--output', type=str, default='results',
                              help='결과 저장 디렉토리')
    analyze_parser.add_argument('--force', action='store_true',
                              help='기존 결과 덮어쓰기')
    analyze_parser.add_argument('--api-key', type=str, default=None,
                               help='Etherscan API 키 (옵션)')
    
    # 데이터 수집 명령
    collect_parser = subparsers.add_parser('collect', help='블록체인 데이터 수집')
    collect_parser.add_argument('--addresses', type=str, required=True,
                              help='주소 목록 파일 (한 줄에 하나씩)')
    collect_parser.add_argument('--output', type=str, default='data',
                              help='데이터 저장 디렉토리')
    collect_parser.add_argument('--api-key', type=str, required=True,
                              help='Etherscan API 키')
    
    # 특성 내보내기 명령
    export_parser = subparsers.add_parser('export', help='특성 데이터 내보내기')
    export_parser.add_argument('--addresses', type=str, required=True,
                             help='주소 목록 파일 (한 줄에 하나씩)')
    export_parser.add_argument('--output', type=str, default='features.csv',
                             help='특성 저장 파일 (CSV)')
    export_parser.add_argument('--api-key', type=str, default=None,
                             help='Etherscan API 키 (옵션)')
    
    # 배치 분석 명령
    batch_parser = subparsers.add_parser('batch', help='배치 주소 분석')
    batch_parser.add_argument('--addresses', type=str, required=True,
                             help='주소 목록 파일 (한 줄에 하나씩)')
    batch_parser.add_argument('--output', type=str, default='batch_results',
                             help='결과 저장 디렉토리')
    batch_parser.add_argument('--force', action='store_true',
                             help='기존 결과 덮어쓰기')
    batch_parser.add_argument('--api-key', type=str, default=None,
                             help='Etherscan API 키 (옵션)')
    
    return parser.parse_args()

def load_addresses(file_path: str) -> List[str]:
    """
    주소 목록 파일 로드
    
    Args:
        file_path: 주소 목록 파일 경로
    
    Returns:
        주소 목록
    """
    try:
        with open(file_path, 'r') as f:
            addresses = [line.strip() for line in f if line.strip()]
        return addresses
    except Exception as e:
        logger.error(f"주소 목록 로딩 실패: {str(e)}")
        sys.exit(1)

def train_models(args):
    """
    AI 모델 학습
    
    Args:
        args: 명령줄 인수
    """
    logger.info("AI 모델 학습 시작")
    
    # 버전 설정
    version = args.version or datetime.datetime.now().strftime("%Y%m%d")
    
    # 파이프라인 초기화
    pipeline = AIPipeline(
        cluster_model_dir=os.path.join(args.output, 'clustering'),
        profile_dir=os.path.join(args.output, 'profiles'),
        deduction_model_dir=os.path.join(args.output, 'deduction'),
        version=version,
        api_key=args.api_key
    )
    
    # 파이프라인 학습
    result = pipeline.train(
        data_file=args.data,
        num_clusters=args.clusters,
        force_retrain=args.force
    )
    
    if result.get('status') == 'success':
        logger.info(f"학습 성공: {result.get('message')}")
        logger.info(f"모델 저장 위치: {args.output}")
        
        # 클러스터 프로필 정보 출력
        cluster_profiles = result.get('cluster_profiles', {})
        for cluster, profile in cluster_profiles.items():
            logger.info(f"클러스터 {cluster}:")
            logger.info(f"  크기: {profile.get('size', 0)} 주소")
            logger.info(f"  주요 특성: {', '.join(profile.get('top_traits', []))}")
    else:
        logger.error(f"학습 실패: {result.get('error')}")
        sys.exit(1)

def analyze_address(args):
    """
    주소 분석
    
    Args:
        args: 명령줄 인수
    """
    logger.info(f"주소 분석 시작: {args.address}")
    
    # 서비스 초기화
    service = IntegratedAnalysisService(
        output_dir=args.output,
        api_key=args.api_key
    )
    
    # 주소 분석
    result = service.analyze_address(
        address=args.address,
        force_refresh=args.force
    )
    
    if 'error' in result:
        logger.error(f"분석 실패: {result['error']}")
        sys.exit(1)
    
    # 결과 출력
    logger.info(f"분석 완료: {args.address}")
    logger.info(f"클러스터: {result['clustering']['cluster']}")
    logger.info(f"주요 특성: {', '.join(result['clustering']['primary_traits'])}")
    
    if 'classification' in result:
        logger.info(f"분류 결과: {result['classification'].get('user_type', 'unknown')}")
        logger.info(f"신뢰도: {result['classification'].get('confidence', 0):.2f}")
    
    logger.info(f"상세 결과: {args.output}/{args.address}_analysis.json")

def collect_data(args):
    """
    블록체인 데이터 수집
    
    Args:
        args: 명령줄 인수
    """
    logger.info("블록체인 데이터 수집 시작")
    
    # 주소 목록 로드
    addresses = load_addresses(args.addresses)
    logger.info(f"수집할 주소 {len(addresses)}개 로드됨")
    
    # 파이프라인 초기화 (데이터 수집용)
    pipeline = AIPipeline(
        data_dir=args.output,
        api_key=args.api_key
    )
    
    # 데이터 수집
    results = pipeline.collect_data(addresses)
    
    # 결과 출력
    success_count = sum(1 for r in results if r.get('status') == 'success')
    logger.info(f"데이터 수집 완료: {success_count}/{len(addresses)} 성공")
    
    # 실패 목록 출력
    failed = [r.get('address') for r in results if r.get('status') != 'success']
    if failed:
        failed_file = os.path.join(args.output, 'failed_addresses.txt')
        with open(failed_file, 'w') as f:
            for addr in failed:
                f.write(f"{addr}\n")
        logger.info(f"실패한 주소 목록: {failed_file}")

def export_features(args):
    """
    특성 데이터 내보내기
    
    Args:
        args: 명령줄 인수
    """
    logger.info("특성 데이터 내보내기 시작")
    
    # 주소 목록 로드
    addresses = load_addresses(args.addresses)
    logger.info(f"처리할 주소 {len(addresses)}개 로드됨")
    
    # 파이프라인 초기화
    pipeline = AIPipeline(api_key=args.api_key)
    
    # 특성 내보내기
    result = pipeline.export_features(addresses, args.output)
    
    if result.get('status') == 'success':
        logger.info(f"특성 내보내기 완료: {result.get('output_file')}")
        logger.info(f"처리된 주소: {result.get('processed_count')}/{len(addresses)}")
    else:
        logger.error(f"특성 내보내기 실패: {result.get('error')}")
        sys.exit(1)

def batch_analyze(args):
    """
    배치 주소 분석
    
    Args:
        args: 명령줄 인수
    """
    logger.info("배치 주소 분석 시작")
    
    # 주소 목록 로드
    addresses = load_addresses(args.addresses)
    logger.info(f"분석할 주소 {len(addresses)}개 로드됨")
    
    # 서비스 초기화
    service = IntegratedAnalysisService(
        output_dir=args.output,
        api_key=args.api_key
    )
    
    # 결과 저장 디렉토리
    os.makedirs(args.output, exist_ok=True)
    
    # 결과 요약 저장용
    summary = []
    
    # 각 주소 분석
    for i, address in enumerate(addresses):
        logger.info(f"[{i+1}/{len(addresses)}] 주소 분석 중: {address}")
        
        try:
            result = service.analyze_address(
                address=address,
                force_refresh=args.force
            )
            
            if 'error' in result:
                logger.warning(f"분석 실패: {address} - {result['error']}")
                summary.append({
                    'address': address,
                    'status': 'failed',
                    'error': result['error']
                })
                continue
            
            # 결과 요약 저장
            summary.append({
                'address': address,
                'status': 'success',
                'cluster': result['clustering']['cluster'],
                'primary_traits': result['clustering'].get('primary_traits', []),
                'user_type': result['classification'].get('user_type', 'unknown') if 'classification' in result else 'unknown',
                'confidence': result['classification'].get('confidence', 0) if 'classification' in result else 0
            })
            
            logger.info(f"분석 완료: {address}")
            
        except Exception as e:
            logger.error(f"예외 발생: {address} - {str(e)}")
            summary.append({
                'address': address,
                'status': 'error',
                'error': str(e)
            })
    
    # 결과 요약 저장
    summary_file = os.path.join(args.output, 'batch_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 성공/실패 통계
    success_count = sum(1 for s in summary if s['status'] == 'success')
    logger.info(f"배치 분석 완료: {success_count}/{len(addresses)} 성공")
    logger.info(f"결과 요약: {summary_file}")

def main():
    """
    메인 함수
    """
    args = parse_args()
    
    if args.command == 'train':
        train_models(args)
    elif args.command == 'analyze':
        analyze_address(args)
    elif args.command == 'collect':
        collect_data(args)
    elif args.command == 'export':
        export_features(args)
    elif args.command == 'batch':
        batch_analyze(args)
    else:
        logger.error("유효한 명령을 지정해주세요.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
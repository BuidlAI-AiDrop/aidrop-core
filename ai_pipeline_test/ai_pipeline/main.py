"""
AI 파이프라인 CLI 모듈
"""

import os
import sys
import json
import argparse
from datetime import datetime

from .service import IntegratedAnalysisService
from .utils import setup_logger

logger = setup_logger('main')

def batch_analyze(args):
    """배치 분석 실행"""
    # 주소 목록 로드
    addresses = []
    if os.path.exists(args.addresses):
        with open(args.addresses, 'r') as f:
            addresses = [line.strip() for line in f if line.strip()]
    else:
        addresses = [args.addresses]
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 서비스 초기화
    service = IntegratedAnalysisService(
        cluster_model_dir=os.path.join(args.output, 'models', 'clustering'),
        profile_dir=os.path.join(args.output, 'models', 'profiles'),
        deduction_model_dir=os.path.join(args.output, 'models', 'deduction'),
        output_dir=args.output
    )
    
    # 각 주소 분석
    results = []
    for address in addresses:
        logger.info(f"분석 중: {address}")
        try:
            result = service.analyze_address(address, force_refresh=args.force)
            
            # 개별 결과 저장
            result_file = os.path.join(args.output, f"{address.lower()}_analysis.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            results.append({
                "address": address,
                "status": "success",
                "file": result_file
            })
            
        except Exception as e:
            logger.error(f"주소 분석 오류: {address} - {str(e)}")
            results.append({
                "address": address,
                "status": "error",
                "message": str(e)
            })
    
    # 요약 저장
    summary_file = os.path.join(args.output, "batch_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"배치 분석 완료: {len(results)} 주소, 요약: {summary_file}")
    return summary_file

def parse_args():
    """인자 파싱"""
    parser = argparse.ArgumentParser(description='AI 파이프라인 CLI')
    subparsers = parser.add_subparsers(dest='command')
    
    # 배치 분석 명령
    batch_parser = subparsers.add_parser('batch', help='배치 주소 분석')
    batch_parser.add_argument('addresses', help='주소 파일 경로 또는 단일 주소')
    batch_parser.add_argument('--output', default='./output', help='출력 디렉토리')
    batch_parser.add_argument('--force', action='store_true', help='캐시된 데이터 무시')
    batch_parser.add_argument('--api-key', help='API 키')
    
    # 학습 명령
    train_parser = subparsers.add_parser('train', help='모델 학습')
    train_parser.add_argument('data', help='학습 데이터 파일')
    train_parser.add_argument('--output', default='./models', help='모델 출력 디렉토리')
    train_parser.add_argument('--clusters', type=int, default=5, help='클러스터 수')
    train_parser.add_argument('--force', action='store_true', help='기존 모델 덮어쓰기')
    train_parser.add_argument('--version', default=f"v{datetime.now().strftime('%Y%m%d')}", help='모델 버전')
    
    return parser.parse_args()

def main():
    """CLI 진입점"""
    args = parse_args()
    
    if args.command == 'batch':
        batch_analyze(args)
    elif args.command == 'train':
        from .pipeline import AIPipeline
        
        pipeline = AIPipeline(
            cluster_model_dir=os.path.join(args.output, 'clustering'),
            profile_dir=os.path.join(args.output, 'profiles'),
            deduction_model_dir=os.path.join(args.output, 'deduction'),
            data_dir=os.path.dirname(args.data),
            output_dir=args.output,
            version=args.version
        )
        
        result = pipeline.train(
            data_file=args.data,
            num_clusters=args.clusters,
            force_retrain=args.force
        )
        
        logger.info(f"학습 결과: {result}")
    else:
        logger.error("알 수 없는 명령입니다. 'batch' 또는 'train'을 사용하세요.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
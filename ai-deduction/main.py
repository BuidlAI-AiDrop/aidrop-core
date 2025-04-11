"""
유저 분류 모델 추론 서비스 실행 예제
"""

import sys
import json
from typing import Dict, Any
from .utils import setup_logger
from .inference_service import InferenceService

logger = setup_logger('main')

def analyze_address(address: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    지갑 주소 분석
    
    Args:
        address: 분석할 지갑 주소
        force_refresh: 기존 결과가 있어도 강제로 새로 분석
        
    Returns:
        분석 결과 (딕셔너리)
    """
    service = InferenceService()
    return service.analyze_address(address, force_refresh)

def print_help():
    """사용법 출력"""
    print("사용법: python -m ai-deduction.main [command] [options]")
    print("\n명령어:")
    print("  analyze <address>  - 지갑 주소 분석")
    print("  stats              - 전체 분석 통계 확인")
    print("\n옵션:")
    print("  --refresh          - 기존 결과가 있어도 새로 분석")
    print("  --json             - JSON 형식으로 출력")

def print_result(result: Dict[str, Any], json_output: bool = False):
    """결과 출력"""
    if json_output:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    
    if 'error' in result:
        print(f"오류: {result['error']}")
        return
    
    print(f"주소: {result['address']}")
    
    if 'classification' in result:
        classification = result['classification']
        print("\n[분류 결과]")
        if 'predicted_class' in classification:
            print(f"예측 클래스: {classification['predicted_class']}")
            print(f"신뢰도: {classification.get('prediction_confidence', 0):.2f}")
        
        if 'top_features' in classification:
            print("\n주요 특성:")
            for feature, importance in classification['top_features'].items():
                print(f"  - {feature}: {importance:.4f}")
    
    if 'clustering' in result:
        clustering = result['clustering']
        print("\n[클러스터링 결과]")
        if 'cluster' in clustering:
            print(f"클러스터: {clustering['cluster']}")
    
    if 'analysis_time' in result:
        print(f"\n분석 소요 시간: {result['analysis_time']:.2f}초")

def print_stats(service: InferenceService, json_output: bool = False):
    """통계 출력"""
    category_dist = service.get_category_distribution()
    cluster_dist = service.get_cluster_distribution()
    
    stats = {
        'category_distribution': category_dist,
        'cluster_distribution': cluster_dist
    }
    
    if json_output:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return
    
    print("[카테고리 분포]")
    for category, count in category_dist.items():
        print(f"  - {category}: {count}명")
    
    print("\n[클러스터 분포]")
    for cluster, count in cluster_dist.items():
        print(f"  - {cluster}: {count}명")

def main():
    """메인 실행 함수"""
    args = sys.argv[1:]
    
    if not args or args[0] in ['-h', '--help', 'help']:
        print_help()
        return
    
    command = args[0]
    json_output = '--json' in args
    force_refresh = '--refresh' in args
    
    if command == 'analyze':
        if len(args) < 2 or args[1].startswith('-'):
            print("오류: 분석할 주소를 지정해야 합니다.")
            print_help()
            return
        
        address = args[1]
        result = analyze_address(address, force_refresh)
        print_result(result, json_output)
    
    elif command == 'stats':
        service = InferenceService()
        print_stats(service, json_output)
    
    else:
        print(f"오류: 알 수 없는 명령어 '{command}'")
        print_help()

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
마케팅 타겟 추출 도구 - 명령줄 인터페이스
"""

import argparse
import pandas as pd
import os
import sys
from datetime import datetime
from target_extractor import MarketingTargetExtractor


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='마케팅 타겟 추출 도구')
    
    parser.add_argument('--type', '-t', choices=['defi-holders', 'nft-enthusiasts', 
                                               'community-users', 'aggressive-traders', 'custom'],
                      help='추출할 타겟 유형', required=True)
    
    parser.add_argument('--output', '-o', default='results/marketing_targets/targets.json',
                      help='결과를 저장할 파일 경로')
    
    parser.add_argument('--mbti', '-m', nargs='+', 
                      help='특정 MBTI 유형으로 필터링 (예: D-H-S D-H-A)')
    
    parser.add_argument('--cluster', '-c', type=int, nargs='+',
                      help='특정 클러스터 ID로 필터링 (예: 0 1 2)')
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    # 타임스탬프 기반 출력 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename, ext = os.path.splitext(args.output)
    if not ext:
        ext = '.json'  # 기본 확장자
    output_path = f"{filename}_{timestamp}{ext}"
    
    try:
        # 타겟 추출기 초기화
        extractor = MarketingTargetExtractor()
        
        # 타겟 유형에 따른 추출
        if args.type == 'defi-holders':
            targets = extractor.get_defi_long_term_holders()
            target_desc = "DeFi 장기 홀더"
        elif args.type == 'nft-enthusiasts':
            targets = extractor.get_nft_enthusiasts()
            target_desc = "NFT 열정 사용자"
        elif args.type == 'community-users':
            targets = extractor.get_community_involved_users()
            target_desc = "커뮤니티 참여 사용자"
        elif args.type == 'aggressive-traders':
            targets = extractor.get_aggressive_traders()
            target_desc = "공격적 트레이더"
        elif args.type == 'custom' and args.mbti:
            targets = extractor.filter_by_mbti(args.mbti)
            target_desc = f"커스텀 MBTI 필터: {', '.join(args.mbti)}"
        else:
            print("오류: 'custom' 타입을 사용할 경우 --mbti 인자가 필요합니다")
            sys.exit(1)
            
        # 클러스터 ID로 추가 필터링 (선택 사항)
        if args.cluster:
            targets = targets[targets['cluster'].isin(args.cluster)]
            target_desc += f" + 클러스터 필터: {args.cluster}"
        
        # 결과 출력
        if len(targets) == 0:
            print(f"조건에 맞는 타겟이 없습니다: {target_desc}")
            return
            
        print(f"추출된 타겟: {target_desc}")
        print(f"총 {len(targets)}명의 사용자가 추출되었습니다")
            
        # 결과 저장
        extractor.export_targets(targets, output_path)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
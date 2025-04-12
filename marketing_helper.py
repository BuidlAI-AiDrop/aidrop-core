#!/usr/bin/env python3
"""
마케팅 타겟팅 및 효과 추적 도구 통합 헬퍼

마케팅 타겟팅과 효과 추적 모듈을 쉽게 통합할 수 있도록 헬퍼 함수 제공
"""

import os
import json
import random
import string
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

from marketing_targeting.target_extractor import MarketingTargetExtractor
from marketing_tracking.tracker import MarketingTracker


def generate_campaign_id(prefix: str = 'campaign') -> str:
    """고유한 캠페인 ID 생성

    Args:
        prefix: 캠페인 ID 접두사

    Returns:
        고유한 캠페인 ID
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{timestamp}_{random_suffix}"


def create_defi_holders_campaign(output_dir: str = 'results/marketing_targets') -> str:
    """DeFi 장기 홀더 타겟팅 캠페인 생성

    Args:
        output_dir: 출력 디렉토리

    Returns:
        생성된 타겟 파일 경로
    """
    # 타겟 추출
    extractor = MarketingTargetExtractor()
    targets = extractor.get_defi_long_term_holders()
    
    # 타겟 파일 생성
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    target_file = os.path.join(output_dir, f"defi_holders_{timestamp}.json")
    extractor.export_targets(targets, target_file)
    
    return target_file


def create_nft_enthusiasts_campaign(output_dir: str = 'results/marketing_targets') -> str:
    """NFT 열정 사용자 타겟팅 캠페인 생성

    Args:
        output_dir: 출력 디렉토리

    Returns:
        생성된 타겟 파일 경로
    """
    # 타겟 추출
    extractor = MarketingTargetExtractor()
    targets = extractor.get_nft_enthusiasts()
    
    # 타겟 파일 생성
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    target_file = os.path.join(output_dir, f"nft_enthusiasts_{timestamp}.json")
    extractor.export_targets(targets, target_file)
    
    return target_file


def create_staking_campaign(mbti_types: List[str] = None, 
                           cluster_ids: List[int] = None, 
                           output_dir: str = 'results/marketing_targets') -> Tuple[str, str]:
    """스테이킹 캠페인 타겟팅 및 추적 설정

    Args:
        mbti_types: 타겟팅할 MBTI 유형 목록 (기본값: DeFi 장기 홀더)
        cluster_ids: 타겟팅할 클러스터 ID 목록
        output_dir: 출력 디렉토리

    Returns:
        타겟 파일 경로, 캠페인 ID
    """
    # 타겟 추출
    extractor = MarketingTargetExtractor()
    
    # 기본값: DeFi 장기 홀더
    if not mbti_types:
        targets = extractor.get_defi_long_term_holders()
        campaign_type = "defi_holders"
    else:
        targets = extractor.filter_by_mbti(mbti_types)
        campaign_type = "custom_mbti"
    
    # 클러스터 필터링 (선택 사항)
    if cluster_ids:
        targets = targets[targets['cluster'].isin(cluster_ids)]
        campaign_type += "_cluster_filtered"
    
    # 타겟 파일 생성
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    target_file = os.path.join(output_dir, f"{campaign_type}_{timestamp}.json")
    extractor.export_targets(targets, target_file)
    
    # 캠페인 생성
    campaign_id = generate_campaign_id("staking")
    tracker = MarketingTracker(campaign_name=campaign_id, target_file=target_file)
    
    return target_file, campaign_id


def initialize_tracking_for_targets(target_file: str, campaign_name: Optional[str] = None) -> str:
    """기존 타겟 파일에 대한 추적 시작

    Args:
        target_file: 타겟 파일 경로
        campaign_name: 캠페인 이름 (없으면 새로 생성)

    Returns:
        캠페인 ID
    """
    if not campaign_name:
        campaign_name = generate_campaign_id()
    
    tracker = MarketingTracker(campaign_name=campaign_name, target_file=target_file)
    return campaign_name


def get_campaign_status(campaign_id: str) -> Dict:
    """캠페인 상태 조회

    Args:
        campaign_id: 캠페인 ID

    Returns:
        캠페인 상태 정보
    """
    tracking_dir = "results/marketing_tracking"
    campaign_dir = os.path.join(tracking_dir, campaign_id)
    metadata_file = os.path.join(campaign_dir, "campaign_metadata.json")
    
    if not os.path.exists(metadata_file):
        return {
            "status": "not_found",
            "campaign_id": campaign_id,
            "error": "캠페인을 찾을 수 없습니다."
        }
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # 최신 보고서 확인
        report_files = [f for f in os.listdir(campaign_dir) if f.startswith('report_') and f.endswith('.json')]
        latest_report = None
        
        if report_files:
            latest_report_file = max(report_files, key=lambda f: os.path.getmtime(os.path.join(campaign_dir, f)))
            with open(os.path.join(campaign_dir, latest_report_file), 'r') as f:
                latest_report = json.load(f)
        
        return {
            "status": "active",
            "campaign_id": campaign_id,
            "created_at": metadata.get("created_at"),
            "target_count": metadata.get("target_count"),
            "snapshots_count": len(metadata.get("snapshots", [])),
            "metrics_count": len(metadata.get("metrics", [])),
            "latest_report": latest_report["report_id"] if latest_report else None,
            "success_rating": latest_report["conclusion"]["overall_success_rating"] if latest_report else None
        }
    except Exception as e:
        return {
            "status": "error",
            "campaign_id": campaign_id,
            "error": str(e)
        }


def get_campaigns_summary() -> List[Dict]:
    """모든 캠페인 요약 정보

    Returns:
        모든 캠페인 요약 정보 목록
    """
    tracking_dir = "results/marketing_tracking"
    
    if not os.path.exists(tracking_dir):
        return []
    
    campaigns = []
    
    for campaign_id in os.listdir(tracking_dir):
        campaign_dir = os.path.join(tracking_dir, campaign_id)
        if os.path.isdir(campaign_dir):
            status = get_campaign_status(campaign_id)
            if status["status"] != "error":
                campaigns.append({
                    "campaign_id": campaign_id,
                    "created_at": status.get("created_at"),
                    "target_count": status.get("target_count"),
                    "snapshots_count": status.get("snapshots_count"),
                    "latest_success_rating": status.get("success_rating")
                })
    
    # 생성 시간 기준 정렬
    campaigns.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    
    return campaigns


if __name__ == "__main__":
    # 간단한 CLI 인터페이스
    import argparse
    
    parser = argparse.ArgumentParser(description='마케팅 타겟팅 및 효과 추적 헬퍼')
    subparsers = parser.add_subparsers(dest='command', help='실행할 명령')
    
    # 타겟팅 명령
    target_parser = subparsers.add_parser('target', help='타겟 추출')
    target_parser.add_argument('--type', '-t', choices=['defi', 'nft', 'community'], required=True,
                             help='타겟 유형')
    
    # 캠페인 생성 명령
    campaign_parser = subparsers.add_parser('campaign', help='캠페인 생성')
    campaign_parser.add_argument('--target-file', '-t', required=True,
                               help='타겟 파일 경로')
    campaign_parser.add_argument('--name', '-n',
                               help='캠페인 이름 (없으면 자동 생성)')
    
    # 캠페인 조회 명령
    list_parser = subparsers.add_parser('list', help='캠페인 목록 조회')
    
    # 캠페인 상태 조회 명령
    status_parser = subparsers.add_parser('status', help='캠페인 상태 조회')
    status_parser.add_argument('--campaign-id', '-c', required=True,
                             help='캠페인 ID')
    
    args = parser.parse_args()
    
    if args.command == 'target':
        if args.type == 'defi':
            target_file = create_defi_holders_campaign()
            print(f"DeFi 장기 홀더 타겟 파일 생성 완료: {target_file}")
        elif args.type == 'nft':
            target_file = create_nft_enthusiasts_campaign()
            print(f"NFT 열정 사용자 타겟 파일 생성 완료: {target_file}")
        elif args.type == 'community':
            extractor = MarketingTargetExtractor()
            targets = extractor.get_community_involved_users()
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            target_file = f"results/marketing_targets/community_users_{timestamp}.json"
            extractor.export_targets(targets, target_file)
            print(f"커뮤니티 참여 사용자 타겟 파일 생성 완료: {target_file}")
    
    elif args.command == 'campaign':
        campaign_id = initialize_tracking_for_targets(args.target_file, args.name)
        print(f"캠페인 생성 완료: {campaign_id}")
    
    elif args.command == 'list':
        campaigns = get_campaigns_summary()
        if campaigns:
            print(f"총 {len(campaigns)}개의 캠페인이 있습니다:")
            for i, c in enumerate(campaigns, 1):
                print(f"{i}. {c['campaign_id']} - 타겟: {c['target_count']}명, 스냅샷: {c['snapshots_count']}개")
        else:
            print("등록된 캠페인이 없습니다.")
    
    elif args.command == 'status':
        status = get_campaign_status(args.campaign_id)
        if status["status"] == "active":
            print(f"캠페인 ID: {status['campaign_id']}")
            print(f"생성 시간: {status['created_at']}")
            print(f"타겟 수: {status['target_count']}명")
            print(f"스냅샷 수: {status['snapshots_count']}개")
            print(f"지표 수: {status['metrics_count']}개")
            if status.get("latest_report"):
                print(f"최신 보고서: {status['latest_report']}")
                print(f"성공 점수: {status['success_rating']:.2f}%")
        else:
            print(f"오류: {status.get('error', '알 수 없는 오류')}") 
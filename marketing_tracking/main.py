#!/usr/bin/env python3
"""
마케팅 효과 추적 도구 - 명령줄 인터페이스
"""

import argparse
import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from tracker import MarketingTracker


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='마케팅 효과 추적 도구')
    
    # 서브파서 생성
    subparsers = parser.add_subparsers(dest='command', help='실행할 명령')
    
    # 새 캠페인 생성 명령
    create_parser = subparsers.add_parser('create', help='새 마케팅 캠페인 생성')
    create_parser.add_argument('--name', '-n', required=True,
                             help='캠페인 이름 (고유 식별자)')
    create_parser.add_argument('--target-file', '-t', required=True,
                             help='타겟 사용자 목록 파일 경로')
    
    # 스냅샷 생성 명령
    snapshot_parser = subparsers.add_parser('snapshot', help='현재 상태 스냅샷 생성')
    snapshot_parser.add_argument('--campaign', '-c', required=True,
                               help='캠페인 이름')
    snapshot_parser.add_argument('--name', '-n', default=f"snapshot_{datetime.now().strftime('%Y%m%d')}",
                               help='스냅샷 이름')
    snapshot_parser.add_argument('--data-file', '-d', required=True,
                               help='블록체인 데이터 파일 경로 (JSON)')
    
    # 지표 계산 명령
    metrics_parser = subparsers.add_parser('metrics', help='두 스냅샷 간 지표 계산')
    metrics_parser.add_argument('--campaign', '-c', required=True,
                              help='캠페인 이름')
    metrics_parser.add_argument('--before', '-b', required=True,
                              help='이전 스냅샷 ID')
    metrics_parser.add_argument('--after', '-a', required=True,
                              help='이후 스냅샷 ID')
    
    # 보고서 생성 명령
    report_parser = subparsers.add_parser('report', help='성과 보고서 생성')
    report_parser.add_argument('--campaign', '-c', required=True,
                             help='캠페인 이름')
    report_parser.add_argument('--metrics-id', '-m', required=True,
                             help='지표 ID')
    report_parser.add_argument('--format', '-f', choices=['json', 'html'], default='html',
                             help='보고서 형식')
    
    # 모니터링 실행 명령
    monitor_parser = subparsers.add_parser('monitor', help='지속적인 모니터링 실행')
    monitor_parser.add_argument('--campaign', '-c', required=True,
                              help='캠페인 이름')
    monitor_parser.add_argument('--interval', '-i', type=float, default=24.0,
                              help='모니터링 간격 (시간)')
    monitor_parser.add_argument('--data-source', '-d', required=True,
                              help='블록체인 데이터 소스 (함수 또는 디렉토리)')
    
    return parser.parse_args()


def load_blockchain_data(data_file: str, target_addresses: List[str] = None) -> Dict:
    """블록체인 데이터 로드

    Args:
        data_file: 데이터 파일 경로
        target_addresses: 필터링할 주소 목록 (None이면 모든 주소)
    
    Returns:
        블록체인 데이터 (주소 -> 데이터 매핑)
    """
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # 주소 필터링 (필요시)
        if target_addresses:
            filtered_data = {addr: data[addr] for addr in target_addresses if addr in data}
            return filtered_data
        
        return data
    except Exception as e:
        print(f"블록체인 데이터 로드 오류: {e}")
        sys.exit(1)


def simulate_blockchain_data(addresses: List[str]) -> Dict:
    """실제 블록체인 데이터 대신 시뮬레이션 데이터 생성 (개발/테스트용)

    Args:
        addresses: 주소 목록
    
    Returns:
        시뮬레이션된 블록체인 데이터 (주소 -> 데이터 매핑)
    """
    import random
    
    data = {}
    timestamp = int(time.time())
    
    for addr in addresses:
        # 이전 데이터가 있는지 확인 (이전 스냅샷 파일 확인)
        prev_data = None
        # TODO: 실제 구현시 이전 스냅샷 파일에서 데이터를 로드하는 로직 추가
        
        # 새 데이터 생성
        txn_count = random.randint(10, 100) if not prev_data else prev_data.get("txn_count", 0) + random.randint(0, 5)
        token_count = random.randint(1, 10) if not prev_data else prev_data.get("token_count", 0) + random.randint(-1, 2)
        
        # 일정 확률로 스테이킹/서비스 상태 변경
        is_staking = random.random() < 0.1 if not prev_data else prev_data.get("is_staking", False) or random.random() < 0.05
        service_used = random.random() < 0.08 if not prev_data else prev_data.get("service_used", False) or random.random() < 0.03
        
        data[addr] = {
            "timestamp": timestamp,
            "txn_count": txn_count,
            "token_count": max(0, token_count),
            "is_staking": is_staking,
            "service_used": service_used,
            "gas_used": random.randint(100000, 10000000),
            "balance": random.uniform(0.1, 10.0),
            # 추가 데이터 필드 (필요시)
        }
    
    return data


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    if not args.command:
        print("오류: 명령이 지정되지 않았습니다. 'create', 'snapshot', 'metrics', 'report', 'monitor' 중 하나를 사용하세요.")
        sys.exit(1)
    
    tracking_dir = "results/marketing_tracking"
    os.makedirs(tracking_dir, exist_ok=True)
    
    try:
        if args.command == 'create':
            # 캠페인 생성
            print(f"캠페인 생성 중: {args.name}")
            tracker = MarketingTracker(
                campaign_name=args.name,
                target_file=args.target_file,
                tracking_dir=tracking_dir
            )
            print(f"캠페인이 생성되었습니다: {tracker.tracking_dir}")
            
        elif args.command == 'snapshot':
            # 스냅샷 생성
            print(f"'{args.campaign}' 캠페인의 스냅샷 생성 중: {args.name}")
            tracker = MarketingTracker(
                campaign_name=args.campaign,
                target_file=os.path.join(tracking_dir, args.campaign, "campaign_metadata.json"),
                tracking_dir=tracking_dir
            )
            
            # 블록체인 데이터 로드
            blockchain_data = load_blockchain_data(args.data_file, tracker.target_addresses)
            
            # 스냅샷 생성
            tracker.create_snapshot(args.name, blockchain_data)
            
        elif args.command == 'metrics':
            # 지표 계산
            print(f"'{args.campaign}' 캠페인의 지표 계산 중: {args.before} -> {args.after}")
            tracker = MarketingTracker(
                campaign_name=args.campaign,
                target_file=os.path.join(tracking_dir, args.campaign, "campaign_metadata.json"),
                tracking_dir=tracking_dir
            )
            
            # 지표 계산
            metrics = tracker.calculate_metrics(args.before, args.after)
            print(f"지표 계산 완료: {metrics['id']}")
            
        elif args.command == 'report':
            # 보고서 생성
            print(f"'{args.campaign}' 캠페인의 보고서 생성 중: {args.metrics_id}")
            tracker = MarketingTracker(
                campaign_name=args.campaign,
                target_file=os.path.join(tracking_dir, args.campaign, "campaign_metadata.json"),
                tracking_dir=tracking_dir
            )
            
            # 보고서 생성
            report_file = tracker.generate_report(args.metrics_id, args.format)
            print(f"보고서 생성 완료: {report_file}")
            
        elif args.command == 'monitor':
            # 지속적인 모니터링
            print(f"'{args.campaign}' 캠페인 모니터링 시작 (간격: {args.interval}시간)")
            tracker = MarketingTracker(
                campaign_name=args.campaign,
                target_file=os.path.join(tracking_dir, args.campaign, "campaign_metadata.json"),
                tracking_dir=tracking_dir
            )
            
            # 데이터 소스가 디렉토리인 경우
            if os.path.isdir(args.data_source):
                # 경로에서 데이터 가져오는 함수 정의
                def fetch_data_from_dir(addresses):
                    # 가장 최근 파일 찾기
                    files = [f for f in os.listdir(args.data_source) if f.endswith('.json')]
                    if not files:
                        print(f"경고: {args.data_source}에 데이터 파일이 없습니다. 시뮬레이션 데이터 사용")
                        return simulate_blockchain_data(addresses)
                    
                    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(args.data_source, f)))
                    return load_blockchain_data(os.path.join(args.data_source, latest_file), addresses)
                
                # 모니터링 시작
                tracker.monitor_continuously(fetch_data_from_dir, args.interval)
            else:
                # 시뮬레이션 데이터 사용
                print("실시간 블록체인 데이터 소스가 제공되지 않았습니다. 시뮬레이션 데이터 사용")
                tracker.monitor_continuously(simulate_blockchain_data, args.interval)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
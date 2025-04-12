import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any, Optional, Tuple, Set


class MarketingTracker:
    """마케팅 효과 추적 클래스

    타겟팅된 사용자의 온체인 활동을 지속적으로 모니터링하고 성과 지표 생성
    """

    def __init__(self, campaign_name: str, target_file: str, 
                 tracking_dir: str = "results/marketing_tracking"):
        """초기화

        Args:
            campaign_name: 마케팅 캠페인 이름 (고유 식별자로 사용)
            target_file: 타겟 사용자 목록 파일 경로
            tracking_dir: 추적 결과 저장 디렉토리
        """
        self.campaign_name = campaign_name
        self.tracking_dir = os.path.join(tracking_dir, campaign_name)
        self.metrics_dir = os.path.join(self.tracking_dir, "metrics")
        self.snapshots_dir = os.path.join(self.tracking_dir, "snapshots")
        
        # 필요한 디렉토리 생성
        os.makedirs(self.tracking_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # 타겟 사용자 로드
        self.targets = self._load_targets(target_file)
        self.target_addresses = self.targets['address'].tolist()
        
        # 캠페인 메타데이터 초기화 및 저장
        self.campaign_metadata = {
            "name": campaign_name,
            "created_at": datetime.now().isoformat(),
            "target_count": len(self.targets),
            "snapshots": [],
            "metrics": []
        }
        self._save_campaign_metadata()
    
    def _load_targets(self, target_file: str) -> pd.DataFrame:
        """타겟 사용자 목록 로드

        Args:
            target_file: 타겟 파일 경로
        
        Returns:
            타겟 사용자 DataFrame
        """
        if target_file.endswith('.csv'):
            return pd.read_csv(target_file)
        elif target_file.endswith('.json'):
            return pd.read_json(target_file)
        else:
            raise ValueError(f"지원되지 않는 파일 형식: {target_file}")
    
    def _save_campaign_metadata(self) -> None:
        """캠페인 메타데이터 저장"""
        metadata_file = os.path.join(self.tracking_dir, "campaign_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.campaign_metadata, f, indent=2, ensure_ascii=False)
    
    def create_snapshot(self, snapshot_name: str, blockchain_data: Dict) -> None:
        """타겟 사용자의 현재 상태 스냅샷 생성

        Args:
            snapshot_name: 스냅샷 이름
            blockchain_data: 가장 최근의 블록체인 데이터 (주소별 최신 상태)
        """
        timestamp = datetime.now().isoformat()
        snapshot_id = f"{snapshot_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 스냅샷 데이터 구성
        snapshot = {
            "id": snapshot_id,
            "name": snapshot_name,
            "timestamp": timestamp,
            "target_addresses": self.target_addresses,
            "data": {}
        }
        
        # 각 타겟 주소에 대한 데이터 추출
        for address in self.target_addresses:
            if address in blockchain_data:
                snapshot["data"][address] = blockchain_data[address]
        
        # 스냅샷 저장
        snapshot_file = os.path.join(self.snapshots_dir, f"{snapshot_id}.json")
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        
        # 캠페인 메타데이터 업데이트
        self.campaign_metadata["snapshots"].append({
            "id": snapshot_id,
            "name": snapshot_name,
            "timestamp": timestamp,
            "file": snapshot_file
        })
        self._save_campaign_metadata()
        
        print(f"스냅샷 생성 완료: {snapshot_name} ({len(snapshot['data'])} 주소)")
    
    def calculate_metrics(self, before_snapshot_id: str, after_snapshot_id: str) -> Dict:
        """두 스냅샷 간의 성과 지표 계산

        Args:
            before_snapshot_id: 이전 스냅샷 ID
            after_snapshot_id: 이후 스냅샷 ID
        
        Returns:
            계산된 성과 지표
        """
        # 스냅샷 로드
        before_file = os.path.join(self.snapshots_dir, f"{before_snapshot_id}.json")
        after_file = os.path.join(self.snapshots_dir, f"{after_snapshot_id}.json")
        
        if not os.path.exists(before_file) or not os.path.exists(after_file):
            raise FileNotFoundError(f"스냅샷 파일이 존재하지 않습니다: {before_file} 또는 {after_file}")
        
        with open(before_file, 'r') as f:
            before_data = json.load(f)
        
        with open(after_file, 'r') as f:
            after_data = json.load(f)
        
        # 지표 계산
        metrics = {
            "id": f"metrics_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "before_snapshot": before_snapshot_id,
            "after_snapshot": after_snapshot_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "transaction_count_change": {},
                "token_holdings_change": {},
                "staking_participation": {},
                "service_adoption": {}
            },
            "summary": {}
        }
        
        # 각 주소별 변화 계산
        for address in self.target_addresses:
            before = before_data["data"].get(address, {})
            after = after_data["data"].get(address, {})
            
            if before and after:
                # 각 지표 업데이트 (구현 필요: 실제 데이터 구조에 맞게)
                # 여기서는 예시로 가상 데이터 구조 사용
                metrics["metrics"]["transaction_count_change"][address] = {
                    "before": before.get("txn_count", 0),
                    "after": after.get("txn_count", 0),
                    "change": after.get("txn_count", 0) - before.get("txn_count", 0)
                }
                
                metrics["metrics"]["token_holdings_change"][address] = {
                    "before": before.get("token_count", 0),
                    "after": after.get("token_count", 0),
                    "change": after.get("token_count", 0) - before.get("token_count", 0)
                }
                
                # 스테이킹 참여 여부 (예시)
                metrics["metrics"]["staking_participation"][address] = {
                    "before": before.get("is_staking", False),
                    "after": after.get("is_staking", False),
                    "new_participant": not before.get("is_staking", False) and after.get("is_staking", False)
                }
                
                # 서비스 채택 여부 (예시)
                metrics["metrics"]["service_adoption"][address] = {
                    "before": before.get("service_used", False),
                    "after": after.get("service_used", False),
                    "new_adoption": not before.get("service_used", False) and after.get("service_used", False)
                }
        
        # 전체 요약 통계 계산
        txn_changes = [m["change"] for m in metrics["metrics"]["transaction_count_change"].values()]
        token_changes = [m["change"] for m in metrics["metrics"]["token_holdings_change"].values()]
        new_stakers = sum(1 for m in metrics["metrics"]["staking_participation"].values() if m["new_participant"])
        new_adopters = sum(1 for m in metrics["metrics"]["service_adoption"].values() if m["new_adoption"])
        
        # 요약 정보 업데이트
        metrics["summary"] = {
            "total_addresses": len(self.target_addresses),
            "addresses_with_data": len([a for a in self.target_addresses if a in after_data["data"]]),
            "avg_txn_change": sum(txn_changes) / len(txn_changes) if txn_changes else 0,
            "avg_token_change": sum(token_changes) / len(token_changes) if token_changes else 0,
            "new_stakers": new_stakers,
            "new_stakers_pct": (new_stakers / len(self.target_addresses)) * 100 if len(self.target_addresses) > 0 else 0,
            "new_adopters": new_adopters,
            "new_adopters_pct": (new_adopters / len(self.target_addresses)) * 100 if len(self.target_addresses) > 0 else 0
        }
        
        # 지표 저장
        metrics_file = os.path.join(self.metrics_dir, f"{metrics['id']}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 캠페인 메타데이터 업데이트
        self.campaign_metadata["metrics"].append({
            "id": metrics["id"],
            "before_snapshot": before_snapshot_id,
            "after_snapshot": after_snapshot_id,
            "timestamp": metrics["timestamp"],
            "file": metrics_file
        })
        self._save_campaign_metadata()
        
        print(f"성과 지표 계산 완료: {before_snapshot_id} -> {after_snapshot_id}")
        return metrics
    
    def generate_report(self, metrics_id: str, output_format: str = 'json') -> str:
        """성과 지표 보고서 생성

        Args:
            metrics_id: 성과 지표 ID
            output_format: 출력 형식 ('json' 또는 'html')
        
        Returns:
            생성된 보고서 파일 경로
        """
        # 지표 데이터 로드
        metrics_file = os.path.join(self.metrics_dir, f"{metrics_id}.json")
        
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"지표 파일이 존재하지 않습니다: {metrics_file}")
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # 보고서 생성
        report = {
            "campaign_name": self.campaign_name,
            "report_id": f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "metrics_id": metrics_id,
            "before_snapshot": metrics["before_snapshot"],
            "after_snapshot": metrics["after_snapshot"],
            "summary": metrics["summary"],
            "visualizations": [],  # 생성된 시각화 파일 경로
            "conclusion": {
                "staking_adoption_rate": metrics["summary"]["new_stakers_pct"],
                "service_adoption_rate": metrics["summary"]["new_adopters_pct"],
                "overall_success_rating": self._calculate_success_rating(metrics["summary"])
            }
        }
        
        # 시각화 생성
        visualizations = self._generate_visualizations(metrics)
        report["visualizations"] = visualizations
        
        # 보고서 저장
        if output_format == 'json':
            report_file = os.path.join(self.tracking_dir, f"{report['report_id']}.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        elif output_format == 'html':
            report_file = os.path.join(self.tracking_dir, f"{report['report_id']}.html")
            self._generate_html_report(report, report_file)
        else:
            raise ValueError(f"지원되지 않는 출력 형식: {output_format}")
        
        print(f"보고서 생성 완료: {report_file}")
        return report_file
    
    def _calculate_success_rating(self, summary: Dict) -> float:
        """캠페인 성공 점수 계산 (0-100)

        Args:
            summary: 지표 요약 정보
        
        Returns:
            성공 점수 (0-100)
        """
        # 예시 계산식 (실제 비즈니스 요구사항에 맞게 조정 필요)
        staking_weight = 0.6  # 스테이킹 참여 가중치
        adoption_weight = 0.4  # 서비스 채택 가중치
        
        staking_score = min(100, summary["new_stakers_pct"] * 2)  # 50% 참여시 만점
        adoption_score = min(100, summary["new_adopters_pct"] * 2.5)  # 40% 채택시 만점
        
        return (staking_weight * staking_score) + (adoption_weight * adoption_score)
    
    def _generate_visualizations(self, metrics: Dict) -> List[str]:
        """지표 시각화 생성

        Args:
            metrics: 성과 지표 데이터
        
        Returns:
            생성된 시각화 파일 경로 목록
        """
        vis_dir = os.path.join(self.tracking_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        vis_files = []
        
        # 1. 스테이킹 참여 파이 차트
        plt.figure(figsize=(8, 6))
        labels = ['새로운 스테이커', '기존 스테이커', '미참여']
        stakers_count = metrics["summary"]["new_stakers"]
        existing_stakers = sum(1 for m in metrics["metrics"]["staking_participation"].values() 
                              if m["before"] and m["after"])
        non_stakers = metrics["summary"]["total_addresses"] - stakers_count - existing_stakers
        
        plt.pie([stakers_count, existing_stakers, non_stakers], 
                labels=labels, 
                autopct='%1.1f%%',
                colors=['#5cb85c', '#5bc0de', '#d9534f'])
        plt.title(f'스테이킹 참여 현황')
        
        pie_file = os.path.join(vis_dir, f"staking_pie_{timestamp}.png")
        plt.savefig(pie_file)
        plt.close()
        vis_files.append(pie_file)
        
        # 2. 서비스 채택 막대 그래프
        plt.figure(figsize=(10, 6))
        
        # 클러스터별 채택률 계산 (만약 클러스터 정보가 있다면)
        cluster_adoption = {}
        for address, metrics_data in metrics["metrics"]["service_adoption"].items():
            # 타겟 데이터에서 해당 주소의 클러스터 찾기
            address_data = self.targets[self.targets['address'] == address]
            if not address_data.empty and 'cluster' in address_data.columns:
                cluster = int(address_data['cluster'].iloc[0])
                
                if cluster not in cluster_adoption:
                    cluster_adoption[cluster] = {'total': 0, 'adopted': 0}
                
                cluster_adoption[cluster]['total'] += 1
                if metrics_data['new_adoption']:
                    cluster_adoption[cluster]['adopted'] += 1
        
        # 클러스터별 데이터 준비
        clusters = list(cluster_adoption.keys())
        adoption_rates = [
            (cluster_adoption[c]['adopted'] / cluster_adoption[c]['total']) * 100
            for c in clusters
        ] if clusters else []
        
        if clusters:
            plt.bar(clusters, adoption_rates, color='#428bca')
            plt.xlabel('클러스터 ID')
            plt.ylabel('서비스 채택률 (%)')
            plt.title('클러스터별 서비스 채택률')
            plt.xticks(clusters)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            bar_file = os.path.join(vis_dir, f"adoption_by_cluster_{timestamp}.png")
            plt.savefig(bar_file)
            plt.close()
            vis_files.append(bar_file)
        
        # 3. 트랜잭션 증가율 히스토그램
        txn_changes = [m["change"] for m in metrics["metrics"]["transaction_count_change"].values()]
        
        if txn_changes:
            plt.figure(figsize=(10, 6))
            plt.hist(txn_changes, bins=10, color='#5cb85c', alpha=0.7)
            plt.xlabel('트랜잭션 증가량')
            plt.ylabel('사용자 수')
            plt.title('트랜잭션 증가 분포')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            hist_file = os.path.join(vis_dir, f"txn_change_hist_{timestamp}.png")
            plt.savefig(hist_file)
            plt.close()
            vis_files.append(hist_file)
        
        return vis_files
    
    def _generate_html_report(self, report: Dict, output_file: str) -> None:
        """HTML 보고서 생성

        Args:
            report: 보고서 데이터
            output_file: 출력 파일 경로
        """
        # 간단한 HTML 템플릿
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{report['campaign_name']} 마케팅 캠페인 보고서</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .metrics {{ margin: 20px 0; }}
                .visualization {{ margin: 20px 0; }}
                .success-meter {{ 
                    width: 100%; 
                    height: 30px; 
                    background-color: #e9ecef; 
                    border-radius: 15px;
                    overflow: hidden;
                    margin: 10px 0;
                }}
                .success-value {{
                    height: 100%;
                    background-color: #28a745;
                    text-align: center;
                    line-height: 30px;
                    color: white;
                    font-weight: bold;
                }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ 
                    border: 1px solid #dee2e6; 
                    padding: 10px; 
                    text-align: left; 
                }}
                th {{ background-color: #e9ecef; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report['campaign_name']} 마케팅 캠페인 보고서</h1>
                <p>생성 시간: {report['generated_at']}</p>
            </div>
            
            <div class="summary">
                <h2>요약</h2>
                <table>
                    <tr>
                        <th>지표</th>
                        <th>값</th>
                    </tr>
                    <tr>
                        <td>분석 대상 주소 수</td>
                        <td>{report['summary']['total_addresses']}</td>
                    </tr>
                    <tr>
                        <td>데이터가 있는 주소 수</td>
                        <td>{report['summary']['addresses_with_data']}</td>
                    </tr>
                    <tr>
                        <td>평균 트랜잭션 증가량</td>
                        <td>{report['summary']['avg_txn_change']:.2f}</td>
                    </tr>
                    <tr>
                        <td>평균 토큰 증가량</td>
                        <td>{report['summary']['avg_token_change']:.2f}</td>
                    </tr>
                    <tr>
                        <td>새로운 스테이커 수</td>
                        <td>{report['summary']['new_stakers']} ({report['summary']['new_stakers_pct']:.2f}%)</td>
                    </tr>
                    <tr>
                        <td>새로운 서비스 채택자 수</td>
                        <td>{report['summary']['new_adopters']} ({report['summary']['new_adopters_pct']:.2f}%)</td>
                    </tr>
                </table>
            </div>
            
            <div class="metrics">
                <h2>캠페인 성공 점수</h2>
                <div class="success-meter">
                    <div class="success-value" style="width: {report['conclusion']['overall_success_rating']}%;">
                        {report['conclusion']['overall_success_rating']:.1f}%
                    </div>
                </div>
                <p>스테이킹 참여율: {report['conclusion']['staking_adoption_rate']:.2f}%</p>
                <p>서비스 채택률: {report['conclusion']['service_adoption_rate']:.2f}%</p>
            </div>
            
            <div class="visualizations">
                <h2>시각화</h2>
                {''.join([f'<div class="visualization"><img src="{os.path.basename(vis)}" alt="Visualization" style="max-width:100%;"></div>' for vis in report['visualizations']])}
            </div>
        </body>
        </html>
        """
        
        # HTML 파일 저장
        with open(output_file, 'w') as f:
            f.write(html)
    
    def monitor_continuously(self, fetch_data_func, interval_hours: float = 24.0) -> None:
        """지속적인 모니터링 실행

        Args:
            fetch_data_func: 최신 블록체인 데이터를 가져오는 함수 (주소 목록을 인자로 받음)
            interval_hours: 데이터 수집 간격 (시간 단위)
        """
        print(f"'{self.campaign_name}' 캠페인 모니터링 시작 (간격: {interval_hours}시간)")
        
        try:
            counter = 0
            interval_seconds = interval_hours * 3600
            
            while True:
                counter += 1
                snapshot_name = f"snapshot_{counter}"
                
                print(f"[{datetime.now().isoformat()}] 데이터 수집 중...")
                blockchain_data = fetch_data_func(self.target_addresses)
                
                print(f"[{datetime.now().isoformat()}] 스냅샷 생성 중: {snapshot_name}")
                self.create_snapshot(snapshot_name, blockchain_data)
                
                # 두 번째 스냅샷부터는 지표 계산 시작
                if counter >= 2:
                    prev_snapshot_id = f"snapshot_{counter-1}_{datetime.now().strftime('%Y%m%d')}"
                    curr_snapshot_id = f"snapshot_{counter}_{datetime.now().strftime('%Y%m%d')}"
                    
                    print(f"[{datetime.now().isoformat()}] 지표 계산 중...")
                    metrics = self.calculate_metrics(prev_snapshot_id, curr_snapshot_id)
                    
                    # 매 5번째 스냅샷마다 보고서 생성
                    if counter % 5 == 0:
                        print(f"[{datetime.now().isoformat()}] 보고서 생성 중...")
                        self.generate_report(metrics["id"])
                
                print(f"[{datetime.now().isoformat()}] 다음 수집까지 {interval_hours}시간 대기 중...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print(f"\n[{datetime.now().isoformat()}] 모니터링 중단됨")
        except Exception as e:
            print(f"\n[{datetime.now().isoformat()}] 오류 발생: {e}") 
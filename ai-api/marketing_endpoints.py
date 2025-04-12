"""
마케팅 타겟팅 및 추적 API 엔드포인트

app.py에 통합할 마케팅 관련 엔드포인트 정의
"""

import os
import json
import sys
import subprocess
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

# 상위 디렉토리 경로 추가 (import를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 마케팅 모듈 import
from marketing_targeting.target_extractor import MarketingTargetExtractor
from marketing_tracking.tracker import MarketingTracker
import marketing_helper as helper


# 라우터 정의
router = APIRouter(prefix="/marketing", tags=["marketing"])


# 데이터 모델
class TargetRequest(BaseModel):
    target_type: str
    mbti_types: Optional[List[str]] = None
    cluster_ids: Optional[List[int]] = None


class CampaignRequest(BaseModel):
    target_file: str
    name: Optional[str] = None


class CampaignStatus(BaseModel):
    campaign_id: str


class SnapshotRequest(BaseModel):
    campaign_id: str
    snapshot_name: Optional[str] = None
    data_source: Optional[str] = None


class MetricsRequest(BaseModel):
    campaign_id: str
    before_snapshot_id: str
    after_snapshot_id: str


class ReportRequest(BaseModel):
    campaign_id: str
    metrics_id: str
    format: Optional[str] = "json"


# 백그라운드 작업
def _create_snapshot_background(campaign_id: str, snapshot_name: str, data_source: Optional[str] = None):
    """스냅샷 생성 백그라운드 작업

    Args:
        campaign_id: 캠페인 ID
        snapshot_name: 스냅샷 이름
        data_source: 데이터 소스 (None이면 시뮬레이션 데이터 사용)
    """
    try:
        tracking_dir = "results/marketing_tracking"
        campaign_metadata = os.path.join(tracking_dir, campaign_id, "campaign_metadata.json")
        
        # 시뮬레이션 데이터 생성
        if not data_source:
            from marketing_tracking.main import simulate_blockchain_data
            
            # 캠페인 메타데이터에서 타겟 주소 목록 가져오기
            with open(campaign_metadata, 'r') as f:
                metadata = json.load(f)
                
            # 타겟 파일 로드
            if 'target_file' in metadata:
                with open(metadata['target_file'], 'r') as f:
                    targets = json.load(f)
                    addresses = [t['address'] for t in targets]
            else:
                # 없으면 캠페인 디렉토리에서 타겟 주소 찾기
                addresses = []
                # TODO: 구현
            
            # 시뮬레이션 데이터 생성
            blockchain_data = simulate_blockchain_data(addresses)
            
            # 임시 파일로 저장
            temp_data_file = os.path.join(tracking_dir, f"temp_data_{campaign_id}_{snapshot_name}.json")
            with open(temp_data_file, 'w') as f:
                json.dump(blockchain_data, f)
                
            data_source = temp_data_file
        
        # 스냅샷 생성 명령 실행
        cmd = [
            "python", "marketing_tracking/main.py", "snapshot",
            "--campaign", campaign_id,
            "--name", snapshot_name,
            "--data-file", data_source
        ]
        subprocess.run(cmd, check=True)
        
        # 임시 파일 삭제
        if os.path.exists(temp_data_file):
            os.remove(temp_data_file)
        
    except Exception as e:
        print(f"스냅샷 생성 오류: {e}")


def _calculate_metrics_background(campaign_id: str, before_snapshot_id: str, after_snapshot_id: str):
    """지표 계산 백그라운드 작업

    Args:
        campaign_id: 캠페인 ID
        before_snapshot_id: 이전 스냅샷 ID
        after_snapshot_id: 이후 스냅샷 ID
    """
    try:
        cmd = [
            "python", "marketing_tracking/main.py", "metrics",
            "--campaign", campaign_id,
            "--before", before_snapshot_id,
            "--after", after_snapshot_id
        ]
        subprocess.run(cmd, check=True)
        
    except Exception as e:
        print(f"지표 계산 오류: {e}")


def _generate_report_background(campaign_id: str, metrics_id: str, format: str = "json"):
    """보고서 생성 백그라운드 작업

    Args:
        campaign_id: 캠페인 ID
        metrics_id: 지표 ID
        format: 보고서 형식
    """
    try:
        cmd = [
            "python", "marketing_tracking/main.py", "report",
            "--campaign", campaign_id,
            "--metrics-id", metrics_id,
            "--format", format
        ]
        subprocess.run(cmd, check=True)
        
    except Exception as e:
        print(f"보고서 생성 오류: {e}")


# 엔드포인트
@router.post("/targets", response_model=Dict[str, Any])
async def create_target_list(request: TargetRequest):
    """타겟 사용자 목록 생성

    Args:
        request: 타겟 요청 정보
    
    Returns:
        생성된 타겟 파일 정보
    """
    try:
        extractor = MarketingTargetExtractor()
        
        # 타겟 유형에 따른 처리
        if request.target_type == "defi":
            targets = extractor.get_defi_long_term_holders()
            target_desc = "DeFi 장기 홀더"
        elif request.target_type == "nft":
            targets = extractor.get_nft_enthusiasts()
            target_desc = "NFT 열정 사용자"
        elif request.target_type == "community":
            targets = extractor.get_community_involved_users()
            target_desc = "커뮤니티 참여 사용자"
        elif request.target_type == "custom" and request.mbti_types:
            targets = extractor.filter_by_mbti(request.mbti_types)
            target_desc = f"커스텀 MBTI: {', '.join(request.mbti_types)}"
        else:
            raise HTTPException(status_code=400, detail="지원되지 않는 타겟 유형이거나 MBTI 유형이 지정되지 않았습니다")
        
        # 클러스터 필터링 (선택 사항)
        if request.cluster_ids:
            targets = targets[targets['cluster'].isin(request.cluster_ids)]
            target_desc += f" + 클러스터: {request.cluster_ids}"
        
        # 결과가 없는 경우
        if len(targets) == 0:
            return {
                "status": "empty",
                "message": "조건에 맞는 타겟이 없습니다",
                "target_type": request.target_type,
                "mbti_types": request.mbti_types,
                "cluster_ids": request.cluster_ids
            }
        
        # 타겟 파일 생성
        timestamp = helper.datetime.now().strftime('%Y%m%d%H%M%S')
        target_type_str = request.target_type
        target_file = f"results/marketing_targets/{target_type_str}_{timestamp}.json"
        extractor.export_targets(targets, target_file)
        
        return {
            "status": "success",
            "message": f"{len(targets)}명의 타겟 사용자를 추출했습니다",
            "target_file": target_file,
            "target_count": len(targets),
            "target_description": target_desc
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"타겟 생성 중 오류 발생: {str(e)}")


@router.post("/campaigns", response_model=Dict[str, Any])
async def create_campaign(request: CampaignRequest):
    """마케팅 캠페인 생성

    Args:
        request: 캠페인 생성 요청 정보
    
    Returns:
        생성된 캠페인 정보
    """
    try:
        campaign_id = helper.initialize_tracking_for_targets(request.target_file, request.name)
        return {
            "status": "success",
            "message": "캠페인이 생성되었습니다",
            "campaign_id": campaign_id,
            "target_file": request.target_file
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"캠페인 생성 중 오류 발생: {str(e)}")


@router.get("/campaigns", response_model=List[Dict[str, Any]])
async def list_campaigns():
    """모든 캠페인 목록 조회

    Returns:
        캠페인 목록
    """
    try:
        campaigns = helper.get_campaigns_summary()
        return campaigns
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"캠페인 목록 조회 중 오류 발생: {str(e)}")


@router.get("/campaigns/{campaign_id}", response_model=Dict[str, Any])
async def get_campaign_details(campaign_id: str):
    """캠페인 상세 정보 조회

    Args:
        campaign_id: 캠페인 ID
    
    Returns:
        캠페인 상세 정보
    """
    try:
        status = helper.get_campaign_status(campaign_id)
        
        if status["status"] == "not_found":
            raise HTTPException(status_code=404, detail=f"캠페인을 찾을 수 없습니다: {campaign_id}")
        elif status["status"] == "error":
            raise HTTPException(status_code=500, detail=f"캠페인 정보 조회 중 오류 발생: {status.get('error')}")
        
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"캠페인 조회 중 오류 발생: {str(e)}")


@router.post("/campaigns/{campaign_id}/snapshots", response_model=Dict[str, Any])
async def create_snapshot(campaign_id: str, request: SnapshotRequest, background_tasks: BackgroundTasks):
    """캠페인 스냅샷 생성

    Args:
        campaign_id: 캠페인 ID
        request: 스냅샷 요청 정보
        background_tasks: 백그라운드 작업
    
    Returns:
        스냅샷 생성 상태
    """
    try:
        # 캠페인 존재 확인
        status = helper.get_campaign_status(campaign_id)
        if status["status"] != "active":
            raise HTTPException(status_code=404, detail=f"활성 캠페인을 찾을 수 없습니다: {campaign_id}")
        
        # 스냅샷 이름 설정
        snapshot_name = request.snapshot_name or f"snapshot_{helper.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 백그라운드 작업으로 스냅샷 생성
        background_tasks.add_task(
            _create_snapshot_background,
            campaign_id,
            snapshot_name,
            request.data_source
        )
        
        return {
            "status": "processing",
            "message": "스냅샷 생성이 백그라운드에서 진행 중입니다",
            "campaign_id": campaign_id,
            "snapshot_name": snapshot_name
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"스냅샷 생성 중 오류 발생: {str(e)}")


@router.post("/campaigns/{campaign_id}/metrics", response_model=Dict[str, Any])
async def calculate_metrics(campaign_id: str, request: MetricsRequest, background_tasks: BackgroundTasks):
    """캠페인 지표 계산

    Args:
        campaign_id: 캠페인 ID
        request: 지표 계산 요청 정보
        background_tasks: 백그라운드 작업
    
    Returns:
        지표 계산 상태
    """
    try:
        # 캠페인 존재 확인
        status = helper.get_campaign_status(campaign_id)
        if status["status"] != "active":
            raise HTTPException(status_code=404, detail=f"활성 캠페인을 찾을 수 없습니다: {campaign_id}")
        
        # 백그라운드 작업으로 지표 계산
        background_tasks.add_task(
            _calculate_metrics_background,
            campaign_id,
            request.before_snapshot_id,
            request.after_snapshot_id
        )
        
        return {
            "status": "processing",
            "message": "지표 계산이 백그라운드에서 진행 중입니다",
            "campaign_id": campaign_id,
            "before_snapshot_id": request.before_snapshot_id,
            "after_snapshot_id": request.after_snapshot_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"지표 계산 중 오류 발생: {str(e)}")


@router.post("/campaigns/{campaign_id}/reports", response_model=Dict[str, Any])
async def generate_report(campaign_id: str, request: ReportRequest, background_tasks: BackgroundTasks):
    """캠페인 보고서 생성

    Args:
        campaign_id: 캠페인 ID
        request: 보고서 생성 요청 정보
        background_tasks: 백그라운드 작업
    
    Returns:
        보고서 생성 상태
    """
    try:
        # 캠페인 존재 확인
        status = helper.get_campaign_status(campaign_id)
        if status["status"] != "active":
            raise HTTPException(status_code=404, detail=f"활성 캠페인을 찾을 수 없습니다: {campaign_id}")
        
        # 백그라운드 작업으로 보고서 생성
        background_tasks.add_task(
            _generate_report_background,
            campaign_id,
            request.metrics_id,
            request.format
        )
        
        return {
            "status": "processing",
            "message": "보고서 생성이 백그라운드에서 진행 중입니다",
            "campaign_id": campaign_id,
            "metrics_id": request.metrics_id,
            "format": request.format
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"보고서 생성 중 오류 발생: {str(e)}")


# app.py에 추가할 코드:
"""
from marketing_endpoints import router as marketing_router
app.include_router(marketing_router)
""" 
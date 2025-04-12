#!/usr/bin/env python3
"""
AI API 서버
블록체인 주소 분석 및 프로필 이미지 생성 API (비동기 백그라운드 처리)
분석 대상: sourceAddress
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib
from PIL import Image, ImageDraw
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# .env 파일 로드 (경로 설정 전에 호출)
load_dotenv()
logger.info(".env 파일 로드 시도 완료.")

# 경로 설정 먼저 수행
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
logger.info(f"프로젝트 루트 경로 추가: {project_root}")
logger.info(f"사용 중인 sys.path: {sys.path}")

# 경로 설정 후 import
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import boto3

# --- 분석 및 프로필 생성 모듈 임포트 ---
try:
    from profile_generator import AIProfileGenerator
    logger.info("profile_generator 임포트 성공")
except ImportError as e:
    logger.error(f"AIProfileGenerator 임포트 실패: {e}. profile_generator.py 파일 위치 확인 필요.")
    AIProfileGenerator = None

# 주석 처리 또는 삭제 (직접 호출하지 않으므로):
# try:
#     from ai_clustering.main import analyze_new_address as analyze_clusters_for_address
#     logger.info("ai_clustering.main.analyze_new_address 임포트 성공")
# except ImportError as e:
#     logger.error(f"ai_clustering 임포트 실패: {e}. 경로 및 __init__.py 확인 필요.")
#     analyze_clusters_for_address = None
#
# try:
#     from ai_deduction.main import analyze_address as run_inference_for_address
#     logger.info("ai_deduction.main.analyze_address 임포트 성공")
# except ImportError as e:
#     logger.error(f"ai_deduction 임포트 실패: {e}. 경로 및 __init__.py 확인 필요.")
#     run_inference_for_address = None
# --- 임포트 완료 ---

# 환경 변수 로드 (load_dotenv 호출 후)
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME", "aidrop-profiles")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 이미지 생성이 제한될 수 있습니다.")


# AWS S3 클라이언트 초기화
s3_client = None
if AWS_ACCESS_KEY and AWS_SECRET_KEY and AWS_BUCKET_NAME:
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        logger.info(f"AWS S3 클라이언트 초기화 완료 (버킷: {AWS_BUCKET_NAME}).")
    except Exception as s3_init_err:
        logger.error(f"AWS S3 클라이언트 초기화 오류: {s3_init_err}")
        s3_client = None
else:
    logger.warning("AWS 자격 증명 또는 버킷 이름이 설정되지 않아 S3 클라이언트를 초기화하지 못했습니다.")

# FastAPI 앱 초기화
app = FastAPI(
    title="AI Pipeline API",
    description="블록체인 주소 분석 및 프로필 이미지 생성 API (비동기)",
    version="1.2.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 결과 저장 디렉토리 생성
RESULTS_DIR = os.path.join(project_root, "results/requests")
ANALYSIS_RESULTS_DIR = os.path.join(project_root, "results/analysis") # 사전 분석 결과 저장 위치
PROFILE_IMG_DIR = os.path.join(project_root, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True) # 디렉토리 생성 확인 추가
os.makedirs(PROFILE_IMG_DIR, exist_ok=True)
logger.info(f"API 요청 상태 저장 디렉토리 확인/생성: {RESULTS_DIR}")
logger.info(f"사전 분석 결과 저장 디렉토리 확인/생성: {ANALYSIS_RESULTS_DIR}") # 로그 추가
logger.info(f"프로필 이미지 저장 디렉토리 확인/생성: {PROFILE_IMG_DIR}")

# 모델 정의
class AnalysisRequest(BaseModel):
    sourceAddress: str = Field(..., description="분석 대상 주소")
    storyAddress: str = Field(..., description="스토리 주소 (참조용)")
    sourceChainId: str = Field(..., description="소스 체인 ID")

class AnalysisResponse(BaseModel):
    status: str = Field(..., description="처리 상태 (e.g., 'processing', 'completed', 'error')")
    message: str = Field(..., description="상태 메시지")
    requestId: Optional[str] = Field(None, description="요청 ID")
    mbti: Optional[str] = Field(None, description="분석된 MBTI 유형 (완료 시)")
    imageUrl: Optional[str] = Field(None, description="생성된 프로필 메타데이터 JSON URL (완료 시)")
    cluster: Optional[int] = Field(None, description="분석된 클러스터 ID (완료 시)")
    traits: Optional[Dict[str, Any]] = Field(None, description="분석된 사용자 특성 (완료 시)")

class ResultRequest(BaseModel):
    requestId: str

# --- Helper Functions for Local File Storage ---
def save_request_status(request_id: str, data: Dict[str, Any]):
    filepath = os.path.join(RESULTS_DIR, f"{request_id}.json")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"요청 상태 저장 완료: {filepath}")
    except IOError as e:
        logger.error(f"요청 상태 파일 저장 오류 ({request_id}): {e}")

def update_request_status_field(request_id: str, updates: Dict[str, Any]):
    filepath = os.path.join(RESULTS_DIR, f"{request_id}.json")
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r+', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"요청 상태 파일 JSON 디코드 오류: {filepath}, 파일을 덮어씁니다.")
                    data = {}
                data.update(updates)
                f.seek(0)
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.truncate()
            logger.debug(f"요청 상태 업데이트 완료: {filepath} - {updates}")
        else:
            logger.warning(f"요청 상태 업데이트 시도: 파일 없음, 새로 생성 - {filepath}")
            updates['request_id'] = request_id
            if 'status' not in updates: updates['status'] = 'unknown'
            save_request_status(request_id, updates)
    except IOError as e:
        logger.error(f"요청 상태 파일 업데이트 오류 ({request_id}): {e}")

def save_result_data(request_id: str, data: Dict[str, Any]):
    filepath = os.path.join(RESULTS_DIR, f"{request_id}_result.json")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"결과 데이터 저장 완료: {filepath}")
    except IOError as e:
        logger.error(f"결과 데이터 파일 저장 오류 ({request_id}): {e}")

def save_error_data(request_id: str, error_message: str):
    filepath = os.path.join(RESULTS_DIR, f"{request_id}_error.json")
    error_data = {
        "request_id": request_id,
        "status": "error",
        "error": error_message,
        "time": datetime.now().isoformat()
    }
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"오류 데이터 저장 완료: {filepath}")
    except IOError as e:
        logger.error(f"오류 데이터 파일 저장 오류 ({request_id}): {e}")

def get_request_data(request_id: str) -> Optional[Dict[str, Any]]:
    filepath = os.path.join(RESULTS_DIR, f"{request_id}.json")
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"요청 정보 파일 읽기 오류 ({request_id}): {e}")
        return None

def get_result_data(request_id: str) -> Optional[Dict[str, Any]]:
    filepath = os.path.join(RESULTS_DIR, f"{request_id}_result.json")
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"결과 정보 파일 읽기 오류 ({request_id}): {e}")
        return None
# --- End Helper Functions ---

# --- S3 Upload Function ---
async def upload_to_s3(file_path: str, key: str, content_type: Optional[str] = None) -> str:
    if not s3_client:
        raise ValueError("AWS S3 클라이언트가 초기화되지 않았습니다.")
    try:
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        logger.info(f"S3 업로드 시도: {file_path} -> s3://{AWS_BUCKET_NAME}/{key} (ExtraArgs: {extra_args})")
        s3_client.upload_file(file_path, AWS_BUCKET_NAME, key, ExtraArgs=extra_args)
        url = f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{key}"
        logger.info(f"S3 업로드 완료: {url} (Content-Type: {content_type})")
        return url
    except Exception as e:
        logger.error(f"S3 업로드 오류 ({key}): {str(e)}", exc_info=True)
        raise
# --- End S3 Upload ---

# --- Background Analysis Task ---
async def process_analysis_background(
    request_id: str,
    source_address: str, # 분석 대상 주소
    story_address: str, # 참조용 스토리 주소
    source_chain_id: str
):
    analyzed_address = source_address.lower() # 주소를 소문자로 통일
    logger.info(f"백그라운드 분석 시작: {request_id} - 분석 대상 주소: {analyzed_address}")

    image_url = None
    metadata_url = None
    mbti_result = "Unknown"
    cluster_result = None
    traits_result = None
    final_status = "processing"
    error_message = None
    image_path = None
    pre_analyzed_data = None # 사전 분석 데이터 저장 변수

    try:
        # 1. 사전 분석 결과 로드 (ai_clustering, ai_deduction 결과)
        analysis_file_path = os.path.join(ANALYSIS_RESULTS_DIR, source_chain_id, f"{analyzed_address}.json")
        logger.info(f"사전 분석 결과 파일 확인 시도: {analysis_file_path}")

        if os.path.exists(analysis_file_path):
            try:
                with open(analysis_file_path, 'r', encoding='utf-8') as f:
                    pre_analyzed_data = json.load(f)
                logger.info(f"사전 분석 결과 로드 성공: {request_id} (주소: {analyzed_address})")

                # 분석 데이터 추출 (키 이름은 실제 저장된 형식에 맞게 조정 필요)
                # 예시: {'mbti': 'ISTJ', 'cluster': 3, 'traits': {...}} 형태 가정
                mbti_result = pre_analyzed_data.get("mbti", "Unknown") # 키 이름 확인 필요
                cluster_result = pre_analyzed_data.get("cluster")     # 키 이름 확인 필요
                traits_result = pre_analyzed_data.get("traits")       # 키 이름 확인 필요

            except (IOError, json.JSONDecodeError) as file_err:
                error_message = f"사전 분석 결과 파일 읽기/파싱 오류: {file_err}"
                logger.error(f"{error_message} ({request_id}) - {analysis_file_path}")
                final_status = "error"
        else:
            error_message = "사전 분석 결과 파일을 찾을 수 없습니다."
            logger.warning(f"{error_message} ({request_id}) - {analysis_file_path}")
            final_status = "error" # 결과 파일 없으면 오류 처리

        # 2. (오류 없을 경우) 프로필 이미지 생성 (AIProfileGenerator)
        if final_status != "error" and AIProfileGenerator:
            try:
                if not OPENAI_API_KEY:
                     logger.warning(f"OpenAI API 키 미설정. 프로필 이미지 생성 건너뜀. ({request_id})")
                     # raise ValueError("OpenAI API 키 미설정") # 오류 대신 경고 후 진행
                else:
                    generator = AIProfileGenerator(
                        output_dir=PROFILE_IMG_DIR,
                        api_key=OPENAI_API_KEY
                    )
                    logger.info(f"프로필 이미지 생성 시작: {request_id} (대상: {analyzed_address}) ")
                    # 이미지 생성 시 분석 결과(mbti, traits 등)를 활용하도록 generate_profile 수정 필요할 수 있음
                    # 현재는 주소만 사용 -> generate_profile(analyzed_address)
                    # 만약 mbti 등 사용한다면 -> generate_profile(analyzed_address, mbti=mbti_result, traits=traits_result)
                    image_path_or_error = generator.generate_profile(analyzed_address) # 필요 시 인자 추가

                    if isinstance(image_path_or_error, str) and image_path_or_error.endswith('.png') and os.path.exists(image_path_or_error):
                        image_path = image_path_or_error
                        logger.info(f"프로필 이미지 생성 완료: {request_id} - {image_path}")
                        # 3. 이미지 S3 업로드
                        if s3_client:
                            try:
                                s3_image_key = f"profiles/{source_chain_id}/{analyzed_address}/{os.path.basename(image_path)}"
                                image_url = await upload_to_s3(image_path, s3_image_key)
                            except Exception as s3_img_err:
                                logger.error(f"S3 이미지 업로드 오류: {request_id} - {str(s3_img_err)}")
                                # 이미지 업로드 실패해도 메타데이터는 생성 시도 가능
                        else:
                             logger.warning(f"S3 클라이언트 미설정. 이미지 S3 업로드 건너뜀. ({request_id})")
                    else:
                        logger.error(f"프로필 이미지 생성 실패: {request_id} - {str(image_path_or_error)}")
            except Exception as img_gen_err:
                 logger.error(f"프로필 이미지 생성/처리 중 오류: {request_id} - {img_gen_err}", exc_info=True)
                 # 이미지 생성 오류는 전체 오류로 처리하지 않고 진행 가능 (image_url=None)
        elif final_status != "error":
             logger.warning(f"프로필 이미지 생성 건너뜀: AIProfileGenerator 로드 실패 - {request_id}")

        # 4. (오류 없을 경우) 메타데이터 JSON 생성 및 S3 업로드
        if final_status != "error":
            # 이미지 URL 없어도 메타데이터는 생성 (이미지 필드 비워두거나 기본 이미지 사용)
            try:
                metadata_content = {
                    "description": f"AI-generated profile for {analyzed_address}",
                    "external_url": "", # 필요 시 채우기
                    "image": image_url or "", # 이미지 없으면 빈 문자열 또는 기본 이미지 URL
                    "name": f"AI Profile - {analyzed_address[:8]}...",
                    "attributes": [
                        {"trait_type": "MBTI", "value": mbti_result}, # 파일에서 읽은 값 사용
                        {"trait_type": "Cluster", "value": cluster_result if cluster_result is not None else "N/A"}, # 파일에서 읽은 값 사용
                        # traits_result가 None이 아닐 경우에만 속성 추가
                        *([{"trait_type": k, "value": v} for k, v in traits_result.items()] if isinstance(traits_result, dict) else [])
                    ]
                }
                # 파일명 고유성 확보 (request_id 사용)
                local_json_filename = f"{request_id}_metadata.json"
                local_json_filepath = os.path.join(RESULTS_DIR, local_json_filename) # 요청별 결과 디렉토리에 임시 저장

                with open(local_json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(metadata_content, f, ensure_ascii=False, indent=2)
                logger.info(f"로컬 메타데이터 JSON 파일 생성 완료: {local_json_filepath}")

                # S3 업로드
                if s3_client:
                    s3_metadata_key = f"profiles/{source_chain_id}/{analyzed_address}/{request_id}_metadata.json" # S3 경로에도 request_id 포함
                    try:
                        metadata_url = await upload_to_s3(local_json_filepath, s3_metadata_key, content_type='application/json')
                        logger.info(f"메타데이터 JSON S3 업로드 완료: {request_id} - {metadata_url}")
                    except Exception as s3_meta_err:
                         logger.error(f"S3 메타데이터 업로드 오류: {request_id} - {str(s3_meta_err)}")
                         metadata_url = None # 업로드 실패 시 URL 없음
                else:
                    logger.warning(f"S3 클라이언트 미설정. 메타데이터 S3 업로드 건너뜀. ({request_id})")
                    metadata_url = None # 로컬 파일 경로를 반환하거나 None 처리

                # 임시 로컬 메타데이터 파일 삭제 (옵션)
                try:
                    # if metadata_url: # S3 업로드 성공 시에만 삭제? 혹은 항상 삭제? -> 항상 삭제
                    os.remove(local_json_filepath)
                    logger.info(f"임시 메타데이터 파일 삭제 완료: {local_json_filepath}")
                except OSError as rm_err:
                    logger.warning(f"임시 메타데이터 파일 삭제 실패: {local_json_filepath} - {rm_err}")

            except Exception as meta_err:
                 logger.error(f"메타데이터 JSON 생성/업로드 오류: {request_id} - {str(meta_err)}", exc_info=True)
                 metadata_url = None
                 # 메타데이터 생성 오류는 심각할 수 있으므로 error 상태로 변경 고려
                 # final_status = "error"
                 # error_message = f"메타데이터 처리 오류: {str(meta_err)}"

            # 모든 단계가 오류 없이 끝나면 completed 상태
            if final_status != "error": # 중간에 error로 설정되지 않았다면 완료
                final_status = "completed"

    except Exception as e:
        # 예상치 못한 최상위 예외 처리
        error_message = f"백그라운드 분석 중 예상치 못한 예외 발생: {str(e)}"
        logger.error(f"{error_message} - Request ID: {request_id}", exc_info=True)
        final_status = "error"

    # 5. 최종 결과 데이터 준비 및 저장
    result_data = {
        "request_id": request_id,
        "source_address": source_address, # 원본 요청 주소 유지
        "story_address": story_address,
        "analyzed_address": analyzed_address, # 실제 분석된 주소 (소문자)
        "source_chain_id": source_chain_id,
        "completion_time": datetime.now().isoformat(),
        "status": final_status,
        "mbti_type": mbti_result, # 최종 결과
        "cluster": cluster_result, # 최종 결과
        "traits": traits_result,   # 최종 결과
        "image_url": image_url, # 최종 이미지 URL (S3)
        "metadata_url": metadata_url # 최종 메타데이터 URL (S3)
    }
    if final_status == "error" and error_message:
         result_data["error"] = error_message
    save_result_data(request_id, result_data) # 결과 파일 저장 (_result.json)

    # 6. 요청 상태 업데이트
    update_payload = {"status": final_status, "updated_at": datetime.now().isoformat()}
    if final_status == "error" and error_message:
         update_payload["error"] = error_message
    update_request_status_field(request_id, update_payload) # 상태 파일 업데이트 (.json)

    # 7. 최종 로그 및 오류 데이터 저장 (오류 시)
    if final_status == "completed":
         logger.info(f"백그라운드 분석 완료: {request_id} (Metadata URL: {metadata_url})")
    else:
         # save_error_data 함수는 이제 save_result_data 및 update_request_status_field 에서 오류 정보를 포함하므로 중복될 수 있음. 필요시 유지.
         # save_error_data(request_id, error_message or "알 수 없는 백그라운드 분석 오류")
         logger.error(f"백그라운드 분석 실패/오류: {request_id} - {error_message or '알 수 없는 오류'}")
# --- End Background Task ---


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "AI Pipeline API"}

@app.post("/analyze", response_model=AnalysisResponse, status_code=202)
async def analyze_address_endpoint(analysis_request: AnalysisRequest, background_tasks: BackgroundTasks):
    """분석 요청(sourceAddress 대상)을 받고 백그라운드 처리를 시작합니다."""
    timestamp = int(time.time())
    unique_str = f"{analysis_request.sourceChainId}_{analysis_request.sourceAddress}_{timestamp}"
    request_id = hashlib.md5(unique_str.encode()).hexdigest()

    logger.info(f"분석 요청 접수: {request_id} - sourceAddress: {analysis_request.sourceAddress}, storyAddress: {analysis_request.storyAddress}")

    request_data = {
        "request_id": request_id,
        "source_chain_id": analysis_request.sourceChainId,
        "story_address": analysis_request.storyAddress,
        "source_address": analysis_request.sourceAddress,
        "status": "processing",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "error": None
    }
    save_request_status(request_id, request_data)
    logger.info(f"요청 기록 로컬 저장 ({request_id})")

    background_tasks.add_task(
        process_analysis_background,
        request_id,
        analysis_request.sourceAddress,
        analysis_request.storyAddress,
        analysis_request.sourceChainId
    )
    logger.info(f"백그라운드 작업 예약 완료: {request_id}")

    return AnalysisResponse(
        status="processing",
        message="분석 요청이 접수되었으며 백그라운드에서 처리 중입니다.",
        requestId=request_id
    )

@app.post("/api/result", response_model=AnalysisResponse)
async def get_analysis_result_endpoint(request: ResultRequest):
    """요청 ID로 분석 상태 및 결과를 조회합니다."""
    request_id = request.requestId
    logger.info(f"결과 조회 요청: {request_id}")

    request_data = get_request_data(request_id)
    if not request_data:
        logger.warning(f"결과 조회 실패: 요청 ID 없음 - {request_id}")
        raise HTTPException(status_code=404, detail="분석 요청을 찾을 수 없습니다.")

    status = request_data.get("status", "unknown")

    if status == "completed":
        result_data = get_result_data(request_id)
        if result_data:
            logger.info(f"결과 조회 성공 (완료): {request_id}")
            metadata_url = result_data.get("metadata_url")
            return AnalysisResponse(
                status="completed",
                message="분석이 완료되었습니다.",
                requestId=request_id,
                mbti=result_data.get("mbti_type"),
                imageUrl=metadata_url,
                cluster=result_data.get("cluster"),
                traits=result_data.get("traits")
            )
        else:
            logger.error(f"결과 조회 오류: 상태는 완료이나 결과 파일 없음 - {request_id}")
            return AnalysisResponse(status="error", message="결과 데이터 오류", requestId=request_id)
    elif status == "processing":
        logger.info(f"결과 조회 성공 (진행중): {request_id}")
        return AnalysisResponse(status="processing", message="분석 진행 중", requestId=request_id)
    elif status == "error":
        error_message = request_data.get("error", "알 수 없는 오류")
        logger.warning(f"결과 조회 성공 (오류): {request_id} - {error_message}")
        return AnalysisResponse(status="error", message=f"분석 오류: {error_message}", requestId=request_id)
    else:
        logger.warning(f"결과 조회: 알 수 없는 상태 ({status}) - {request_id}")
        raise HTTPException(status_code=500, detail=f"알 수 없는 상태: {status}")
# --- End API Endpoints ---

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info") 
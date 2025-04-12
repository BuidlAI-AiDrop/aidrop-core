#!/usr/bin/env python3
"""
AI 파이프라인 유틸리티 함수
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Union
from datetime import datetime

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    로그 설정
    
    Args:
        name: 로거 이름
        level: 로깅 레벨
        
    Returns:
        로거 객체
    """
    # 로깅 디렉토리 생성
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 현재 시간 기반 로그 파일명
    log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 포맷 지정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def validate_data(df: pd.DataFrame) -> bool:
    """
    데이터 유효성 검증
    
    Args:
        df: 검증할 데이터프레임
        
    Returns:
        유효성 여부
    """
    # 필수 열 확인
    if 'address' not in df.columns:
        logging.error("필수 열 'address'가 없습니다.")
        return False
    
    # 주소 형식 확인 (이더리움 주소)
    if not all(addr.startswith('0x') and len(addr) == 42 for addr in df['address']):
        logging.warning("유효하지 않은 이더리움 주소가 있습니다.")
        # 경고만 하고 계속 진행
    
    # 중복 주소 확인
    if df['address'].duplicated().any():
        logging.warning(f"중복된 주소가 있습니다: {df['address'].duplicated().sum()}개")
        # 경고만 하고 계속 진행
    
    # 추가 검증 규칙...
    
    return True

def save_results(results: Dict[str, Any], output_file: str) -> bool:
    """
    결과 저장
    
    Args:
        results: 저장할 결과 데이터
        output_file: 출력 파일 경로
        
    Returns:
        저장 성공 여부
    """
    try:
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # JSON으로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"결과 저장 완료: {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"결과 저장 중 오류 발생: {str(e)}")
        return False

def load_addresses_from_file(file_path: str) -> List[str]:
    """
    파일에서 주소 목록 로드
    
    Args:
        file_path: 주소 목록 파일 경로 (txt, csv, json)
        
    Returns:
        주소 목록
    """
    if not os.path.exists(file_path):
        logging.error(f"파일을 찾을 수 없음: {file_path}")
        return []
    
    addresses = []
    
    try:
        # 파일 형식에 따라 로드
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                addresses = [line.strip() for line in f if line.strip()]
                
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if 'address' in df.columns:
                addresses = df['address'].tolist()
            else:
                # 첫 번째 열을 주소로 가정
                addresses = df.iloc[:, 0].tolist()
                
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # 형식에 따라 주소 추출
                if isinstance(data, list):
                    if all(isinstance(item, str) for item in data):
                        # 문자열 목록인 경우
                        addresses = data
                    elif all(isinstance(item, dict) for item in data):
                        # 사전 목록인 경우
                        addresses = [item.get('address') for item in data if 'address' in item]
                elif isinstance(data, dict) and 'addresses' in data:
                    # {'addresses': [...]} 형식
                    addresses = data['addresses']
                    
        else:
            logging.error(f"지원하지 않는 파일 형식: {file_path}")
            return []
            
        # 주소 필터링 및 중복 제거
        addresses = [addr for addr in addresses if addr and isinstance(addr, str)]
        addresses = list(set(addresses))  # 중복 제거
        
        logging.info(f"{len(addresses)}개 주소 로드 완료")
        return addresses
        
    except Exception as e:
        logging.error(f"주소 로드 중 오류 발생: {str(e)}")
        return []

def merge_data_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    데이터 수집/분석 결과 병합
    
    Args:
        results: 결과 목록
        
    Returns:
        병합된 결과
    """
    if not results:
        return {'status': 'error', 'message': '결과 없음'}
    
    # 성공/실패 카운트
    success_count = sum(1 for r in results if r.get('status') == 'success')
    error_count = len(results) - success_count
    
    # 주소별 결과
    address_results = {}
    for result in results:
        if 'address' in result:
            address_results[result['address']] = {
                'status': result.get('status'),
                'error': result.get('error'),
                'data': {k: v for k, v in result.items() 
                         if k not in ['status', 'error', 'address']}
            }
    
    return {
        'status': 'success' if error_count == 0 else 'partial',
        'total': len(results),
        'success_count': success_count,
        'error_count': error_count,
        'address_results': address_results
    }

def format_output_path(base_path: str, suffix: str = None, 
                     ext: str = None, timestamp: bool = True) -> str:
    """
    출력 파일 경로 포맷
    
    Args:
        base_path: 기본 파일 경로
        suffix: 추가할 접미사
        ext: 확장자 (기본값: json)
        timestamp: 타임스탬프 추가 여부
    
    Returns:
        포맷된 파일 경로
    """
    # 디렉토리와 파일명 분리
    directory, filename = os.path.split(base_path)
    
    # 파일명과 확장자 분리
    if '.' in filename:
        name, original_ext = filename.rsplit('.', 1)
        ext = ext or original_ext
    else:
        name = filename
        ext = ext or 'json'
    
    # 접미사 추가
    if suffix:
        name = f"{name}_{suffix}"
    
    # 타임스탬프 추가
    if timestamp:
        name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 최종 경로 구성
    final_path = os.path.join(directory, f"{name}.{ext}")
    
    # 디렉토리 생성
    os.makedirs(directory, exist_ok=True)
    
    return final_path 
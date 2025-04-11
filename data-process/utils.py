"""
데이터 처리 및 저장 모듈을 위한 유틸리티 함수
"""

import os
import json
import logging
import hashlib
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# 로깅 설정
def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    로깅 설정 함수
    
    Args:
        name: 로거 이름
        log_level: 로깅 레벨 (기본값: INFO)
        
    Returns:
        설정된 로거 객체
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger

logger = setup_logger('data_process_utils')

# 이더리움 주소 검증
def is_valid_eth_address(address: str) -> bool:
    """
    이더리움 주소 유효성 검사
    
    Args:
        address: 검사할 이더리움 주소
        
    Returns:
        유효성 여부 (True/False)
    """
    if not address or not isinstance(address, str):
        return False
    
    # 0x로 시작하고 길이가 42인지 확인 (0x + 40자의 16진수)
    if not address.startswith('0x') or len(address) != 42:
        return False
    
    # 0x 이후가 16진수 문자인지 확인
    try:
        int(address[2:], 16)
        return True
    except ValueError:
        return False

# 파일 경로 생성
def ensure_directory(directory: str) -> str:
    """
    디렉토리가 존재하지 않으면 생성
    
    Args:
        directory: 생성할 디렉토리 경로
        
    Returns:
        생성된 디렉토리 경로
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory

# 데이터 해시 생성
def generate_data_hash(data: Dict[str, Any]) -> str:
    """
    데이터의 해시값 생성
    
    Args:
        data: 해시할 데이터 딕셔너리
        
    Returns:
        데이터 해시값 (SHA-256)
    """
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()

# 파일명 생성
def generate_filename(prefix: str, address: str, suffix: str = "") -> str:
    """
    데이터 파일명 생성
    
    Args:
        prefix: 파일명 접두어
        address: 이더리움 주소
        suffix: 파일명 접미어 (선택사항)
        
    Returns:
        생성된 파일명
    """
    # 주소의 처음 6자와 마지막 4자만 사용
    short_address = f"{address[:8]}...{address[-6:]}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if suffix:
        return f"{prefix}_{short_address}_{timestamp}_{suffix}.json"
    return f"{prefix}_{short_address}_{timestamp}.json"

# 데이터 저장
def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """
    데이터를 JSON 파일로 저장
    
    Args:
        data: 저장할 데이터 딕셔너리
        file_path: 저장할 파일 경로
        
    Returns:
        저장 성공 여부
    """
    try:
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory(directory)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"데이터가 {file_path}에 저장되었습니다.")
        return True
    except Exception as e:
        logger.error(f"데이터 저장 실패: {str(e)}")
        return False

# 데이터 로드
def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    JSON 파일에서 데이터 로드
    
    Args:
        file_path: 로드할 파일 경로
        
    Returns:
        로드된 데이터 딕셔너리 또는 None (실패 시)
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"파일이 존재하지 않습니다: {file_path}")
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"데이터 로드 실패: {str(e)}")
        return None

# 타임스탬프 변환
def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime.datetime:
    """
    유닉스 타임스탬프를 datetime 객체로 변환
    
    Args:
        timestamp: 유닉스 타임스탬프
        
    Returns:
        datetime 객체
    """
    return datetime.datetime.fromtimestamp(timestamp)

def datetime_to_timestamp(dt: datetime.datetime) -> int:
    """
    datetime 객체를 유닉스 타임스탬프로 변환
    
    Args:
        dt: datetime 객체
        
    Returns:
        유닉스 타임스탬프
    """
    return int(dt.timestamp())

# 블록체인 데이터 처리를 위한 유틸리티 함수
def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    로깅 설정 함수
    
    Args:
        log_level: 로깅 레벨 (기본값: logging.INFO)
        log_file: 로그 파일 경로 (기본값: None)
        console_output: 콘솔 출력 여부 (기본값: True)
        
    Returns:
        설정된 로거 객체
    """
    logger = logging.getLogger('data_process')
    logger.setLevel(log_level)
    
    # 포맷터 생성
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 모든 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 콘솔 출력 설정
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 파일 출력 설정
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 이더리움 주소 검증
def validate_ethereum_address(address: str) -> bool:
    """
    이더리움 주소 형식 검증
    
    Args:
        address: 검증할 이더리움 주소
        
    Returns:
        유효성 여부 (True/False)
    """
    if not address.startswith('0x'):
        return False
    
    # 주소 길이 검증 (0x 제외 40자)
    if len(address) != 42:
        return False
    
    # 16진수 형식 검증
    pattern = re.compile(r'^0x[a-fA-F0-9]{40}$')
    return bool(pattern.match(address))

# 해시 생성 함수
def generate_hash(data: Any) -> str:
    """
    입력 데이터의 해시값 생성
    
    Args:
        data: 해시를 생성할 데이터 (문자열, 딕셔너리 등)
        
    Returns:
        해시 문자열
    """
    if isinstance(data, dict) or isinstance(data, list):
        data = json.dumps(data, sort_keys=True)
    elif not isinstance(data, str):
        data = str(data)
    
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

# 디렉토리 생성 함수
def ensure_directory(directory_path: str) -> str:
    """
    지정된 디렉토리가 없으면 생성
    
    Args:
        directory_path: 생성할 디렉토리 경로
        
    Returns:
        생성된 디렉토리 경로
    """
    os.makedirs(directory_path, exist_ok=True)
    return directory_path

# 데이터 파일 경로 생성
def get_data_filepath(
    data_dir: str,
    identifier: str,
    data_type: str,
    extension: str = 'json'
) -> str:
    """
    데이터 파일 경로 생성
    
    Args:
        data_dir: 데이터 디렉토리 기본 경로
        identifier: 데이터 식별자 (주로 지갑 주소 해시)
        data_type: 데이터 유형 (raw, processed, vectors)
        extension: 파일 확장자 (기본값: json)
        
    Returns:
        생성된 파일 경로
    """
    # 디렉토리 확인 및 생성
    full_dir = os.path.join(data_dir, data_type)
    ensure_directory(full_dir)
    
    # 파일명 생성
    filename = f"{identifier}.{extension}"
    return os.path.join(full_dir, filename)

# 타임스탬프 생성
def get_timestamp() -> str:
    """
    현재 시간의 표준 형식 타임스탬프 반환
    
    Returns:
        ISO 8601 형식의 타임스탬프 문자열
    """
    return datetime.datetime.now().isoformat()

# JSON 파일 저장
def save_json(data: Union[Dict, List], filepath: str) -> bool:
    """
    데이터를 JSON 파일로 저장
    
    Args:
        data: 저장할 데이터 (딕셔너리 또는 리스트)
        filepath: 저장할 파일 경로
        
    Returns:
        성공 여부 (True/False)
    """
    try:
        # 디렉토리 확인
        directory = os.path.dirname(filepath)
        if directory:
            ensure_directory(directory)
        
        # 데이터 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger = logging.getLogger('data_process')
        logger.error(f"JSON 파일 저장 실패: {str(e)}")
        return False

# JSON 파일 로드
def load_json(filepath: str) -> Optional[Union[Dict, List]]:
    """
    JSON 파일에서 데이터 로드
    
    Args:
        filepath: 로드할 파일 경로
        
    Returns:
        로드된 데이터 (딕셔너리 또는 리스트) 또는 None (실패 시)
    """
    try:
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger = logging.getLogger('data_process')
        logger.error(f"JSON 파일 로드 실패: {str(e)}")
        return None

# API 결과 유효성 검증
def validate_api_response(response: Dict) -> Tuple[bool, Optional[str]]:
    """
    API 응답 결과의 유효성 검증
    
    Args:
        response: API 응답 데이터
        
    Returns:
        (유효성 여부, 오류 메시지)
    """
    # 기본 응답 형식 검증
    if not isinstance(response, dict):
        return False, "응답이 유효한 JSON 형식이 아닙니다"
    
    # 상태 코드 확인 (Etherscan API 형식 기준)
    if 'status' in response and response['status'] == '0':
        error_msg = response.get('message', '알 수 없는 오류')
        return False, f"API 오류: {error_msg}"
        
    # 결과 필드 확인
    if 'result' not in response:
        return False, "응답에 결과 필드가 없습니다"
    
    return True, None

# 문자열 대소문자 정규화
def normalize_address(address: str) -> str:
    """
    이더리움 주소를 정규화된 형식으로 변환
    
    Args:
        address: 이더리움 주소
        
    Returns:
        정규화된 주소 (소문자)
    """
    if address and isinstance(address, str):
        return address.lower()
    return address 
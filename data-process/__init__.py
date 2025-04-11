"""
블록체인 데이터 처리 모듈 (data-process)

이 패키지는 이더리움 블록체인 데이터를 수집, 처리 및 저장하기 위한 
통합 파이프라인을 제공합니다.
"""

import os
import logging

# 모듈 버전
__version__ = '0.1.0'

# 패키지 경로
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# 기본 데이터 디렉토리 설정
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser('~'), '.blockchain_data')

# 각 데이터 유형별 하위 디렉토리
RAW_DATA_DIR = 'raw'
PROCESSED_DATA_DIR = 'processed'
VECTORS_DATA_DIR = 'vectors'
CACHE_DIR = 'cache'

# 데이터베이스 파일명
DB_FILENAME = 'data_index.db'

# 로거 설정
logger = logging.getLogger('data_process')

# 모듈 초기화 메시지
logger.info(f"블록체인 데이터 처리 모듈 초기화 (버전 {__version__})")

# 공개 API 설정
from .blockchain_data import BlockchainDataCollector
from .data_processor import DataProcessor
from .data_storage import DataStorage
from .utils import setup_logging, validate_ethereum_address

__all__ = [
    'BlockchainDataCollector',
    'DataProcessor',
    'DataStorage',
    'setup_logging',
    'validate_ethereum_address',
    'DEFAULT_DATA_DIR',
    'RAW_DATA_DIR',
    'PROCESSED_DATA_DIR',
    'VECTORS_DATA_DIR',
    'CACHE_DIR',
    'DB_FILENAME'
] 
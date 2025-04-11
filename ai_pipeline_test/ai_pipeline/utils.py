"""
유틸리티 함수
"""

import logging
import os
from datetime import datetime

def setup_logger(name):
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # 콘솔 핸들러
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
    
    return logger 
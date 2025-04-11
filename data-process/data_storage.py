"""
블록체인 데이터 저장 및 관리 모듈

수집 및 처리된 데이터의 저장, 관리, 검색 기능 제공
"""

import os
import json
import shutil
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging

from .utils import setup_logger, ensure_directory, save_json, load_json, generate_data_hash

logger = setup_logger('data_storage')

class DataStorage:
    """블록체인 데이터 저장 및 관리 클래스"""
    
    def __init__(self, data_dir: str = "data", db_path: str = "data/storage.db"):
        """
        데이터 저장소 초기화
        
        Args:
            data_dir: 데이터 기본 디렉토리
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.data_dir = ensure_directory(data_dir)
        self.raw_data_dir = ensure_directory(os.path.join(data_dir, "raw"))
        self.processed_data_dir = ensure_directory(os.path.join(data_dir, "processed"))
        self.vectors_dir = ensure_directory(os.path.join(data_dir, "vectors"))
        self.cache_dir = ensure_directory(os.path.join(data_dir, "cache"))
        
        # 데이터베이스 파일 경로
        db_dir = os.path.dirname(db_path)
        if db_dir:
            ensure_directory(db_dir)
        self.db_path = db_path
        
        # 데이터베이스 초기화
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화 및 테이블 생성"""
        try:
            # 데이터베이스 연결
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 주소 메타데이터 테이블
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS address_data (
                address TEXT PRIMARY KEY,
                first_seen_at TEXT,
                last_updated_at TEXT,
                raw_count INTEGER DEFAULT 0,
                processed_count INTEGER DEFAULT 0,
                feature_vector_path TEXT,
                raw_data_path TEXT,
                processed_data_path TEXT,
                data_hash TEXT
            )
            ''')
            
            # 데이터 파일 인덱스 테이블
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                address TEXT,
                file_path TEXT,
                file_type TEXT,
                created_at TEXT,
                file_size INTEGER,
                data_hash TEXT,
                is_latest BOOLEAN,
                FOREIGN KEY (address) REFERENCES address_data(address)
            )
            ''')
            
            # 변경 적용 및 연결 종료
            conn.commit()
            conn.close()
            
            logger.info("데이터베이스 초기화 완료")
            
        except sqlite3.Error as e:
            logger.error(f"데이터베이스 초기화 오류: {str(e)}")
    
    def register_address(self, address: str) -> bool:
        """
        신규 주소 등록 또는 기존 주소 정보 업데이트
        
        Args:
            address: 이더리움 주소
            
        Returns:
            성공 여부
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 주소가 이미 존재하는지 확인
            cursor.execute("SELECT address FROM address_data WHERE address=?", (address.lower(),))
            result = cursor.fetchone()
            
            current_time = datetime.now().isoformat()
            
            if result:
                # 기존 주소 정보 업데이트
                cursor.execute(
                    "UPDATE address_data SET last_updated_at=? WHERE address=?",
                    (current_time, address.lower())
                )
                logger.info(f"주소 정보 업데이트: {address}")
            else:
                # 신규 주소 등록
                cursor.execute(
                    "INSERT INTO address_data (address, first_seen_at, last_updated_at) VALUES (?, ?, ?)",
                    (address.lower(), current_time, current_time)
                )
                logger.info(f"신규 주소 등록: {address}")
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"주소 등록 오류: {str(e)}")
            return False
    
    def register_data_file(self, address: str, file_path: str, file_type: str, 
                          data_hash: str = "", is_latest: bool = True) -> bool:
        """
        데이터 파일 등록
        
        Args:
            address: 이더리움 주소
            file_path: 파일 경로
            file_type: 파일 유형 (raw, processed, vector)
            data_hash: 데이터 해시 (선택사항)
            is_latest: 최신 파일 여부
            
        Returns:
            성공 여부
        """
        try:
            # 파일 크기 확인
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 파일이 최신인 경우 기존 최신 파일 상태를 업데이트
            if is_latest:
                cursor.execute(
                    "UPDATE data_files SET is_latest=0 WHERE address=? AND file_type=? AND is_latest=1",
                    (address.lower(), file_type)
                )
            
            # 새 파일 등록
            created_at = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO data_files 
                (address, file_path, file_type, created_at, file_size, data_hash, is_latest)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (address.lower(), file_path, file_type, created_at, file_size, data_hash, is_latest)
            )
            
            # 주소 메타데이터 업데이트
            if is_latest:
                if file_type == 'raw':
                    update_field = "raw_data_path=?, raw_count=raw_count+1"
                elif file_type == 'processed':
                    update_field = "processed_data_path=?, processed_count=processed_count+1"
                elif file_type == 'vector':
                    update_field = "feature_vector_path=?"
                else:
                    update_field = ""
                
                if update_field:
                    cursor.execute(
                        f"UPDATE address_data SET {update_field}, data_hash=?, last_updated_at=? WHERE address=?",
                        (file_path, data_hash, created_at, address.lower())
                    )
            
            conn.commit()
            conn.close()
            logger.info(f"데이터 파일 등록 완료: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"데이터 파일 등록 오류: {str(e)}")
            return False
    
    def get_latest_data_path(self, address: str, data_type: str = "processed") -> str:
        """
        주소의 최신 데이터 파일 경로 조회
        
        Args:
            address: 이더리움 주소
            data_type: 데이터 유형 (raw, processed, vector)
            
        Returns:
            파일 경로 또는 빈 문자열
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT file_path FROM data_files WHERE address=? AND file_type=? AND is_latest=1",
                (address.lower(), data_type)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else ""
            
        except sqlite3.Error as e:
            logger.error(f"데이터 파일 조회 오류: {str(e)}")
            return ""
    
    def get_address_data_info(self, address: str) -> Dict[str, Any]:
        """
        주소 데이터 정보 조회
        
        Args:
            address: 이더리움 주소
            
        Returns:
            주소 데이터 정보
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM address_data WHERE address=?",
                (address.lower(),)
            )
            
            result = cursor.fetchone()
            if not result:
                return {}
            
            # 컬럼명 가져오기
            columns = [desc[0] for desc in cursor.description]
            
            # 결과를 딕셔너리로 변환
            info = dict(zip(columns, result))
            
            # 파일 개수 정보 추가
            cursor.execute(
                "SELECT file_type, COUNT(*) FROM data_files WHERE address=? GROUP BY file_type",
                (address.lower(),)
            )
            
            file_counts = {}
            for file_type, count in cursor.fetchall():
                file_counts[f"{file_type}_files"] = count
            
            info.update(file_counts)
            conn.close()
            
            return info
            
        except sqlite3.Error as e:
            logger.error(f"주소 정보 조회 오류: {str(e)}")
            return {}
    
    def list_addresses(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        등록된 주소 목록 조회
        
        Args:
            limit: 최대 조회 개수
            offset: 조회 시작 오프셋
            
        Returns:
            주소 정보 목록
        """
        try:
            conn = sqlite3.connect(self.db_path)
            # 행 결과를 딕셔너리로 가져오도록 설정
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT address, first_seen_at, last_updated_at, raw_count, processed_count 
                FROM address_data ORDER BY last_updated_at DESC LIMIT ? OFFSET ?
                """,
                (limit, offset)
            )
            
            # 결과를 딕셔너리 리스트로 변환
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"주소 목록 조회 오류: {str(e)}")
            return []
    
    def store_processed_data(self, address: str, data: Dict[str, Any], 
                           file_name: str = "") -> Tuple[bool, str]:
        """
        처리된 데이터 저장
        
        Args:
            address: 이더리움 주소
            data: 저장할 데이터
            file_name: 파일명 (기본값: 자동 생성)
            
        Returns:
            (성공 여부, 파일 경로)
        """
        try:
            # 주소 등록
            self.register_address(address)
            
            # 주소별 디렉토리 생성
            addr_short = address[:8].lower()
            addr_dir = ensure_directory(os.path.join(self.processed_data_dir, addr_short))
            
            # 파일명 생성 또는 사용
            if not file_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"{addr_short}_{timestamp}_processed.json"
            
            file_path = os.path.join(addr_dir, file_name)
            
            # 데이터 해시 생성
            data_hash = generate_data_hash(data)
            
            # 데이터 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # 파일 등록
            self.register_data_file(address, file_path, "processed", data_hash)
            
            logger.info(f"처리된 데이터 저장 완료: {file_path}")
            return True, file_path
            
        except Exception as e:
            logger.error(f"데이터 저장 오류: {str(e)}")
            return False, ""
    
    def store_feature_vector(self, address: str, vector_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        특성 벡터 데이터 저장
        
        Args:
            address: 이더리움 주소
            vector_data: 특성 벡터 데이터
            
        Returns:
            (성공 여부, 파일 경로)
        """
        try:
            # 주소 등록
            self.register_address(address)
            
            # 주소별 디렉토리 생성
            addr_short = address[:8].lower()
            vector_dir = ensure_directory(os.path.join(self.vectors_dir, addr_short))
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{addr_short}_{timestamp}_vector.json"
            file_path = os.path.join(vector_dir, file_name)
            
            # 데이터 해시 생성
            data_hash = generate_data_hash(vector_data)
            
            # 데이터 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(vector_data, f, indent=2, ensure_ascii=False)
            
            # 파일 등록
            self.register_data_file(address, file_path, "vector", data_hash)
            
            logger.info(f"특성 벡터 저장 완료: {file_path}")
            return True, file_path
            
        except Exception as e:
            logger.error(f"특성 벡터 저장 오류: {str(e)}")
            return False, ""
    
    def load_data(self, address: str, data_type: str = "processed") -> Dict[str, Any]:
        """
        주소의 최신 데이터 로드
        
        Args:
            address: 이더리움 주소
            data_type: 데이터 유형 (raw, processed, vector)
            
        Returns:
            로드된 데이터
        """
        file_path = self.get_latest_data_path(address, data_type)
        
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"데이터 파일을 찾을 수 없음: {address}, {data_type}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
            
        except Exception as e:
            logger.error(f"데이터 로드 오류: {str(e)}")
            return {}
    
    def export_to_dataframe(self, addresses: List[str] = None) -> pd.DataFrame:
        """
        여러 주소의 특성 벡터를 데이터프레임으로 변환
        
        Args:
            addresses: 주소 목록 (기본값: 모든 주소)
            
        Returns:
            pandas DataFrame
        """
        try:
            all_vectors = []
            
            # 주소 목록이 없으면 모든 주소 사용
            if not addresses:
                address_info = self.list_addresses(limit=1000)
                addresses = [info['address'] for info in address_info]
            
            # 각 주소의 특성 벡터 수집
            for address in addresses:
                vector_data = self.load_data(address, "vector")
                
                if not vector_data or 'feature_vector' not in vector_data:
                    continue
                
                # 기본 정보 추가
                row = {'address': address}
                
                # 특성 벡터 추가
                row.update(vector_data.get('feature_vector', {}))
                
                all_vectors.append(row)
            
            # 데이터프레임 생성
            df = pd.DataFrame(all_vectors)
            return df
            
        except Exception as e:
            logger.error(f"데이터프레임 변환 오류: {str(e)}")
            return pd.DataFrame()
    
    def clear_old_data(self, address: str = None, data_type: str = None, 
                      keep_latest: int = 1) -> bool:
        """
        오래된 데이터 파일 정리
        
        Args:
            address: 이더리움 주소 (기본값: 모든 주소)
            data_type: 데이터 유형 (기본값: 모든 유형)
            keep_latest: 보존할 최신 파일 수
            
        Returns:
            성공 여부
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 조건 쿼리 구성
            conditions = []
            params = []
            
            if address:
                conditions.append("address = ?")
                params.append(address.lower())
            
            if data_type:
                conditions.append("file_type = ?")
                params.append(data_type)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # 삭제할 파일 조회
            query = f"""
            SELECT df.file_id, df.file_path FROM data_files df
            WHERE df.{where_clause} AND df.file_id NOT IN (
                SELECT df2.file_id FROM data_files df2
                WHERE df2.address = df.address AND df2.file_type = df.file_type
                ORDER BY df2.created_at DESC
                LIMIT {keep_latest}
            )
            """
            
            cursor.execute(query, params)
            files_to_delete = cursor.fetchall()
            
            # 파일 삭제
            for file_id, file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"파일 삭제: {file_path}")
                
                # 데이터베이스에서 제거
                cursor.execute("DELETE FROM data_files WHERE file_id = ?", (file_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"데이터 정리 완료, {len(files_to_delete)}개 파일 삭제됨")
            return True
            
        except Exception as e:
            logger.error(f"데이터 정리 오류: {str(e)}")
            return False 
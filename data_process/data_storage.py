#!/usr/bin/env python3
"""
데이터 저장 모듈
처리된 블록체인 데이터를 로컬 DB에 저장하고 관리
"""

import os
import json
import sqlite3
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

class DataStorage:
    """데이터 저장 및 캐싱을 담당하는 클래스"""
    
    def __init__(self, data_dir="./cache", db_path="./cache/data_storage.db"):
        """
        초기화 함수
        
        Args:
            data_dir: 데이터 디렉토리
            db_path: SQLite DB 파일 경로
        """
        self.data_dir = data_dir
        self.db_path = db_path
        
        # 데이터 디렉토리 생성
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # DB 초기화
        self._init_db()
    
    def _init_db(self):
        """SQLite DB 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 주소 데이터 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS address_data (
            address TEXT,
            chain_id TEXT,
            data_json TEXT,
            last_updated DATETIME,
            PRIMARY KEY (address, chain_id)
        )
        ''')
        
        # 특성 데이터 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_data (
            address TEXT,
            chain_id TEXT,
            features_json TEXT,
            last_updated DATETIME,
            PRIMARY KEY (address, chain_id)
        )
        ''')
        
        # 분석 결과 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            address TEXT,
            chain_id TEXT,
            results_json TEXT,
            cluster INTEGER,
            user_type TEXT,
            last_updated DATETIME,
            PRIMARY KEY (address, chain_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_address_data(self, address: str, chain_id: str, data: Dict[str, Any]) -> bool:
        """
        주소 데이터 저장
        
        Args:
            address: 블록체인 주소
            chain_id: 체인 ID
            data: 저장할 데이터
            
        Returns:
            성공 여부
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 현재 시간
            now = datetime.now().isoformat()
            
            # 데이터를 JSON으로 직렬화
            data_json = json.dumps(data)
            
            # 데이터 삽입 또는 업데이트
            cursor.execute(
                '''
                INSERT OR REPLACE INTO address_data (address, chain_id, data_json, last_updated)
                VALUES (?, ?, ?, ?)
                ''',
                (address.lower(), chain_id, data_json, now)
            )
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            print(f"주소 데이터 저장 오류: {str(e)}")
            return False
    
    def get_address_data(self, address: str, chain_id: str, max_age_hours: int = 24) -> Optional[Dict[str, Any]]:
        """
        주소 데이터 가져오기
        
        Args:
            address: 블록체인 주소
            chain_id: 체인 ID
            max_age_hours: 최대 캐시 유효 시간 (시간 단위)
            
        Returns:
            저장된 데이터 또는 None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 데이터 조회
            cursor.execute(
                '''
                SELECT data_json, last_updated FROM address_data
                WHERE address = ? AND chain_id = ?
                ''',
                (address.lower(), chain_id)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                data_json, last_updated = result
                
                # 캐시 유효 시간 확인
                if max_age_hours > 0:
                    try:
                        last_updated_dt = datetime.fromisoformat(last_updated)
                        now = datetime.now()
                        age_hours = (now - last_updated_dt).total_seconds() / 3600
                        
                        if age_hours > max_age_hours:
                            return None  # 캐시 만료
                    except:
                        pass  # 날짜 파싱 오류, 계속 진행
                
                # JSON 역직렬화
                return json.loads(data_json)
            
            return None
        
        except Exception as e:
            print(f"주소 데이터 조회 오류: {str(e)}")
            return None
    
    def save_feature_data(self, address: str, chain_id: str, features: Dict[str, Any]) -> bool:
        """
        특성 데이터 저장
        
        Args:
            address: 블록체인 주소
            chain_id: 체인 ID
            features: 특성 데이터
            
        Returns:
            성공 여부
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 현재 시간
            now = datetime.now().isoformat()
            
            # 데이터를 JSON으로 직렬화
            features_json = json.dumps(features)
            
            # 데이터 삽입 또는 업데이트
            cursor.execute(
                '''
                INSERT OR REPLACE INTO feature_data (address, chain_id, features_json, last_updated)
                VALUES (?, ?, ?, ?)
                ''',
                (address.lower(), chain_id, features_json, now)
            )
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            print(f"특성 데이터 저장 오류: {str(e)}")
            return False
    
    def get_feature_data(self, address: str, chain_id: str) -> Optional[Dict[str, Any]]:
        """
        특성 데이터 가져오기
        
        Args:
            address: 블록체인 주소
            chain_id: 체인 ID
            
        Returns:
            특성 데이터 또는 None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 데이터 조회
            cursor.execute(
                '''
                SELECT features_json FROM feature_data
                WHERE address = ? AND chain_id = ?
                ''',
                (address.lower(), chain_id)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                features_json = result[0]
                # JSON 역직렬화
                return json.loads(features_json)
            
            return None
        
        except Exception as e:
            print(f"특성 데이터 조회 오류: {str(e)}")
            return None
    
    def save_analysis_result(self, address: str, chain_id: str, results: Dict[str, Any], 
                          cluster: int = -1, user_type: str = "unknown") -> bool:
        """
        분석 결과 저장
        
        Args:
            address: 블록체인 주소
            chain_id: 체인 ID
            results: 분석 결과
            cluster: 클러스터 ID
            user_type: 사용자 유형
            
        Returns:
            성공 여부
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 현재 시간
            now = datetime.now().isoformat()
            
            # 데이터를 JSON으로 직렬화
            results_json = json.dumps(results)
            
            # 데이터 삽입 또는 업데이트
            cursor.execute(
                '''
                INSERT OR REPLACE INTO analysis_results (address, chain_id, results_json, cluster, user_type, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (address.lower(), chain_id, results_json, cluster, user_type, now)
            )
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            print(f"분석 결과 저장 오류: {str(e)}")
            return False
    
    def get_analysis_result(self, address: str, chain_id: str) -> Optional[Dict[str, Any]]:
        """
        분석 결과 가져오기
        
        Args:
            address: 블록체인 주소
            chain_id: 체인 ID
            
        Returns:
            분석 결과 또는 None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 데이터 조회
            cursor.execute(
                '''
                SELECT results_json, cluster, user_type, last_updated FROM analysis_results
                WHERE address = ? AND chain_id = ?
                ''',
                (address.lower(), chain_id)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                results_json, cluster, user_type, last_updated = result
                
                # JSON 역직렬화
                analysis_data = json.loads(results_json)
                
                # 추가 메타데이터
                analysis_data["cluster"] = cluster
                analysis_data["user_type"] = user_type
                analysis_data["last_updated"] = last_updated
                
                return analysis_data
            
            return None
        
        except Exception as e:
            print(f"분석 결과 조회 오류: {str(e)}")
            return None 
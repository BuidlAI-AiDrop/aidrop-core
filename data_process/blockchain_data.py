#!/usr/bin/env python3
"""
블록체인 데이터 수집 모듈
Supabase에서 블록체인 거래 데이터를 가져오는 기능 구현
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from supabase import create_client, Client

class BlockchainDataCollector:
    """Supabase에서 블록체인 데이터를 수집하는 클래스"""
    
    def __init__(self, api_key=None, data_dir="./cache/raw"):
        """
        초기화 함수
        
        Args:
            api_key: Supabase API 키 (없으면 환경 변수에서 가져옴)
            data_dir: 데이터 캐시 디렉토리
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Supabase 클라이언트 초기화
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.api_key = api_key or os.environ.get("SUPABASE_KEY")
        
        self.supabase = None
        if self.supabase_url and self.api_key:
            try:
                self.supabase = create_client(self.supabase_url, self.api_key)
                print("Supabase 클라이언트가 성공적으로 초기화되었습니다.")
            except Exception as e:
                print(f"Supabase 클라이언트 초기화 오류: {str(e)}")
                print("테스트 데이터를 사용합니다.")
    
    def get_address_transactions(self, address: str, chain_id: str = "1", force_refresh: bool = False) -> Dict[str, Any]:
        """
        주소에 대한 트랜잭션 데이터 가져오기
        
        Args:
            address: 블록체인 주소
            chain_id: 체인 ID (기본값: "1", Ethereum 메인넷)
            force_refresh: 캐시를 무시하고 새로 가져올지 여부
            
        Returns:
            트랜잭션 데이터 딕셔너리
        """
        # 캐시 확인
        cache_file = os.path.join(self.data_dir, f"{address.lower()}_{chain_id}.json")
        if not force_refresh and os.path.exists(cache_file):
            # 캐시가 24시간 이내인 경우에만 사용
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < 86400:  # 24시간 (초 단위)
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # Supabase에서 데이터 가져오기
        if self.supabase:
            try:
                # 트랜잭션 데이터 쿼리
                query = self.supabase.table("blockchain_transactions")
                query = query.select("*")
                
                # sender 또는 receiver가 address인 데이터
                # 필터링 방식은 or_와 .eq 조합 사용
                sender_filter = self.supabase.table("blockchain_transactions").select("*").eq("sender", address)
                receiver_filter = self.supabase.table("blockchain_transactions").select("*").eq("receiver", address)
                
                # 결과 조합 쿼리 작성
                combined_response = []
                sender_response = sender_filter.eq("chain_id", chain_id).order("timestamp", desc=True).limit(500).execute()
                receiver_response = receiver_filter.eq("chain_id", chain_id).order("timestamp", desc=True).limit(500).execute()
                
                # 결과 병합
                if sender_response.data:
                    combined_response.extend(sender_response.data)
                if receiver_response.data:
                    combined_response.extend(receiver_response.data)
                
                # 토큰 거래 데이터 쿼리
                # 같은 방식으로 분리 쿼리
                from_filter = self.supabase.table("token_transfers").select("*").eq("from_address", address)
                to_filter = self.supabase.table("token_transfers").select("*").eq("to_address", address)
                
                # 결과 조합 쿼리 작성
                token_combined = []
                from_response = from_filter.eq("chain_id", chain_id).order("timestamp", desc=True).limit(500).execute()
                to_response = to_filter.eq("chain_id", chain_id).order("timestamp", desc=True).limit(500).execute()
                
                # 결과 병합
                if from_response.data:
                    token_combined.extend(from_response.data)
                if to_response.data:
                    token_combined.extend(to_response.data)
                
                # 컨트랙트 인터랙션 데이터 쿼리
                contract_query = self.supabase.table("contract_interactions")
                contract_query = contract_query.select("*")
                contract_query = contract_query.eq("user_address", address)
                contract_query = contract_query.eq("chain_id", chain_id)
                contract_query = contract_query.order("timestamp", desc=True)
                contract_query = contract_query.limit(1000)
                
                contract_response = contract_query.execute()
                
                # 결과 조합
                result = {
                    "address": address,
                    "chain_id": chain_id,
                    "transactions": combined_response,  # 병합된 결과 사용
                    "token_transfers": token_combined,  # 병합된 결과 사용
                    "contract_interactions": contract_response.data if contract_response.data else [],
                    "last_updated": datetime.now().isoformat()
                }
                
                # 캐시에 저장
                with open(cache_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                return result
            
            except Exception as e:
                print(f"Supabase 쿼리 오류: {str(e)}")
                print("테스트 데이터를 사용합니다.")
        
        # Supabase 연결 실패 또는 오류 발생 시 테스트 데이터 사용
        test_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data/blockchain_test_data.json")
        if os.path.exists(test_data_path):
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)
                if address in test_data:
                    return test_data[address]
                else:
                    # 첫 번째 주소 데이터 반환 (테스트용)
                    for addr, data in test_data.items():
                        return {
                            "address": address,
                            "chain_id": chain_id,
                            "transactions": data.get("transactions", []),
                            "token_transfers": data.get("token_transfers", []),
                            "contract_interactions": data.get("contract_interactions", []),
                            "last_updated": datetime.now().isoformat()
                        }
        
        # 빈 데이터 반환
        return {
            "address": address,
            "chain_id": chain_id,
            "transactions": [],
            "token_transfers": [],
            "contract_interactions": [],
            "last_updated": datetime.now().isoformat()
        }
    
    def get_token_balances(self, address: str, chain_id: str = "1") -> List[Dict[str, Any]]:
        """
        주소에 대한 토큰 잔액 데이터 가져오기
        
        Args:
            address: 블록체인 주소
            chain_id: 체인 ID (기본값: "1", Ethereum 메인넷)
            
        Returns:
            토큰 잔액 리스트
        """
        if self.supabase:
            try:
                query = self.supabase.table("token_balances")
                query = query.select("*")
                query = query.eq("address", address)
                query = query.eq("chain_id", chain_id)
                
                response = query.execute()
                return response.data
            
            except Exception as e:
                print(f"토큰 잔액 쿼리 오류: {str(e)}")
        
        # 테스트 데이터 또는 빈 배열 반환
        return []
    
    def get_nft_holdings(self, address: str, chain_id: str = "1") -> List[Dict[str, Any]]:
        """
        주소에 대한 NFT 보유 데이터 가져오기
        
        Args:
            address: 블록체인 주소
            chain_id: 체인 ID (기본값: "1", Ethereum 메인넷)
            
        Returns:
            NFT 보유 리스트
        """
        if self.supabase:
            try:
                query = self.supabase.table("nft_holdings")
                query = query.select("*")
                query = query.eq("owner_address", address)
                query = query.eq("chain_id", chain_id)
                
                response = query.execute()
                return response.data
            
            except Exception as e:
                print(f"NFT 보유 쿼리 오류: {str(e)}")
        
        # 테스트 데이터 또는 빈 배열 반환
        return [] 
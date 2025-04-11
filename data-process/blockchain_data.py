"""
블록체인 데이터 수집 모듈

이더리움 블록체인에서 지갑 데이터를 수집하는 기능 제공
"""

import os
import time
import json
import logging
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

from . import utils

logger = logging.getLogger('data_process')

class BlockchainDataCollector:
    """
    블록체인 데이터 수집기 클래스
    
    이더리움 블록체인에서 지갑 주소 관련 데이터 수집 기능 제공
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.etherscan.io/api",
        rate_limit: float = 0.2,  # 초당 5회 요청 제한 (기본값)
        data_dir: Optional[str] = None,
        max_workers: int = 5
    ):
        """
        블록체인 데이터 수집기 초기화
        
        Args:
            api_key: Etherscan API 키 (환경 변수에서 가져올 수 있음)
            base_url: API 기본 URL
            rate_limit: API 요청 간격 (초)
            data_dir: 데이터 저장 디렉토리
            max_workers: 동시 요청 최대 스레드 수
        """
        self.api_key = api_key or os.environ.get('ETHERSCAN_API_KEY', '')
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.data_dir = data_dir
        self.max_workers = max_workers
        self.last_request_time = 0
        
        if not self.api_key:
            logger.warning("API 키가 설정되지 않았습니다. 일부 기능이 제한될 수 있습니다.")
    
    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        API 요청 메서드
        
        Args:
            endpoint: API 엔드포인트
            params: 요청 파라미터
            
        Returns:
            API 응답 데이터
        """
        # API 키 추가
        if self.api_key:
            params['apikey'] = self.api_key
        
        # 속도 제한 준수
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last_request
            logger.debug(f"속도 제한 준수를 위해 {sleep_time:.2f}초 대기")
            time.sleep(sleep_time)
        
        # 요청 수행
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"API 요청: {url}, 파라미터: {params}")
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            self.last_request_time = time.time()
            
            # JSON 응답 파싱
            data = response.json()
            
            # 응답 유효성 검증
            is_valid, error_msg = utils.validate_api_response(data)
            if not is_valid:
                logger.error(f"API 응답 오류: {error_msg}")
                return {"status": "0", "message": error_msg, "result": []}
            
            return data
        except requests.RequestException as e:
            logger.error(f"API 요청 실패: {str(e)}")
            return {"status": "0", "message": str(e), "result": []}
    
    def get_account_balance(self, address: str) -> Dict[str, Any]:
        """
        계정 잔액 조회
        
        Args:
            address: 이더리움 주소
            
        Returns:
            계정 잔액 데이터
        """
        # 주소 유효성 검증
        if not utils.validate_ethereum_address(address):
            logger.error(f"유효하지 않은 이더리움 주소: {address}")
            return {"status": "0", "message": "유효하지 않은 주소", "result": "0"}
        
        # API 요청 파라미터
        params = {
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest"
        }
        
        return self._make_request("", params)
    
    def get_transactions(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100,
        sort: str = "desc"
    ) -> Dict[str, Any]:
        """
        계정 트랜잭션 목록 조회
        
        Args:
            address: 이더리움 주소
            start_block: 시작 블록 번호
            end_block: 종료 블록 번호
            page: 페이지 번호
            offset: 페이지당 결과 수
            sort: 정렬 방식 (asc/desc)
            
        Returns:
            트랜잭션 목록 데이터
        """
        # 주소 유효성 검증
        if not utils.validate_ethereum_address(address):
            logger.error(f"유효하지 않은 이더리움 주소: {address}")
            return {"status": "0", "message": "유효하지 않은 주소", "result": []}
        
        # API 요청 파라미터
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": offset,
            "sort": sort
        }
        
        return self._make_request("", params)
    
    def get_token_transfers(
        self,
        address: str,
        contract_address: Optional[str] = None,
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100,
        sort: str = "desc"
    ) -> Dict[str, Any]:
        """
        ERC-20 토큰 전송 내역 조회
        
        Args:
            address: 이더리움 주소
            contract_address: 토큰 컨트랙트 주소 (특정 토큰만 조회 시)
            start_block: 시작 블록 번호
            end_block: 종료 블록 번호
            page: 페이지 번호
            offset: 페이지당 결과 수
            sort: 정렬 방식 (asc/desc)
            
        Returns:
            토큰 전송 내역 데이터
        """
        # 주소 유효성 검증
        if not utils.validate_ethereum_address(address):
            logger.error(f"유효하지 않은 이더리움 주소: {address}")
            return {"status": "0", "message": "유효하지 않은 주소", "result": []}
        
        # API 요청 파라미터
        params = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": offset,
            "sort": sort
        }
        
        # 특정 토큰 컨트랙트 주소 추가
        if contract_address and utils.validate_ethereum_address(contract_address):
            params["contractaddress"] = contract_address
        
        return self._make_request("", params)
    
    def get_nft_transfers(
        self,
        address: str,
        contract_address: Optional[str] = None,
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100,
        sort: str = "desc"
    ) -> Dict[str, Any]:
        """
        ERC-721 (NFT) 전송 내역 조회
        
        Args:
            address: 이더리움 주소
            contract_address: NFT 컨트랙트 주소 (특정 NFT만 조회 시)
            start_block: 시작 블록 번호
            end_block: 종료 블록 번호
            page: 페이지 번호
            offset: 페이지당 결과 수
            sort: 정렬 방식 (asc/desc)
            
        Returns:
            NFT 전송 내역 데이터
        """
        # 주소 유효성 검증
        if not utils.validate_ethereum_address(address):
            logger.error(f"유효하지 않은 이더리움 주소: {address}")
            return {"status": "0", "message": "유효하지 않은 주소", "result": []}
        
        # API 요청 파라미터
        params = {
            "module": "account",
            "action": "tokennfttx",
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": offset,
            "sort": sort
        }
        
        # 특정 NFT 컨트랙트 주소 추가
        if contract_address and utils.validate_ethereum_address(contract_address):
            params["contractaddress"] = contract_address
        
        return self._make_request("", params)
    
    def get_internal_transactions(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100,
        sort: str = "desc"
    ) -> Dict[str, Any]:
        """
        내부 트랜잭션 내역 조회
        
        Args:
            address: 이더리움 주소
            start_block: 시작 블록 번호
            end_block: 종료 블록 번호
            page: 페이지 번호
            offset: 페이지당 결과 수
            sort: 정렬 방식 (asc/desc)
            
        Returns:
            내부 트랜잭션 내역 데이터
        """
        # 주소 유효성 검증
        if not utils.validate_ethereum_address(address):
            logger.error(f"유효하지 않은 이더리움 주소: {address}")
            return {"status": "0", "message": "유효하지 않은 주소", "result": []}
        
        # API 요청 파라미터
        params = {
            "module": "account",
            "action": "txlistinternal",
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": offset,
            "sort": sort
        }
        
        return self._make_request("", params)
    
    def collect_wallet_data(
        self,
        address: str,
        save_data: bool = True
    ) -> Dict[str, Any]:
        """
        지갑 주소에 대한 모든 데이터 수집
        
        Args:
            address: 이더리움 주소
            save_data: 데이터 저장 여부
            
        Returns:
            수집된 지갑 데이터
        """
        # 주소 정규화
        address = utils.normalize_address(address)
        
        # 주소 유효성 검증
        if not utils.validate_ethereum_address(address):
            logger.error(f"유효하지 않은 이더리움 주소: {address}")
            return {"status": "error", "message": "유효하지 않은 주소", "data": None}
        
        logger.info(f"지갑 데이터 수집 시작: {address}")
        
        # 여러 API 요청을 병렬로 수행
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 각 데이터 유형별 요청 설정
            balance_future = executor.submit(self.get_account_balance, address)
            transactions_future = executor.submit(self.get_transactions, address)
            token_transfers_future = executor.submit(self.get_token_transfers, address)
            nft_transfers_future = executor.submit(self.get_nft_transfers, address)
            internal_tx_future = executor.submit(self.get_internal_transactions, address)
            
            # 결과 수집
            balance_data = balance_future.result()
            transactions_data = transactions_future.result()
            token_transfers_data = token_transfers_future.result()
            nft_transfers_data = nft_transfers_future.result()
            internal_tx_data = internal_tx_future.result()
        
        # 수집된 데이터 병합
        wallet_data = {
            "address": address,
            "timestamp": utils.get_timestamp(),
            "balance": balance_data.get("result", "0"),
            "transactions": transactions_data.get("result", []),
            "token_transfers": token_transfers_data.get("result", []),
            "nft_transfers": nft_transfers_data.get("result", []),
            "internal_transactions": internal_tx_data.get("result", [])
        }
        
        # 데이터 저장
        if save_data and self.data_dir:
            # 주소 해시 기반 식별자 생성
            address_hash = utils.generate_hash(address)
            
            # 파일 경로 생성
            filepath = utils.get_data_filepath(
                self.data_dir,
                address_hash,
                "raw"
            )
            
            # 데이터 저장
            if utils.save_json(wallet_data, filepath):
                logger.info(f"지갑 데이터 저장 완료: {filepath}")
            else:
                logger.error(f"지갑 데이터 저장 실패: {filepath}")
        
        logger.info(f"지갑 데이터 수집 완료: {address}")
        return {
            "status": "success",
            "message": "지갑 데이터 수집 성공",
            "data": wallet_data
        }
    
    def collect_multiple_wallets(
        self,
        addresses: List[str],
        save_data: bool = True
    ) -> Dict[str, Any]:
        """
        여러 지갑 주소에 대한 데이터 수집
        
        Args:
            addresses: 이더리움 주소 목록
            save_data: 데이터 저장 여부
            
        Returns:
            수집 결과 요약
        """
        logger.info(f"다중 지갑 데이터 수집 시작: {len(addresses)}개 주소")
        
        results = {
            "success": 0,
            "failed": 0,
            "details": {}
        }
        
        # 각 주소별로 데이터 수집
        for address in addresses:
            try:
                # 주소 정규화
                normalized_address = utils.normalize_address(address)
                
                # 데이터 수집
                result = self.collect_wallet_data(normalized_address, save_data)
                
                # 결과 기록
                if result["status"] == "success":
                    results["success"] += 1
                    results["details"][normalized_address] = "성공"
                else:
                    results["failed"] += 1
                    results["details"][normalized_address] = result["message"]
            
            except Exception as e:
                logger.error(f"지갑 데이터 수집 중 오류 발생: {address}, {str(e)}")
                results["failed"] += 1
                results["details"][address] = str(e)
        
        logger.info(f"다중 지갑 데이터 수집 완료: 성공 {results['success']}개, 실패 {results['failed']}개")
        return results 
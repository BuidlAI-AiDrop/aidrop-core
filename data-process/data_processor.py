"""
블록체인 데이터 처리 모듈

수집된 블록체인 데이터를 정제하고 특성을 추출하는 기능 제공
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

from . import utils

logger = logging.getLogger('data_process')

class BlockchainDataProcessor:
    """
    블록체인 데이터 처리기 클래스
    
    수집된 이더리움 블록체인 데이터 정제 및 특성 추출 기능 제공
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        processed_dir: Optional[str] = None
    ):
        """
        블록체인 데이터 처리기 초기화
        
        Args:
            data_dir: 원시 데이터 디렉토리
            processed_dir: 처리된 데이터 저장 디렉토리
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir or (data_dir and os.path.join(data_dir, 'processed'))
        
        # 디렉토리 생성
        if self.processed_dir:
            utils.ensure_directory(self.processed_dir)
    
    def load_raw_data(
        self,
        address_or_file: str
    ) -> Dict[str, Any]:
        """
        원시 데이터 로드
        
        Args:
            address_or_file: 이더리움 주소 또는 파일 경로
            
        Returns:
            로드된 원시 데이터
        """
        if os.path.isfile(address_or_file):
            # 파일 경로인 경우 직접 로드
            file_path = address_or_file
        else:
            # 주소인 경우 해시 생성 후 파일 탐색
            address = utils.normalize_address(address_or_file)
            address_hash = utils.generate_hash(address)
            
            # 파일 경로 생성
            file_path = utils.get_data_filepath(
                self.data_dir,
                address_hash,
                "raw"
            )
        
        # 데이터 로드
        data = utils.load_json(file_path)
        if not data:
            logger.error(f"원시 데이터 로드 실패: {file_path}")
            return {}
        
        return data
    
    def clean_transaction_data(
        self,
        transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        트랜잭션 데이터 정제
        
        Args:
            transactions: 원시 트랜잭션 목록
            
        Returns:
            정제된 트랜잭션 목록
        """
        cleaned_txs = []
        
        for tx in transactions:
            try:
                # 필수 필드 확인
                if not all(k in tx for k in ['hash', 'from', 'to', 'value', 'timeStamp']):
                    continue
                
                # 타임스탬프를 datetime으로 변환
                timestamp = int(tx.get('timeStamp', 0))
                tx_date = datetime.fromtimestamp(timestamp)
                
                # 정제된 트랜잭션 생성
                cleaned_tx = {
                    'hash': tx.get('hash', ''),
                    'from_address': utils.normalize_address(tx.get('from', '')),
                    'to_address': utils.normalize_address(tx.get('to', '')),
                    'value': int(tx.get('value', '0')),
                    'gas_used': int(tx.get('gasUsed', '0')),
                    'gas_price': int(tx.get('gasPrice', '0')),
                    'timestamp': timestamp,
                    'datetime': tx_date.isoformat(),
                    'block_number': int(tx.get('blockNumber', '0')),
                    'is_error': tx.get('isError', '0') == '1',
                    'tx_type': 'normal'
                }
                
                cleaned_txs.append(cleaned_tx)
                
            except Exception as e:
                logger.error(f"트랜잭션 정제 중 오류: {str(e)}")
                continue
        
        return cleaned_txs
    
    def clean_token_transfer_data(
        self,
        token_transfers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        토큰 전송 데이터 정제
        
        Args:
            token_transfers: 원시 토큰 전송 목록
            
        Returns:
            정제된 토큰 전송 목록
        """
        cleaned_transfers = []
        
        for transfer in token_transfers:
            try:
                # 필수 필드 확인
                if not all(k in transfer for k in ['hash', 'from', 'to', 'value', 'timeStamp', 'tokenSymbol']):
                    continue
                
                # 타임스탬프를 datetime으로 변환
                timestamp = int(transfer.get('timeStamp', 0))
                tx_date = datetime.fromtimestamp(timestamp)
                
                # 토큰 값 계산 (토큰 소수점 고려)
                token_decimals = int(transfer.get('tokenDecimal', '0'))
                raw_value = int(transfer.get('value', '0'))
                token_value = raw_value / (10 ** token_decimals) if token_decimals > 0 else raw_value
                
                # 정제된 전송 내역 생성
                cleaned_transfer = {
                    'hash': transfer.get('hash', ''),
                    'from_address': utils.normalize_address(transfer.get('from', '')),
                    'to_address': utils.normalize_address(transfer.get('to', '')),
                    'raw_value': raw_value,
                    'token_value': token_value,
                    'token_symbol': transfer.get('tokenSymbol', ''),
                    'token_name': transfer.get('tokenName', ''),
                    'token_address': utils.normalize_address(transfer.get('contractAddress', '')),
                    'token_decimals': token_decimals,
                    'timestamp': timestamp,
                    'datetime': tx_date.isoformat(),
                    'block_number': int(transfer.get('blockNumber', '0')),
                    'tx_type': 'token'
                }
                
                cleaned_transfers.append(cleaned_transfer)
                
            except Exception as e:
                logger.error(f"토큰 전송 정제 중 오류: {str(e)}")
                continue
        
        return cleaned_transfers
    
    def extract_wallet_features(
        self,
        wallet_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        지갑 데이터에서 특성 추출
        
        Args:
            wallet_data: 지갑 원시 데이터
            
        Returns:
            추출된 지갑 특성
        """
        features = {
            'address': wallet_data.get('address', ''),
            'extraction_time': utils.get_timestamp(),
            'basic_metrics': {},
            'temporal_metrics': {},
            'token_metrics': {},
            'network_metrics': {}
        }
        
        # 기본 지표 계산
        eth_balance = int(wallet_data.get('balance', '0'))
        features['basic_metrics']['eth_balance'] = eth_balance
        features['basic_metrics']['eth_balance_eth'] = eth_balance / 1e18 if eth_balance > 0 else 0
        
        # 트랜잭션 데이터 정제
        normal_txs = self.clean_transaction_data(wallet_data.get('transactions', []))
        token_txs = self.clean_token_transfer_data(wallet_data.get('token_transfers', []))
        internal_txs = self.clean_transaction_data(wallet_data.get('internal_transactions', []))
        nft_txs = self.clean_token_transfer_data(wallet_data.get('nft_transfers', []))
        
        # 모든 트랜잭션 합치기
        all_txs = normal_txs + internal_txs
        all_transfers = token_txs + nft_txs
        
        # 트랜잭션 수 계산
        features['basic_metrics']['normal_tx_count'] = len(normal_txs)
        features['basic_metrics']['internal_tx_count'] = len(internal_txs)
        features['basic_metrics']['token_tx_count'] = len(token_txs)
        features['basic_metrics']['nft_tx_count'] = len(nft_txs)
        features['basic_metrics']['total_tx_count'] = len(all_txs) + len(all_transfers)
        
        # 트랜잭션이 없는 경우 기본 특성만 반환
        if features['basic_metrics']['total_tx_count'] == 0:
            return features
        
        # 시간 관련 특성 추출
        self._extract_temporal_features(features, all_txs, all_transfers)
        
        # 토큰 관련 특성 추출
        self._extract_token_features(features, token_txs, nft_txs)
        
        # 네트워크 관련 특성 추출
        self._extract_network_features(features, all_txs, all_transfers, wallet_data.get('address', ''))
        
        return features
    
    def _extract_temporal_features(
        self,
        features: Dict[str, Any],
        transactions: List[Dict[str, Any]],
        token_transfers: List[Dict[str, Any]]
    ) -> None:
        """
        시간 관련 특성 추출
        
        Args:
            features: 특성 딕셔너리
            transactions: 정제된 트랜잭션 목록
            token_transfers: 정제된 토큰 전송 목록
        """
        # 모든 활동 병합 및 정렬
        all_activities = transactions + token_transfers
        if not all_activities:
            return
        
        # 타임스탬프 기준 정렬
        all_activities.sort(key=lambda x: x.get('timestamp', 0))
        
        # 첫 활동 및 마지막 활동 시간
        first_tx_time = datetime.fromtimestamp(all_activities[0].get('timestamp', 0))
        last_tx_time = datetime.fromtimestamp(all_activities[-1].get('timestamp', 0))
        
        # 지갑 연령 (일)
        wallet_age_days = (last_tx_time - first_tx_time).days
        features['temporal_metrics']['first_activity'] = first_tx_time.isoformat()
        features['temporal_metrics']['last_activity'] = last_tx_time.isoformat()
        features['temporal_metrics']['wallet_age_days'] = max(wallet_age_days, 1)  # 최소 1일
        
        # 활동 기간 구분 (최근 1일, 7일, 30일, 90일, 180일)
        now = datetime.now()
        activity_periods = {
            'last_day': 0,
            'last_week': 0,
            'last_month': 0,
            'last_quarter': 0,
            'last_half_year': 0
        }
        
        # 각 기간별 활동 수 계산
        for activity in all_activities:
            tx_time = datetime.fromtimestamp(activity.get('timestamp', 0))
            time_diff = now - tx_time
            
            if time_diff <= timedelta(days=1):
                activity_periods['last_day'] += 1
            if time_diff <= timedelta(days=7):
                activity_periods['last_week'] += 1
            if time_diff <= timedelta(days=30):
                activity_periods['last_month'] += 1
            if time_diff <= timedelta(days=90):
                activity_periods['last_quarter'] += 1
            if time_diff <= timedelta(days=180):
                activity_periods['last_half_year'] += 1
        
        features['temporal_metrics']['activity_periods'] = activity_periods
        
        # 활동 빈도 계산
        total_activities = len(all_activities)
        days_with_activity = len(set(datetime.fromtimestamp(tx.get('timestamp', 0)).date() for tx in all_activities))
        
        features['temporal_metrics']['total_activities'] = total_activities
        features['temporal_metrics']['days_with_activity'] = days_with_activity
        features['temporal_metrics']['activity_frequency'] = total_activities / max(wallet_age_days, 1)
        features['temporal_metrics']['active_days_ratio'] = days_with_activity / max(wallet_age_days, 1)
    
    def _extract_token_features(
        self,
        features: Dict[str, Any],
        token_transfers: List[Dict[str, Any]],
        nft_transfers: List[Dict[str, Any]]
    ) -> None:
        """
        토큰 관련 특성 추출
        
        Args:
            features: 특성 딕셔너리
            token_transfers: 정제된 토큰 전송 목록
            nft_transfers: 정제된 NFT 전송 목록
        """
        wallet_address = features.get('address', '').lower()
        
        # 고유 토큰 정보 수집
        unique_tokens = {}
        token_balances = defaultdict(float)
        
        # ERC-20 토큰 처리
        for transfer in token_transfers:
            token_address = transfer.get('token_address', '').lower()
            token_symbol = transfer.get('token_symbol', '')
            from_address = transfer.get('from_address', '').lower()
            to_address = transfer.get('to_address', '').lower()
            token_value = transfer.get('token_value', 0)
            
            # 고유 토큰 정보 저장
            if token_address and token_address not in unique_tokens:
                unique_tokens[token_address] = {
                    'symbol': token_symbol,
                    'name': transfer.get('token_name', ''),
                    'decimals': transfer.get('token_decimals', 0),
                    'transaction_count': 0
                }
            
            # 토큰 거래 카운트 증가
            if token_address in unique_tokens:
                unique_tokens[token_address]['transaction_count'] += 1
            
            # 토큰 잔액 계산
            if from_address == wallet_address:
                token_balances[token_address] -= token_value
            if to_address == wallet_address:
                token_balances[token_address] += token_value
        
        # NFT 처리
        unique_nfts = {}
        for transfer in nft_transfers:
            nft_address = transfer.get('token_address', '').lower()
            nft_symbol = transfer.get('token_symbol', '')
            
            # 고유 NFT 정보 저장
            if nft_address and nft_address not in unique_nfts:
                unique_nfts[nft_address] = {
                    'symbol': nft_symbol,
                    'name': transfer.get('token_name', ''),
                    'transaction_count': 0
                }
            
            # NFT 거래 카운트 증가
            if nft_address in unique_nfts:
                unique_nfts[nft_address]['transaction_count'] += 1
        
        # 토큰 지표 계산
        features['token_metrics']['unique_tokens_count'] = len(unique_tokens)
        features['token_metrics']['unique_nfts_count'] = len(unique_nfts)
        features['token_metrics']['positive_token_balances'] = sum(1 for balance in token_balances.values() if balance > 0)
        features['token_metrics']['token_list'] = list(unique_tokens.keys())
        features['token_metrics']['nft_list'] = list(unique_nfts.keys())
        
        # 상위 토큰 정보
        if unique_tokens:
            top_tokens = sorted(
                [(addr, data) for addr, data in unique_tokens.items()],
                key=lambda x: x[1]['transaction_count'],
                reverse=True
            )[:5]  # 상위 5개
            
            features['token_metrics']['top_tokens'] = [
                {
                    'address': addr,
                    'symbol': data['symbol'],
                    'name': data['name'],
                    'tx_count': data['transaction_count'],
                    'balance': token_balances.get(addr, 0)
                } for addr, data in top_tokens
            ]
    
    def _extract_network_features(
        self,
        features: Dict[str, Any],
        transactions: List[Dict[str, Any]],
        token_transfers: List[Dict[str, Any]],
        wallet_address: str
    ) -> None:
        """
        네트워크 관련 특성 추출
        
        Args:
            features: 특성 딕셔너리
            transactions: 정제된 트랜잭션 목록
            token_transfers: 정제된 토큰 전송 목록
            wallet_address: 지갑 주소
        """
        wallet_address = wallet_address.lower()
        
        # 상호작용한 고유 주소 집합
        interacted_addresses = set()
        
        # 트랜잭션 타입별 발신/수신 횟수
        sent_tx_count = 0
        received_tx_count = 0
        sent_token_count = 0
        received_token_count = 0
        
        # ETH 전송 금액
        total_eth_sent = 0
        total_eth_received = 0
        
        # 일반 트랜잭션 분석
        for tx in transactions:
            from_addr = tx.get('from_address', '').lower()
            to_addr = tx.get('to_address', '').lower()
            value = tx.get('value', 0) / 1e18  # Wei to ETH
            
            # 상호작용 주소 추가
            if from_addr and from_addr != wallet_address:
                interacted_addresses.add(from_addr)
            if to_addr and to_addr != wallet_address:
                interacted_addresses.add(to_addr)
            
            # 발신/수신 트랜잭션 집계
            if from_addr == wallet_address:
                sent_tx_count += 1
                total_eth_sent += value
            if to_addr == wallet_address:
                received_tx_count += 1
                total_eth_received += value
        
        # 토큰 거래 분석
        for transfer in token_transfers:
            from_addr = transfer.get('from_address', '').lower()
            to_addr = transfer.get('to_address', '').lower()
            
            # 상호작용 주소 추가
            if from_addr and from_addr != wallet_address:
                interacted_addresses.add(from_addr)
            if to_addr and to_addr != wallet_address:
                interacted_addresses.add(to_addr)
            
            # 발신/수신 토큰 거래 집계
            if from_addr == wallet_address:
                sent_token_count += 1
            if to_addr == wallet_address:
                received_token_count += 1
        
        # 네트워크 지표 저장
        features['network_metrics']['unique_interacted_addresses'] = len(interacted_addresses)
        features['network_metrics']['sent_tx_count'] = sent_tx_count
        features['network_metrics']['received_tx_count'] = received_tx_count
        features['network_metrics']['sent_token_count'] = sent_token_count
        features['network_metrics']['received_token_count'] = received_token_count
        features['network_metrics']['total_eth_sent'] = total_eth_sent
        features['network_metrics']['total_eth_received'] = total_eth_received
        
        # 송신/수신 비율 계산
        total_sent = sent_tx_count + sent_token_count
        total_received = received_tx_count + received_token_count
        total_txs = total_sent + total_received
        
        if total_txs > 0:
            features['network_metrics']['sent_ratio'] = total_sent / total_txs
            features['network_metrics']['received_ratio'] = total_received / total_txs
        else:
            features['network_metrics']['sent_ratio'] = 0
            features['network_metrics']['received_ratio'] = 0
    
    def process_wallet_data(
        self,
        address_or_data: Union[str, Dict[str, Any]],
        save_processed: bool = True
    ) -> Dict[str, Any]:
        """
        지갑 데이터 처리 및 특성 추출
        
        Args:
            address_or_data: 이더리움 주소 또는 지갑 데이터
            save_processed: 처리된 데이터 저장 여부
            
        Returns:
            처리된 지갑 특성 데이터
        """
        # 원시 데이터 로드
        wallet_data = {}
        if isinstance(address_or_data, str):
            wallet_data = self.load_raw_data(address_or_data)
            if not wallet_data:
                logger.error(f"지갑 데이터 로드 실패: {address_or_data}")
                return {}
        else:
            wallet_data = address_or_data
        
        # 필수 데이터 확인
        if 'address' not in wallet_data:
            logger.error("유효하지 않은 지갑 데이터: 주소 누락")
            return {}
        
        logger.info(f"지갑 데이터 처리 시작: {wallet_data.get('address')}")
        
        # 특성 추출
        processed_data = self.extract_wallet_features(wallet_data)
        
        # 처리된 데이터 저장
        if save_processed and self.processed_dir and processed_data:
            address = wallet_data.get('address')
            address_hash = utils.generate_hash(address)
            
            # 파일 경로 생성
            filepath = utils.get_data_filepath(
                self.processed_dir,
                address_hash,
                "processed"
            )
            
            # 데이터 저장
            if utils.save_json(processed_data, filepath):
                logger.info(f"처리된 데이터 저장 완료: {filepath}")
            else:
                logger.error(f"처리된 데이터 저장 실패: {filepath}")
        
        logger.info(f"지갑 데이터 처리 완료: {wallet_data.get('address')}")
        return processed_data
    
    def process_multiple_wallets(
        self,
        addresses_or_data: List[Union[str, Dict[str, Any]]],
        save_processed: bool = True
    ) -> Dict[str, Any]:
        """
        여러 지갑 데이터 처리
        
        Args:
            addresses_or_data: 이더리움 주소 목록 또는 지갑 데이터 목록
            save_processed: 처리된 데이터 저장 여부
            
        Returns:
            처리 결과 요약
        """
        logger.info(f"다중 지갑 데이터 처리 시작: {len(addresses_or_data)}개")
        
        results = {
            "success": 0,
            "failed": 0,
            "details": {}
        }
        
        # 각 지갑 데이터 처리
        for item in addresses_or_data:
            try:
                # 주소 또는 데이터에서 식별자 추출
                identifier = ""
                if isinstance(item, str):
                    identifier = item
                else:
                    identifier = item.get('address', '')
                
                # 데이터 처리
                processed_data = self.process_wallet_data(item, save_processed)
                
                # 결과 기록
                if processed_data:
                    results["success"] += 1
                    results["details"][identifier] = "성공"
                else:
                    results["failed"] += 1
                    results["details"][identifier] = "처리 실패"
            
            except Exception as e:
                logger.error(f"지갑 데이터 처리 중 오류 발생: {str(e)}")
                results["failed"] += 1
                identifier = item if isinstance(item, str) else str(item)[:50]
                results["details"][identifier] = str(e)
        
        logger.info(f"다중 지갑 데이터 처리 완료: 성공 {results['success']}개, 실패 {results['failed']}개")
        return results
    
    def create_feature_dataframe(
        self,
        processed_wallets: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        처리된 지갑 데이터에서 특성 데이터프레임 생성
        
        Args:
            processed_wallets: 처리된 지갑 데이터 목록
            
        Returns:
            특성 데이터프레임
        """
        if not processed_wallets:
            logger.error("처리된 지갑 데이터가 없습니다.")
            return pd.DataFrame()
        
        # 특성 추출
        features_list = []
        
        for wallet in processed_wallets:
            if not wallet:
                continue
                
            try:
                address = wallet.get('address', '')
                
                # 기본 지표
                basic_metrics = wallet.get('basic_metrics', {})
                
                # 시간 지표
                temporal_metrics = wallet.get('temporal_metrics', {})
                activity_periods = temporal_metrics.get('activity_periods', {})
                
                # 토큰 지표
                token_metrics = wallet.get('token_metrics', {})
                
                # 네트워크 지표
                network_metrics = wallet.get('network_metrics', {})
                
                # 특성 딕셔너리 생성
                features = {
                    'address': address,
                    
                    # 기본 지표
                    'eth_balance': basic_metrics.get('eth_balance_eth', 0),
                    'total_tx_count': basic_metrics.get('total_tx_count', 0),
                    'normal_tx_count': basic_metrics.get('normal_tx_count', 0),
                    'token_tx_count': basic_metrics.get('token_tx_count', 0),
                    'nft_tx_count': basic_metrics.get('nft_tx_count', 0),
                    
                    # 시간 지표
                    'wallet_age_days': temporal_metrics.get('wallet_age_days', 0),
                    'activity_frequency': temporal_metrics.get('activity_frequency', 0),
                    'active_days_ratio': temporal_metrics.get('active_days_ratio', 0),
                    'activity_last_day': activity_periods.get('last_day', 0),
                    'activity_last_week': activity_periods.get('last_week', 0),
                    'activity_last_month': activity_periods.get('last_month', 0),
                    'activity_last_quarter': activity_periods.get('last_quarter', 0),
                    
                    # 토큰 지표
                    'unique_tokens_count': token_metrics.get('unique_tokens_count', 0),
                    'unique_nfts_count': token_metrics.get('unique_nfts_count', 0),
                    'positive_token_balances': token_metrics.get('positive_token_balances', 0),
                    
                    # 네트워크 지표
                    'unique_interacted_addresses': network_metrics.get('unique_interacted_addresses', 0),
                    'sent_tx_count': network_metrics.get('sent_tx_count', 0),
                    'received_tx_count': network_metrics.get('received_tx_count', 0),
                    'sent_token_count': network_metrics.get('sent_token_count', 0),
                    'received_token_count': network_metrics.get('received_token_count', 0),
                    'total_eth_sent': network_metrics.get('total_eth_sent', 0),
                    'total_eth_received': network_metrics.get('total_eth_received', 0),
                    'sent_ratio': network_metrics.get('sent_ratio', 0),
                    'received_ratio': network_metrics.get('received_ratio', 0)
                }
                
                features_list.append(features)
                
            except Exception as e:
                logger.error(f"특성 추출 중 오류: {str(e)}")
                continue
        
        # 데이터프레임 생성
        df = pd.DataFrame(features_list)
        
        logger.info(f"특성 데이터프레임 생성 완료: {len(df)} 행 x {len(df.columns)} 열")
        return df
    
    def save_feature_dataframe(
        self,
        df: pd.DataFrame,
        filename: str = "wallet_features.csv"
    ) -> str:
        """
        특성 데이터프레임 저장
        
        Args:
            df: 특성 데이터프레임
            filename: 저장할 파일명
            
        Returns:
            저장된 파일 경로
        """
        if df.empty:
            logger.error("저장할 데이터프레임이 비어 있습니다.")
            return ""
        
        try:
            # 파일 경로 생성
            filepath = os.path.join(self.processed_dir, filename)
            
            # CSV 파일로 저장
            df.to_csv(filepath, index=False)
            logger.info(f"특성 데이터프레임 저장 완료: {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"데이터프레임 저장 중 오류: {str(e)}")
            return "" 
"""
이더리움 지갑 주소 데이터 수집, 처리 및 저장 모듈의 명령행 인터페이스
"""

import os
import argparse
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from .utils import setup_logger, is_valid_eth_address
from .blockchain_data import BlockchainDataCollector
from .data_processor import DataProcessor
from .data_storage import DataStorage

logger = setup_logger('main')

def collect_address_data(address: str, api_key: str = None, include_internal: bool = True,
                        include_tokens: bool = True, data_dir: str = "data") -> Dict[str, Any]:
    """
    이더리움 주소 데이터 수집
    
    Args:
        address: 이더리움 주소
        api_key: Etherscan API 키
        include_internal: 내부 트랜잭션 포함 여부
        include_tokens: 토큰 전송 내역 포함 여부
        data_dir: 데이터 저장 디렉토리
        
    Returns:
        수집 결과
    """
    logger.info(f"주소 데이터 수집 시작: {address}")
    
    # 주소 유효성 검사
    if not is_valid_eth_address(address):
        return {'error': f"유효하지 않은 이더리움 주소: {address}"}
    
    # 데이터 수집기 초기화
    collector = BlockchainDataCollector(api_key, data_dir=data_dir)
    
    # 데이터 수집 및 저장
    data, file_path = collector.collect_and_save(
        address, include_internal, include_tokens)
    
    if 'error' in data:
        logger.error(f"데이터 수집 실패: {data['error']}")
        return data
    
    logger.info(f"주소 데이터 수집 완료: {address}")
    
    result = {
        'address': address,
        'collection_time': datetime.now().isoformat(),
        'data_file': file_path,
        'stats': data.get('stats', {}),
        'success': True
    }
    
    return result

def process_data(address: str = None, raw_data_path: str = None,
               data_dir: str = "data", save_interim: bool = True) -> Dict[str, Any]:
    """
    이더리움 주소 데이터 처리
    
    Args:
        address: 이더리움 주소
        raw_data_path: 원시 데이터 파일 경로 (지정하지 않으면 주소를 기반으로 최신 파일 사용)
        data_dir: 데이터 디렉토리
        save_interim: 중간 처리 결과 저장 여부
        
    Returns:
        처리 결과
    """
    logger.info(f"데이터 처리 시작: {address or raw_data_path}")
    
    # 데이터 처리기 및 저장소 초기화
    processor = DataProcessor(
        raw_data_dir=os.path.join(data_dir, "raw"),
        processed_data_dir=os.path.join(data_dir, "processed")
    )
    
    storage = DataStorage(data_dir=data_dir)
    
    # 원시 데이터 파일 경로가 지정되지 않았다면 주소에서 최신 데이터 파일 조회
    if not raw_data_path and address:
        raw_data_path = storage.get_latest_data_path(address, "raw")
        
        if not raw_data_path:
            return {'error': f"주소의 원시 데이터를 찾을 수 없음: {address}"}
    
    # 데이터 처리 및 저장
    processed_data, processed_file = processor.process_and_save(raw_data_path)
    
    if 'error' in processed_data:
        logger.error(f"데이터 처리 실패: {processed_data['error']}")
        return processed_data
    
    # 처리된 데이터에서 주소 추출
    address = processed_data.get('address', '')
    
    # 벡터 저장
    if 'feature_vector' in processed_data:
        vector_data = {
            'address': address,
            'created_at': datetime.now().isoformat(),
            'feature_vector': processed_data['feature_vector']
        }
        success, vector_file = storage.store_feature_vector(address, vector_data)
        
        if success:
            logger.info(f"특성 벡터 저장 완료: {vector_file}")
    
    logger.info(f"데이터 처리 완료: {address}")
    
    result = {
        'address': address,
        'processing_time': datetime.now().isoformat(),
        'processed_file': processed_file,
        'feature_count': len(processed_data.get('feature_vector', {})),
        'success': True
    }
    
    return result

def analyze_address(address: str, api_key: str = None, data_dir: str = "data",
                  force_collect: bool = False) -> Dict[str, Any]:
    """
    이더리움 주소 분석 (수집 및 처리 통합)
    
    Args:
        address: 이더리움 주소
        api_key: Etherscan API 키
        data_dir: 데이터 디렉토리
        force_collect: 기존 데이터가 있어도 강제로 새로 수집
        
    Returns:
        분석 결과
    """
    logger.info(f"주소 분석 시작: {address}")
    
    # 주소 유효성 검사
    if not is_valid_eth_address(address):
        return {'error': f"유효하지 않은 이더리움 주소: {address}"}
    
    # 데이터 저장소 초기화
    storage = DataStorage(data_dir=data_dir)
    
    # 기존 데이터 확인
    raw_data_path = storage.get_latest_data_path(address, "raw")
    
    # 기존 데이터가 없거나 강제 수집인 경우
    if not raw_data_path or force_collect:
        # 데이터 수집
        collect_result = collect_address_data(address, api_key, data_dir=data_dir)
        
        if 'error' in collect_result:
            return collect_result
        
        raw_data_path = collect_result.get('data_file', '')
    
    # 데이터 처리
    process_result = process_data(address, raw_data_path, data_dir)
    
    if 'error' in process_result:
        return process_result
    
    # 처리된 데이터 로드
    processed_data = storage.load_data(address, "processed")
    feature_vector = storage.load_data(address, "vector")
    
    # 결과 통합
    result = {
        'address': address,
        'analysis_time': datetime.now().isoformat(),
        'features': processed_data.get('features', {}),
        'feature_vector': feature_vector.get('feature_vector', {}),
        'files': {
            'raw': raw_data_path,
            'processed': process_result.get('processed_file', '')
        },
        'success': True
    }
    
    logger.info(f"주소 분석 완료: {address}")
    return result

def export_data(output_format: str = "json", addresses: int = 100,
               data_dir: str = "data", output_file: str = None) -> Dict[str, Any]:
    """
    처리된 데이터 내보내기
    
    Args:
        output_format: 출력 형식 (json, csv)
        addresses: 내보낼 주소 수 (최대)
        data_dir: 데이터 디렉토리
        output_file: 출력 파일 경로
        
    Returns:
        내보내기 결과
    """
    logger.info(f"데이터 내보내기 시작 (형식: {output_format})")
    
    # 데이터 저장소 초기화
    storage = DataStorage(data_dir=data_dir)
    
    # 주소 목록 조회
    address_list = storage.list_addresses(limit=addresses)
    
    if not address_list:
        return {'error': "내보낼 주소 데이터가 없습니다."}
    
    # 출력 파일 경로 설정
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"exported_data_{timestamp}.{output_format}"
    
    # 데이터프레임 생성
    df = storage.export_to_dataframe([info['address'] for info in address_list])
    
    if df.empty:
        return {'error': "내보낼 특성 벡터 데이터가 없습니다."}
    
    # 데이터 내보내기
    try:
        if output_format.lower() == 'csv':
            df.to_csv(output_file, index=False)
        else:  # json
            result_json = df.to_json(orient='records', indent=2)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result_json)
        
        logger.info(f"데이터 내보내기 완료: {output_file}")
        
        return {
            'output_file': output_file,
            'address_count': len(df),
            'feature_count': len(df.columns) - 1,  # 주소 열 제외
            'success': True
        }
        
    except Exception as e:
        logger.error(f"데이터 내보내기 오류: {str(e)}")
        return {'error': f"데이터 내보내기 오류: {str(e)}"}

def main():
    """메인 함수"""
    # 인자 파서 설정
    parser = argparse.ArgumentParser(
        description="이더리움 지갑 주소 데이터 수집, 처리 및 저장 도구"
    )
    
    # 하위 명령어 설정
    subparsers = parser.add_subparsers(dest='command', help='실행할 명령')
    
    # collect 명령어
    collect_parser = subparsers.add_parser('collect', help='주소 데이터 수집')
    collect_parser.add_argument('address', help='이더리움 주소')
    collect_parser.add_argument('--api-key', help='Etherscan API 키')
    collect_parser.add_argument('--no-internal', action='store_true', help='내부 트랜잭션 제외')
    collect_parser.add_argument('--no-tokens', action='store_true', help='토큰 전송 내역 제외')
    collect_parser.add_argument('--data-dir', default='data', help='데이터 디렉토리')
    
    # process 명령어
    process_parser = subparsers.add_parser('process', help='수집한 데이터 처리')
    process_parser.add_argument('--address', help='이더리움 주소')
    process_parser.add_argument('--raw-file', help='원시 데이터 파일 경로')
    process_parser.add_argument('--data-dir', default='data', help='데이터 디렉토리')
    process_parser.add_argument('--no-interim', action='store_true', help='중간 처리 결과 저장 안 함')
    
    # analyze 명령어
    analyze_parser = subparsers.add_parser('analyze', help='주소 분석 (수집 및 처리 통합)')
    analyze_parser.add_argument('address', help='이더리움 주소')
    analyze_parser.add_argument('--api-key', help='Etherscan API 키')
    analyze_parser.add_argument('--data-dir', default='data', help='데이터 디렉토리')
    analyze_parser.add_argument('--force-collect', action='store_true', help='기존 데이터가 있어도 새로 수집')
    
    # export 명령어
    export_parser = subparsers.add_parser('export', help='처리된 데이터 내보내기')
    export_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='출력 형식')
    export_parser.add_argument('--addresses', type=int, default=100, help='내보낼 최대 주소 수')
    export_parser.add_argument('--data-dir', default='data', help='데이터 디렉토리')
    export_parser.add_argument('--output', help='출력 파일 경로')
    
    # cleanup 명령어
    cleanup_parser = subparsers.add_parser('cleanup', help='오래된 데이터 정리')
    cleanup_parser.add_argument('--address', help='특정 주소만 정리 (선택사항)')
    cleanup_parser.add_argument('--type', choices=['raw', 'processed', 'vector'], help='특정 유형만 정리 (선택사항)')
    cleanup_parser.add_argument('--keep', type=int, default=2, help='보존할 최신 파일 수')
    cleanup_parser.add_argument('--data-dir', default='data', help='데이터 디렉토리')
    
    # 인자 파싱
    args = parser.parse_args()
    
    # 명령 실행
    if args.command == 'collect':
        result = collect_address_data(
            args.address,
            args.api_key,
            not args.no_internal,
            not args.no_tokens,
            args.data_dir
        )
    elif args.command == 'process':
        if not args.address and not args.raw_file:
            print("오류: 주소 또는 원시 데이터 파일 경로를 지정해야 합니다.")
            return 1
        
        result = process_data(
            args.address,
            args.raw_file,
            args.data_dir,
            not args.no_interim
        )
    elif args.command == 'analyze':
        result = analyze_address(
            args.address,
            args.api_key,
            args.data_dir,
            args.force_collect
        )
    elif args.command == 'export':
        result = export_data(
            args.format,
            args.addresses,
            args.data_dir,
            args.output
        )
    elif args.command == 'cleanup':
        storage = DataStorage(data_dir=args.data_dir)
        success = storage.clear_old_data(args.address, args.type, args.keep)
        result = {
            'success': success,
            'message': "데이터 정리 완료" if success else "데이터 정리 실패"
        }
    else:
        parser.print_help()
        return 0
    
    # 결과 출력
    if 'error' in result:
        print(f"오류: {result['error']}")
        return 1
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    sys.exit(main()) 
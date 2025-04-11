# 블록체인 데이터 처리 모듈 (data-process)

## 개요
이 모듈은 이더리움 블록체인 데이터를 수집, 처리 및 저장하기 위한 파이프라인을 제공합니다. 지갑 주소의 온체인 활동 데이터를 Etherscan API를 통해 수집하고, 이를 머신러닝 및 분석에 적합한 형태로 가공하여 저장합니다.

## 파일 구조
- `__init__.py`: 모듈 초기화 파일
- `utils.py`: 로깅, 데이터 검증, 파일 관리 등 유틸리티 함수
- `blockchain_data.py`: 블록체인 데이터 수집 기능
- `data_processor.py`: 수집된 데이터 처리 및 특성 추출
- `data_storage.py`: 데이터 저장, 관리 및 검색
- `main.py`: 명령행 인터페이스 및 실행 스크립트

## 주요 기능

### 1. 블록체인 데이터 수집
- Etherscan API를 통한 트랜잭션 내역 수집
- 일반 트랜잭션, 내부 트랜잭션, 토큰 전송 내역 통합 수집
- 수집된 원시 데이터 저장 및 관리

### 2. 데이터 처리 및 특성 추출
- 트랜잭션 데이터에서 다양한 특성 추출
  - 트랜잭션 활동 특성 (전송/수신 비율, 컨트랙트 호출 등)
  - 시간 패턴 특성 (활동 시간대, 활동 기간 등)
  - 가치 전송 특성 (평균 거래액, 최대 거래액 등)
  - 토큰 관련 특성 (고유 토큰 수, 토큰 전송 패턴 등)
- 파생 특성 계산 및 정규화
- 머신러닝 분석을 위한 특성 벡터 생성

### 3. 데이터 저장 및 관리
- SQLite 데이터베이스를 활용한 메타데이터 관리
- 주소별 데이터 파일 관리 및 인덱싱
- 특성 벡터 저장 및 검색
- 데이터 내보내기 (JSON, CSV 형식)

### 4. 데이터 파이프라인 통합
- 수집-처리-저장 파이프라인 자동화
- 기존 데이터 재활용 및 증분 업데이트
- 명령행 인터페이스를 통한 사용자 친화적 실행

## 데이터 흐름
1. 이더리움 주소를 입력하여 Etherscan API를 통해 데이터 수집
2. 수집된 원시 데이터를 JSON 형식으로 저장
3. 원시 데이터를 처리하여 특성 추출 및 가공
4. 처리된 데이터와 특성 벡터를 저장
5. 데이터베이스에 메타데이터 등록 및 관리
6. 분석 및 모델링을 위한 데이터 내보내기

## AI 파이프라인 단계별 구현

| 단계 | 구현 파일 | 주요 함수/클래스 |
|-----|----------|-----------------|
| 데이터 수집 | blockchain_data.py | BlockchainDataCollector.collect_address_data() |
| 데이터 전처리 | data_processor.py | DataProcessor.extract_transaction_features() |
| 특성 추출 | data_processor.py | DataProcessor.compute_derived_features() |
| 특성 정규화 | data_processor.py | DataProcessor.normalize_features() |
| 특성 벡터 생성 | data_processor.py | DataProcessor.create_feature_vector() |
| 데이터 저장 | data_storage.py | DataStorage.store_processed_data() |
| 데이터 검색 | data_storage.py | DataStorage.load_data() |
| 데이터 내보내기 | data_storage.py | DataStorage.export_to_dataframe() |

## 사용 방법

### 데이터 수집
```bash
python -m data-process.main collect 0x123...abc --api-key YOUR_API_KEY
```

### 데이터 처리
```bash
python -m data-process.main process --address 0x123...abc
```

### 통합 분석 (수집 및 처리)
```bash
python -m data-process.main analyze 0x123...abc --force-collect
```

### 데이터 내보내기
```bash
python -m data-process.main export --format csv --output exported_data.csv
```

### 오래된 데이터 정리
```bash
python -m data-process.main cleanup --keep 2
```

## 요구사항
- Python 3.7 이상
- pandas, numpy, web3, requests 라이브러리
- SQLite 데이터베이스 
- Etherscan API 키 (선택사항, 사용 제한이 있을 수 있음)

## 확장 가능성
- 다중 블록체인 데이터 소스 지원 (BSC, Polygon 등)
- 고급 특성 추출 알고리즘 적용
- 분산 저장소 지원 (대규모 데이터셋용)
- 실시간 데이터 수집 및 처리 파이프라인 
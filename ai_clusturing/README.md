# AI 클러스터링 모듈 (ai_clusturing)

## 개요
이 모듈은 이더리움 지갑 주소의 온체인 활동 데이터를 비지도 학습 기법으로 분석하여 유사한 행동 패턴을 가진 사용자들을 클러스터링합니다. 지갑 주소의 특성을 추출하고, 다양한 클러스터링 알고리즘을 적용하여 사용자 프로필을 생성합니다.

## 파일 구조
- `__init__.py`: 모듈 초기화 파일
- `utils.py`: 로깅, 데이터 로드, 결과 저장 등 유틸리티 함수
- `feature_extraction.py`: 온체인 데이터에서 특성 추출 기능
- `clustering.py`: 클러스터링 알고리즘 구현 (K-means, DBSCAN 등)
- `cluster_analyzer.py`: 클러스터 분석 및 해석 기능
- `main.py`: 메인 실행 스크립트 및 명령행 인터페이스

## 주요 기능

### 1. 데이터 수집 및 전처리
- 이더리움 지갑 주소의 트랜잭션 데이터 로드
- 데이터 정규화 및 결측치 처리
- 특성 추출 및 차원 축소

### 2. 특성 추출 (Feature Engineering)
- 트랜잭션 기반 특성: 트랜잭션 빈도, 금액, 가스비 등 추출
- 시간 기반 특성: 활동 시간대, 패턴, 간격 분석
- 컨트랙트 상호작용 특성: 스마트 컨트랙트 호출 패턴 분석
- 토큰 관련 특성: 토큰 보유량, 거래 패턴 분석

### 3. 클러스터링 모델
- K-means 클러스터링: 유사한 사용자 그룹 식별
- DBSCAN: 밀도 기반 클러스터링으로 이상치 탐지
- 계층적 클러스터링: 사용자 그룹 간 계층 구조 분석
- 최적 클러스터 수 자동 결정 (실루엣 점수, 엘보우 방법)

### 4. 클러스터 분석
- 클러스터 특성 프로필링: 각 클러스터의 주요 특성 식별
- 클러스터 시각화: t-SNE, PCA 등을 통한 차원 축소 및 시각화
- 특성 중요도 분석: 클러스터 형성에 주요한 특성 식별

### 5. 결과 저장 및 활용
- 클러스터 모델 저장 및 로드
- 새로운 지갑 주소의 클러스터 예측
- 클러스터 프로필 저장 및 업데이트

## 사용 방법

### 클러스터링 실행하기
```bash
python -m ai_clusturing.main cluster --data_path=/path/to/data.json --n_clusters=5
```

### 특정 주소 분석하기
```bash
python -m ai_clusturing.main analyze --address=0x123...abc
```

### 모델 평가하기
```bash
python -m ai_clusturing.main evaluate --model_path=/path/to/model.pkl
```

## AI 파이프라인 단계별 구현

| AI 단계 | 구현 파일 | 주요 함수/클래스 |
|---------|----------|-----------------|
| 데이터 수집 | utils.py | load_data(), load_address_data() |
| 데이터 전처리 | feature_extraction.py | FeatureExtractor.preprocess() |
| 특성 추출 | feature_extraction.py | FeatureExtractor.extract_features() |
| 모델 학습 | clustering.py | ClusteringModel.fit() |
| 모델 평가 | clustering.py | evaluate_clustering() |
| 모델 저장 | utils.py | save_model(), load_model() |
| 추론(인퍼런스) | cluster_analyzer.py | ClusterAnalyzer.predict_cluster() |
| 결과 해석 | cluster_analyzer.py | ClusterAnalyzer.analyze_cluster() | 
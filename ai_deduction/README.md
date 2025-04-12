# AI 추론 모듈 (ai_deduction)

## 개요
이 모듈은 이더리움 지갑 주소의 온체인 활동 데이터를 지도 학습 모델을 통해 분석하여 사용자 유형을 분류합니다. 사전에 라벨링된 데이터로 학습된 모델을 활용하여 새로운 지갑 주소가 어떤 유형의 사용자인지 예측합니다.

## 파일 구조
- `__init__.py`: 모듈 초기화 파일
- `utils.py`: 로깅, 데이터 로드, 모델 관리 등 유틸리티 함수
- `feature_engineering.py`: 분류 모델을 위한 특성 추출 및 가공
- `model.py`: 지도 학습 모델 정의 및 학습/평가 기능
- `inference_service.py`: 모델 로딩 및 실시간 추론 서비스 구현
- `main.py`: 메인 실행 스크립트 및 명령행 인터페이스

## 주요 기능

### 1. 데이터 수집 및 전처리
- 라벨링된 이더리움 지갑 주소 데이터 로드
- 데이터 정제 및 표준화
- 학습/검증/테스트 데이터 분할

### 2. 특성 공학 (Feature Engineering)
- 온체인 트랜잭션 패턴 분석
- 시계열 특성 추출 (활동 시간, 빈도 등)
- 토큰 보유 및 거래 특성 생성
- 컨트랙트 호출 패턴 분석
- 특성 선택 및 차원 축소

### 3. 모델 학습
- 분류 모델 구현 (랜덤 포레스트, XGBoost, 신경망 등)
- 하이퍼파라미터 최적화
- 교차 검증 및 모델 평가
- 앙상블 기법 적용

### 4. 추론 서비스
- 학습된 모델 로드 및 추론 환경 설정
- 실시간 추론 처리
- 추론 결과 캐싱 및 관리
- 예측 신뢰도 및 설명 가능성 제공

### 5. 결과 해석 및 활용
- 예측 결과 해석 및 시각화
- 중요 특성 분석
- 모델 업데이트 메커니즘

## 사용 방법

### 모델 훈련하기
```bash
python -m ai_deduction.main train --data_path=/path/to/labeled_data.json --model_type=xgboost
```

### 주소 분류하기
```bash
python -m ai_deduction.main classify --address=0x123...abc
```

### 모델 평가하기
```bash
python -m ai_deduction.main evaluate --test_data=/path/to/test_data.json
```

## AI 파이프라인 단계별 구현

| AI 단계 | 구현 파일 | 주요 함수/클래스 |
|---------|----------|-----------------|
| 데이터 수집 | utils.py | load_data(), load_labeled_data() |
| 데이터 전처리 | feature_engineering.py | preprocess_data(), normalize_features() |
| 특성 추출 | feature_engineering.py | FeatureEngineer.extract_features() |
| 모델 학습 | model.py | ClassificationModel.train() |
| 모델 평가 | model.py | evaluate_model(), cross_validate() |
| 모델 저장 | model.py | save_model(), load_model() |
| 추론(인퍼런스) | inference_service.py | InferenceService.predict() |
| 결과 해석 | inference_service.py | interpret_prediction(), feature_importance() | 
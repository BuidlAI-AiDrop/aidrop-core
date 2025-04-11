# AI 파이프라인

블록체인 사용자 분석 AI 파이프라인 테스트용 프로젝트입니다.

## 설치

```bash
pip install -e .
```

## 테스트 실행

```bash
python -m unittest discover -s tests
```

## 사용 방법

### 모델 학습

```bash
python -m ai_pipeline.main train path/to/training_data.csv --output ./models --clusters 5
```

### 주소 분석

```bash
python -m ai_pipeline.main batch 0x123456789abcdef --output ./results
```

### 배치 분석

```bash
python -m ai_pipeline.main batch path/to/addresses.txt --output ./results
``` 
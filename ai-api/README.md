# AI API 서버

블록체인 주소 분석 및 프로필 이미지 생성 API 서버입니다.

## 기능

- 블록체인 주소 분석 요청 (비동기 처리)
- 분석 결과 조회 (이미지 URL, 사용자 유형, 특성 등)
- AWS S3를 활용한 이미지 저장
- Supabase를 활용한 분석 결과 저장

## 설치 및 실행

### 의존성 설치

```bash
pip install -r requirements.txt
```

### 환경 변수 설정

다음 환경 변수를 설정하세요:

- `SUPABASE_URL`: Supabase URL
- `SUPABASE_KEY`: Supabase API 키
- `AWS_ACCESS_KEY`: AWS 접근 키
- `AWS_SECRET_KEY`: AWS 비밀 키
- `AWS_BUCKET_NAME`: AWS S3 버킷 이름 (기본값: aidrop-profiles)
- `OPENAI_API_KEY`: OpenAI API 키 (프로필 이미지 생성용)

### 서버 실행

```bash
cd ai-api
python app.py
```

또는 직접 uvicorn으로 실행:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API 엔드포인트

### 분석 요청

```
POST /api/analyze
```

**요청 본문:**

```json
{
  "sourceAddress": "0x1234567890abcdef1234567890abcdef12345678",
  "storyAddress": "0xabcdef1234567890abcdef1234567890abcdef12",
  "sourceChainId": "1"
}
```

**응답:**

```json
{
  "status": "accepted",
  "message": "분석 요청이 접수되었습니다. 결과를 확인하려면 /api/result를 사용하세요.",
  "requestId": "1_0xabcdef1234567890abcdef1234567890abcdef12_1650123456"
}
```

### 분석 결과 조회

```
POST /api/result
```

**요청 본문:**

```json
{
  "sourceAddress": "0x1234567890abcdef1234567890abcdef12345678",
  "storyAddress": "0xabcdef1234567890abcdef1234567890abcdef12",
  "sourceChainId": "1"
}
```

**응답 (처리 중):**

```json
{
  "sourceAddress": "0x1234567890abcdef1234567890abcdef12345678",
  "storyAddress": "0xabcdef1234567890abcdef1234567890abcdef12",
  "sourceChainId": "1",
  "status": "processing"
}
```

**응답 (완료):**

```json
{
  "sourceAddress": "0x1234567890abcdef1234567890abcdef12345678",
  "storyAddress": "0xabcdef1234567890abcdef1234567890abcdef12",
  "sourceChainId": "1",
  "imageUrl": "https://aidrop-profiles.s3.amazonaws.com/profiles/1/0xabcdef1234567890abcdef1234567890abcdef12/character_123456789.png",
  "userType": "D-T-A-I",
  "traits": {
    "defi_focus": 0.75,
    "nft_interest": 0.25,
    "risk_appetite": 0.8,
    "community_participation": 0.4
  },
  "cluster": 3,
  "analysisTime": "2023-04-15T12:34:56.789Z",
  "status": "completed"
}
```

## Supabase 데이터베이스 스키마

### analysis_requests 테이블

| 필드 | 타입 | 설명 |
|------|------|------|
| request_id | string | 요청 ID (기본 키) |
| source_address | string | 소스 주소 |
| story_address | string | 스토리 주소 |
| source_chain_id | string | 소스 체인 ID |
| status | string | 처리 상태 (pending, processing, completed, failed) |
| message | string | 상태 메시지 |
| image_url | string | 이미지 URL |
| user_type | string | 사용자 유형 |
| created_at | timestamp | 생성 시간 |
| updated_at | timestamp | 업데이트 시간 |

### analysis_results 테이블

| 필드 | 타입 | 설명 |
|------|------|------|
| id | uuid | 고유 ID (기본 키) |
| request_id | string | 요청 ID |
| source_address | string | 소스 주소 |
| story_address | string | 스토리 주소 |
| source_chain_id | string | 소스 체인 ID |
| image_url | string | 이미지 URL |
| user_type | string | 사용자 유형 |
| cluster | integer | 클러스터 |
| traits | json | 사용자 특성 |
| full_result | json | 전체 분석 결과 |
| status | string | 처리 상태 |
| created_at | timestamp | 생성 시간 |
| updated_at | timestamp | 업데이트 시간 | 
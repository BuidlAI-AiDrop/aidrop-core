FROM python:3.9-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .
COPY ../profile_generator.py /app/
COPY ../ai_pipeline /app/ai_pipeline

# 데이터 디렉토리 생성
RUN mkdir -p /app/test_data /app/results/profiles /app/results/api

# 애플리케이션 실행
CMD ["python", "app.py"]

EXPOSE 8000 
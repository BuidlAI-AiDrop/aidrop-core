"""
AI 파이프라인 패키지 설정
"""

from setuptools import setup, find_packages

setup(
    name="ai_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
    ],
    author="AI Pipeline Team",
    author_email="test@example.com",
    description="블록체인 사용자 분석 AI 파이프라인",
    keywords="blockchain, ai, analytics",
    python_requires=">=3.8",
) 
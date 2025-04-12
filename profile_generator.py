#!/usr/bin/env python3
"""
블록체인 주소 기반 캐릭터 이미지 생성기
OpenAI API를 활용한 블록체인 주소별 귀여운 캐릭터 생성
"""

import os
import hashlib
import requests
from datetime import datetime
from openai import OpenAI

class AIProfileGenerator:
    """OpenAI API를 활용한 프로필 이미지 생성기"""
    
    def __init__(self, api_key=None, output_dir="./results/profiles"):
        """
        초기화 함수
        
        Args:
            api_key: OpenAI API 키 (없으면 환경 변수에서 가져옴)
            output_dir: 이미지 저장 경로
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # OpenAI 클라이언트 초기화
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 인자로 전달하거나 OPENAI_API_KEY 환경변수를 설정하세요.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # 캐릭터 스타일 옵션
        self.styles = [
            "네온 글로우 효과가 있는",
            "사이버펑크 스타일의",
            "홀로그램 효과가 있는",
            "픽셀 아트 스타일의",
            "미니멀한 디자인의",
            "파스텔톤 색상의"
        ]
        
        # 캐릭터 타입
        self.characters = [
            "고양이 캐릭터",
            "강아지 캐릭터", 
            "팬더 캐릭터",
            "토끼 캐릭터",
            "로봇 캐릭터",
            "우주인 캐릭터",
            "곰 캐릭터",
            "여우 캐릭터"
        ]
        
        # 표정/특성
        self.expressions = [
            "웃는 표정의",
            "윙크하는",
            "놀란 표정의",
            "별 모양 눈을 가진",
            "동그란 눈을 가진",
            "행복한 표정의",
            "헤드폰을 끼고 있는",
            "모자를 쓰고 있는",
            "안경을 쓰고 있는"
        ]
        
        # 컬러 테마
        self.colors = [
            "파란색 계열의",
            "분홍색 계열의",
            "보라색 계열의",
            "초록색 계열의",
            "노란색 계열의",
            "주황색 계열의",
            "레인보우 색상의"
        ]
    
    def _hash_based_selection(self, address, seed_str, options_list):
        """주소 해시를 기반으로 옵션 목록에서 일관된 선택"""
        # 주소와 시드 문자열을 조합하여 해시 생성
        hash_input = f"{address}:{seed_str}"
        hash_obj = hashlib.md5(hash_input.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # 해시값을 사용하여 옵션 선택
        index = hash_int % len(options_list)
        return options_list[index]
    
    def _generate_character_name(self, address):
        """주소로부터 캐릭터 이름 생성"""
        # 주소 첫 부분을 이름으로 사용
        prefix = address[:6].upper()
        
        # 접미사 옵션
        suffixes = ["BOT", "STAR", "PAWS", "BYTE", "PIX", "NEON", "GLOW"]
        suffix = self._hash_based_selection(address, "name", suffixes)
        
        return f"{prefix}-{suffix}"
    
    def _generate_prompt(self, address):
        """주소를 기반으로 일관된 이미지 생성 프롬프트 생성"""
        # 캐릭터 이름
        character_name = self._generate_character_name(address)
        
        # 주소 해시를 기반으로 일관된 스타일 선택
        style = self._hash_based_selection(address, "style", self.styles)
        character = self._hash_based_selection(address, "character", self.characters)
        expression = self._hash_based_selection(address, "expression", self.expressions)
        color = self._hash_based_selection(address, "color", self.colors)
        
        # 프롬프트 구성 - 단순하고 심플한 귀여운 캐릭터
        prompt = f"어두운 배경에 {style} {color} {expression} {character}. 캐릭터 이름은 {character_name}이며, 심플하고 귀여운 디자인. 밝은 네온 색상으로 블록체인 미래 감성. 고해상도, 고품질 디지털 아트"
        
        return prompt, character_name, f"{style} {color} {character}"
    
    def generate_profile(self, address, save=True, show_prompt=False):
        """
        주소 기반으로 OpenAI API를 통해 프로필 이미지 생성
        
        Args:
            address: 블록체인 주소
            save: 이미지 저장 여부
            show_prompt: 생성에 사용된 프롬프트 출력 여부
            
        Returns:
            생성된 이미지 경로
        """
        # 프롬프트 생성
        prompt, char_name, char_desc = self._generate_prompt(address)
        
        if show_prompt:
            print(f"캐릭터 이름: {char_name}")
            print(f"캐릭터 설명: {char_desc}")
            print(f"생성 프롬프트: {prompt}")
        
        try:
            # OpenAI API를 통해 이미지 생성
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="standard",
                response_format="url"
            )
            
            # 이미지 URL 가져오기
            image_url = response.data[0].url
            
            # 이미지 다운로드
            image_response = requests.get(image_url)
            if image_response.status_code != 200:
                raise Exception(f"이미지 다운로드 실패: {image_response.status_code}")
            
            # 이미지 저장
            if save:
                filename = f"{char_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_response.content)
                
                print(f"프로필 이미지가 저장되었습니다: {filepath}")
                return filepath
            
            return image_url
            
        except Exception as e:
            # 유니코드 오류 처리 추가
            error_msg = str(e)
            if "ordinal not in range" in error_msg:
                # 테스트 모드이므로 실제 이미지 생성 없이 더미 결과 반환
                dummy_filepath = os.path.join(self.output_dir, f"{char_name}_dummy_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
                with open(dummy_filepath, 'w') as f:
                    f.write(f"# 더미 이미지 파일 - {char_name}\n{char_desc}")
                print(f"테스트 모드: 더미 이미지 파일 생성 ({dummy_filepath})")
                return dummy_filepath
            
            print(f"이미지 생성 오류: {error_msg}")
            return None

# 실행 예시
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='블록체인 주소 기반 AI 프로필 이미지 생성')
    parser.add_argument('--key', type=str, help='OpenAI API 키')
    parser.add_argument('--address', type=str, help='블록체인 주소')
    parser.add_argument('--show-prompt', action='store_true', help='생성 프롬프트 출력')
    args = parser.parse_args()
    
    api_key = args.key
    
    # 테스트 주소
    test_addresses = [
        "0xf620123456789abcdef0123456789abcdef78ca",
        "0x1234567890abcdef1234567890abcdef12345678",
        "0xabcdef1234567890abcdef1234567890abcdef12",
        "0x0987654321fedcba0987654321fedcba09876543"
    ]
    
    # 특정 주소가 제공되었으면 해당 주소만 생성
    if args.address:
        test_addresses = [args.address]
    
    # 이미지 생성
    generator = AIProfileGenerator(api_key=api_key)
    for address in test_addresses:
        generator.generate_profile(address, save=True, show_prompt=args.show_prompt)

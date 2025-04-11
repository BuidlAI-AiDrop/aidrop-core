#!/usr/bin/env python3
"""
기본 기능 테스트
"""

import unittest
import tempfile
import os
import shutil
import numpy as np

class TestBasic(unittest.TestCase):
    """기본 기능 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.test_dir)
    
    def test_imports(self):
        """모듈 임포트 테스트"""
        try:
            import pandas as pd
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            
            # 기본적인 라이브러리 테스트
            df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            self.assertEqual(len(df), 3)
            
            # 넘파이 배열 생성
            arr = np.array([1, 2, 3])
            self.assertEqual(arr.shape, (3,))
            
            print("기본 라이브러리 임포트 성공")
        except ImportError as e:
            self.fail(f"모듈 임포트 실패: {str(e)}")
    
    def test_file_operations(self):
        """파일 작업 테스트"""
        test_file = os.path.join(self.test_dir, "test.txt")
        
        # 파일 쓰기
        with open(test_file, 'w') as f:
            f.write("테스트 데이터")
        
        # 파일 확인
        self.assertTrue(os.path.exists(test_file))
        
        # 파일 읽기
        with open(test_file, 'r') as f:
            content = f.read()
        
        self.assertEqual(content, "테스트 데이터")
        print("파일 작업 테스트 성공")

if __name__ == '__main__':
    unittest.main() 
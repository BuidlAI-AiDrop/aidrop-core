import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set


class MarketingTargetExtractor:
    """마케팅 타겟 사용자 추출 클래스

    기존 사용자 분석 결과(results/analysis/)에서 다양한 타겟 기준에 맞는 사용자 목록을 추출
    """

    def __init__(self, analysis_dir: str = "results/analysis"):
        """초기화

        Args:
            analysis_dir: 분석 결과 저장 디렉토리 경로
        """
        self.analysis_dir = analysis_dir
        self.chain_dirs = self._get_chain_dirs()
        self.combined_data = None
        self.load_all_data()

    def _get_chain_dirs(self) -> List[str]:
        """사용 가능한 체인 데이터 디렉토리 목록 반환"""
        return [d for d in os.listdir(self.analysis_dir) 
                if os.path.isdir(os.path.join(self.analysis_dir, d))]
    
    def load_all_data(self) -> None:
        """모든 체인의 모든 분석 데이터를 로드하여 DataFrame으로 변환"""
        data = []
        
        for chain_id in self.chain_dirs:
            chain_dir = os.path.join(self.analysis_dir, chain_id)
            for file in os.listdir(chain_dir):
                if file.endswith('.json'):
                    address = file.replace('.json', '')
                    file_path = os.path.join(chain_dir, file)
                    
                    try:
                        with open(file_path, 'r') as f:
                            user_data = json.load(f)
                            user_data['address'] = address
                            user_data['chain_id'] = chain_id
                            data.append(user_data)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        self.combined_data = pd.DataFrame(data)
        print(f"Loaded {len(self.combined_data)} user profiles from {len(self.chain_dirs)} chains")
    
    def filter_by_mbti(self, mbti_patterns: List[str]) -> pd.DataFrame:
        """MBTI 유형으로 필터링

        Args:
            mbti_patterns: 필터링할 MBTI 패턴 목록 (예: ['D-H-S', 'D-T-S'])
        
        Returns:
            필터링된 사용자 DataFrame
        """
        if self.combined_data is None or len(self.combined_data) == 0:
            return pd.DataFrame()
            
        return self.combined_data[self.combined_data['mbti'].isin(mbti_patterns)]
    
    def filter_by_cluster(self, cluster_ids: List[int]) -> pd.DataFrame:
        """클러스터 ID로 필터링

        Args:
            cluster_ids: 필터링할 클러스터 ID 목록
        
        Returns:
            필터링된 사용자 DataFrame
        """
        if self.combined_data is None or len(self.combined_data) == 0:
            return pd.DataFrame()
            
        return self.combined_data[self.combined_data['cluster'].isin(cluster_ids)]
    
    def get_defi_long_term_holders(self) -> pd.DataFrame:
        """DeFi 장기 홀더(D-H-*) 사용자 추출

        Returns:
            DeFi 장기 홀더 사용자 DataFrame
        """
        defi_holders = self.combined_data[self.combined_data['mbti'].str.startswith('D-H')]
        return defi_holders
    
    def get_nft_enthusiasts(self) -> pd.DataFrame:
        """NFT 열정 사용자(N-*) 추출

        Returns:
            NFT 열정 사용자 DataFrame
        """
        nft_users = self.combined_data[self.combined_data['mbti'].str.startswith('N')]
        return nft_users

    def get_community_involved_users(self) -> pd.DataFrame:
        """커뮤니티 참여 사용자(*-*-*-C) 추출

        Returns:
            커뮤니티 참여 사용자 DataFrame
        """
        community_users = self.combined_data[self.combined_data['mbti'].str.endswith('C')]
        return community_users
    
    def get_aggressive_traders(self) -> pd.DataFrame:
        """공격적 트레이더(*-T-A-*) 추출

        Returns:
            공격적 트레이더 DataFrame
        """
        # MBTI 형식이 4개 부분으로 구성된 경우 (*-T-A-*)
        if '-' in self.combined_data['mbti'].iloc[0] and len(self.combined_data['mbti'].iloc[0].split('-')) >= 3:
            pattern = '-T-A-'
            aggressive_traders = self.combined_data[self.combined_data['mbti'].str.contains(pattern)]
            return aggressive_traders
        
        # 3개 부분으로 구성된 경우는 다른 패턴 적용 (*-T-A)
        elif '-' in self.combined_data['mbti'].iloc[0] and len(self.combined_data['mbti'].iloc[0].split('-')) == 3:
            traders = self.combined_data[self.combined_data['mbti'].str.contains('-T-')]
            aggressive = traders[traders['mbti'].str.endswith('A')]
            return aggressive
        
        return pd.DataFrame()
    
    def custom_filter(self, filter_func) -> pd.DataFrame:
        """사용자 정의 필터 함수로 필터링

        Args:
            filter_func: 필터링 함수 (DataFrame을 받아 필터링된 DataFrame을 반환)
        
        Returns:
            필터링된 사용자 DataFrame
        """
        return filter_func(self.combined_data)
    
    def export_targets(self, targets: pd.DataFrame, output_file: str) -> None:
        """타겟 사용자 목록을 파일로 내보내기

        Args:
            targets: 타겟 사용자 DataFrame
            output_file: 출력 파일 경로
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # CSV 포맷으로 저장
        if output_file.endswith('.csv'):
            targets.to_csv(output_file, index=False)
        # JSON 포맷으로 저장
        elif output_file.endswith('.json'):
            targets.to_json(output_file, orient='records')
        else:
            # 기본 포맷은 JSON
            targets.to_json(f"{output_file}.json", orient='records')
        
        print(f"Exported {len(targets)} target users to {output_file}") 
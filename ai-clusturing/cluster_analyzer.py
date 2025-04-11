"""
클러스터 분석 및 사용자 특성 생성 모듈
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

# 상대 경로 임포트를 절대 경로 임포트로 변경
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import setup_logger

logger = setup_logger(__name__)

class ClusterFeatureAnalyzer:
    """클러스터별 특성 분석 및 사용자 특징 추출 클래스"""
    
    # 사용자 특성 정의
    USER_TRAITS = {
        # 금융 vs 창작 축
        'defi_focus': 'DeFi 프로토콜 활동과 금융 거래에 중점을 둔 사용자 (D)',
        'nft_focus': 'NFT 수집과 창작 활동에 중점을 둔 사용자 (N)',
        
        # 트레이딩 vs 홀딩 축
        'short_term_trader': '단기 거래와 빈번한 트랜잭션을 보이는 사용자 (T)',
        'long_term_holder': '장기 보유 성향과 낮은 거래 빈도를 보이는 사용자 (H)',
        
        # 위험 vs 안정성 축
        'risk_taking': '새로운 프로토콜 사용과 고위험 활동을 선호하는 사용자 (A)',
        'security_focused': '검증된 프로토콜만 사용하고 안정적인 자산을 선호하는 사용자 (S)',
        
        # 커뮤니티 vs 독립성 축
        'community_builder': '다양한 DAO 참여와 사회적 상호작용이 많은 사용자 (C)',
        'independent_actor': '독립적인 활동을 선호하고 상호작용이 제한적인 사용자 (I)',
        
        # 복합 유형 예시
        'defi_trader_risk': 'DeFi에서 단기 거래와 위험 감수 성향을 보이는 사용자 (D-T-A)',
        'nft_holder_community': 'NFT 장기 보유와 커뮤니티 활동에 중점을 두는 사용자 (N-H-C)',
        'defi_holder_security': '안정적인 DeFi 프로토콜에 장기 투자하는 사용자 (D-H-S)',
        'nft_trader_risk': '위험을 감수하며 NFT 거래를 활발히 하는 사용자 (N-T-A)',
        'defi_holder_community': 'DeFi 장기 투자와 DAO 참여가 활발한 사용자 (D-H-C)',
        'nft_holder_independent': 'NFT를 장기 보유하며 독립적으로 활동하는 사용자 (N-H-I)',
        'defi_trader_independent': 'DeFi 단기 거래를 독립적으로 수행하는 사용자 (D-T-I)',
        'nft_trader_community': 'NFT 거래와 커뮤니티 활동이 모두 활발한 사용자 (N-T-C)'
    }
    
    def __init__(self, output_dir: str = 'cluster_profiles'):
        """
        Args:
            output_dir: 결과 저장 디렉토리
        """
        self.logger = logger
        self.output_dir = output_dir
        self.cluster_profiles = {}
        self.feature_thresholds = {}
        
        # 저장 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_clusters(self, features_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """
        클러스터별 특성 분석
        
        Args:
            features_df: 특성 DataFrame
            labels: 클러스터 레이블
            
        Returns:
            클러스터별 프로필 딕셔너리
        """
        # 클러스터 레이블 열 추가
        df = features_df.copy()
        df['cluster'] = labels
        
        # 고유 클러스터 목록 (노이즈 포인트 제외)
        unique_clusters = sorted([c for c in np.unique(labels) if c != -1])
        
        # 클러스터별 특성 요약
        cluster_summaries = {}
        for cluster in unique_clusters:
            cluster_df = df[df['cluster'] == cluster]
            
            # 기본 통계 요약
            summary = {}
            for col in df.columns:
                if col != 'cluster' and pd.api.types.is_numeric_dtype(df[col]):
                    summary[col] = {
                        'mean': float(cluster_df[col].mean()),
                        'median': float(cluster_df[col].median()),
                        'std': float(cluster_df[col].std()),
                        'min': float(cluster_df[col].min()),
                        'max': float(cluster_df[col].max()),
                        'count': int(cluster_df[col].count())
                    }
            
            cluster_summaries[f'cluster_{cluster}'] = {
                'summary': summary,
                'size': len(cluster_df),
                'percentage': len(cluster_df) / len(df) * 100
            }
        
        # 클러스터 간 유의미한 특성 차이 식별
        significant_features = self._identify_significant_features(df, unique_clusters)
        
        # 각 클러스터의 특화된 특성 식별
        specialized_traits = self._identify_specialized_traits(df, unique_clusters)
        
        # 클러스터 프로필 생성
        self.cluster_profiles = {}
        for cluster in unique_clusters:
            cluster_key = f'cluster_{cluster}'
            
            # 클러스터 크기
            size = cluster_summaries[cluster_key]['size']
            percentage = cluster_summaries[cluster_key]['percentage']
            
            # 프로필 특성 추출
            profile = {
                'size': size,
                'percentage': percentage,
                'significant_features': significant_features.get(cluster, {}),
                'specialized_traits': specialized_traits.get(cluster, {})
            }
            
            # 요약 특성 추가
            summary_data = cluster_summaries[cluster_key]['summary']
            key_metrics = {
                'txn_per_day': summary_data.get('txn_per_day', {}).get('mean', 0),
                'token_count': summary_data.get('token_count', {}).get('mean', 0),
                'nft_count': summary_data.get('nft_count', {}).get('mean', 0),
                'nft_ratio': summary_data.get('nft_ratio', {}).get('mean', 0),
                'defi_interaction_count': summary_data.get('defi_interaction_count', {}).get('mean', 0),
                'unique_contracts': summary_data.get('unique_contracts', {}).get('mean', 0),
                'unique_counterparties': summary_data.get('unique_counterparties', {}).get('mean', 0),
                'avg_txn_value': summary_data.get('avg_txn_value', {}).get('mean', 0),
                'gas_to_value_ratio': summary_data.get('gas_to_value_ratio', {}).get('mean', 0)
            }
            
            profile['key_metrics'] = key_metrics
            
            # 클러스터 사용자 특성 식별
            user_traits = self._identify_user_traits(cluster, key_metrics, specialized_traits.get(cluster, {}))
            profile['user_traits'] = user_traits
            
            self.cluster_profiles[cluster_key] = profile
        
        self.logger.info(f"{len(unique_clusters)}개 클러스터 분석 완료")
        return self.cluster_profiles
    
    def _identify_significant_features(self, df: pd.DataFrame, clusters: List[int]) -> Dict[int, Dict[str, float]]:
        """
        클러스터 간 유의미한 특성 차이 식별
        
        Args:
            df: 클러스터 레이블이 포함된 특성 DataFrame
            clusters: 클러스터 번호 목록
            
        Returns:
            클러스터별 유의미한 특성 차이 딕셔너리
        """
        significant_features = {}
        
        for cluster in clusters:
            # 현재 클러스터 vs 나머지
            cluster_df = df[df['cluster'] == cluster]
            rest_df = df[df['cluster'] != cluster]
            
            cluster_significant = {}
            
            for col in df.columns:
                if col != 'cluster' and pd.api.types.is_numeric_dtype(df[col]):
                    # t-검정으로 통계적 유의성 확인
                    if len(cluster_df) > 1 and len(rest_df) > 1:
                        try:
                            t_stat, p_value = ttest_ind(
                                cluster_df[col].dropna(), 
                                rest_df[col].dropna(), 
                                equal_var=False
                            )
                            
                            # p-value < 0.05는 통계적으로 유의미
                            if p_value < 0.05:
                                # 평균 비교로 방향성 확인
                                cluster_mean = cluster_df[col].mean()
                                rest_mean = rest_df[col].mean()
                                
                                direction = "higher" if cluster_mean > rest_mean else "lower"
                                magnitude = abs(cluster_mean - rest_mean) / (rest_mean if rest_mean != 0 else 1)
                                
                                cluster_significant[col] = {
                                    'p_value': float(p_value),
                                    'direction': direction,
                                    'magnitude': float(magnitude),
                                    'cluster_mean': float(cluster_mean),
                                    'overall_mean': float(rest_mean)
                                }
                        except Exception as e:
                            self.logger.warning(f"t-검정 오류 ({col}): {str(e)}")
            
            # 유의성 기준으로 정렬
            sorted_features = {k: v for k, v in sorted(
                cluster_significant.items(), key=lambda item: item[1]['p_value']
            )}
            
            # 상위 10개 특성
            significant_features[cluster] = dict(list(sorted_features.items())[:10])
        
        return significant_features
    
    def _identify_specialized_traits(self, df: pd.DataFrame, clusters: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        클러스터별 특화된 특성 식별
        
        Args:
            df: 클러스터 레이블이 포함된 특성 DataFrame
            clusters: 클러스터 번호 목록
            
        Returns:
            클러스터별 특화된 특성 딕셔너리
        """
        # 특화 특성 정의 (직접 매핑)
        trait_features = {
            'nft_activity': ['nft_count', 'nft_ratio', 'nft_marketplace_count'],
            'defi_activity': ['defi_interaction_count'],
            'transaction_volume': ['total_txn_count', 'txn_per_day'],
            'transaction_value': ['avg_txn_value', 'total_value'],
            'diversity': ['unique_counterparties', 'unique_contracts', 'token_count'],
            'efficiency': ['gas_to_value_ratio', 'avg_gas_price'],
            'holding_behavior': ['token_count', 'erc20_count', 'account_age_days'],
            'recency': ['days_since_last_activity']
        }
        
        specialized_traits = {}
        
        for cluster in clusters:
            cluster_df = df[df['cluster'] == cluster]
            rest_df = df[df['cluster'] != cluster]
            
            traits = {}
            
            for trait_name, feature_list in trait_features.items():
                trait_scores = []
                
                for feature in feature_list:
                    if feature in df.columns:
                        cluster_mean = cluster_df[feature].mean()
                        rest_mean = rest_df[feature].mean()
                        
                        if rest_mean != 0:
                            relative_score = (cluster_mean - rest_mean) / rest_mean
                        else:
                            relative_score = cluster_mean
                        
                        trait_scores.append(relative_score)
                
                if trait_scores:
                    avg_trait_score = np.mean(trait_scores)
                    
                    # 특성 점수 정규화 (모든 클러스터 중 백분위)
                    all_means = []
                    for c in clusters:
                        c_means = []
                        for feature in feature_list:
                            if feature in df.columns:
                                c_means.append(df[df['cluster'] == c][feature].mean())
                        
                        if c_means:
                            all_means.append(np.mean(c_means))
                    
                    if all_means:
                        percentile = sum(1 for m in all_means if m < avg_trait_score) / len(all_means)
                    else:
                        percentile = 0.5
                    
                    traits[trait_name] = {
                        'score': float(avg_trait_score),
                        'percentile': float(percentile)
                    }
            
            specialized_traits[cluster] = traits
        
        return specialized_traits
    
    def _identify_user_traits(self, cluster: int, metrics: Dict[str, float], 
                             specialized_traits: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        클러스터에 적합한 사용자 특성 식별
        
        Args:
            cluster: 클러스터 번호
            metrics: 주요 지표
            specialized_traits: 특화된 특성
            
        Returns:
            사용자 특성과 적합도 점수 딕셔너리
        """
        user_traits = {}
        
        # 1. 금융 vs 창작 축 (D-N)
        
        # DeFi 포커스 (D)
        defi_score = 0.0
        if 'defi_activity' in specialized_traits:
            defi_score += specialized_traits['defi_activity'].get('percentile', 0) * 0.6
        defi_score += min(metrics.get('defi_interaction_count', 0) / 10, 0.4)  # DeFi 상호작용
        user_traits['defi_focus'] = float(defi_score)
        
        # NFT 포커스 (N)
        nft_score = 0.0
        if 'nft_activity' in specialized_traits:
            nft_score += specialized_traits['nft_activity'].get('percentile', 0) * 0.6
        nft_score += min(metrics.get('nft_ratio', 0) * 10, 0.4)  # NFT 비율
        user_traits['nft_focus'] = float(nft_score)
        
        # 2. 트레이딩 vs 홀딩 축 (T-H)
        
        # 단기 트레이더 (T)
        trader_score = 0.0
        if 'transaction_volume' in specialized_traits:
            trader_score += specialized_traits['transaction_volume'].get('percentile', 0) * 0.5
        trader_score += min(metrics.get('txn_per_day', 0) / 10, 0.5)  # 일일 트랜잭션
        user_traits['short_term_trader'] = float(trader_score)
        
        # 장기 보유자 (H)
        holder_score = 0.0
        if 'holding_behavior' in specialized_traits:
            holder_score += specialized_traits['holding_behavior'].get('percentile', 0) * 0.6
        # 트랜잭션 빈도가 낮을수록 holder 점수 높음
        holder_score += max(0, 0.4 * (1 - min(metrics.get('txn_per_day', 0) / 5, 1)))
        user_traits['long_term_holder'] = float(holder_score)
        
        # 3. 위험 vs 안정성 축 (A-S)
        
        # 위험 감수 (A)
        risk_score = 0.0
        if 'new_protocols' in specialized_traits:
            risk_score += specialized_traits['new_protocols'].get('percentile', 0) * 0.4
        # 다양한 컨트랙트와 상호작용할수록 위험 감수 성향
        risk_score += min(metrics.get('unique_contracts', 0) / 30, 0.3)
        # 높은 변동성은 위험 감수 성향 암시
        if 'value_std' in metrics and 'avg_txn_value' in metrics and metrics['avg_txn_value'] > 0:
            volatility = metrics['value_std'] / metrics['avg_txn_value']
            risk_score += min(volatility / 5, 0.3)
        user_traits['risk_taking'] = float(risk_score)
        
        # 보안 중심 (S)
        security_score = 0.0
        # 안정적인 자산 선호, 검증된 컨트랙트 사용
        security_score = 1.0 - risk_score * 0.7  # 완전한 반대는 아님
        if 'established_protocols' in specialized_traits:
            security_score += specialized_traits['established_protocols'].get('percentile', 0) * 0.3
        user_traits['security_focused'] = float(security_score)
        
        # 4. 커뮤니티 vs 독립성 축 (C-I)
        
        # 커뮤니티 빌더 (C)
        community_score = 0.0
        # 다양한 상대방과 상호작용
        community_score += min(metrics.get('unique_counterparties', 0) / 50, 0.5)
        if 'social_interaction' in specialized_traits:
            community_score += specialized_traits['social_interaction'].get('percentile', 0) * 0.5
        user_traits['community_builder'] = float(community_score)
        
        # 독립적 행위자 (I)
        independent_score = 0.0
        # 제한된 상호작용
        independent_score = 1.0 - community_score * 0.8  # 완전한 반대는 아님
        if 'solo_activity' in specialized_traits:
            independent_score += specialized_traits['solo_activity'].get('percentile', 0) * 0.2
        user_traits['independent_actor'] = float(independent_score)
        
        # 5. 복합 유형 계산
        
        # DeFi-Trader-Risk (D-T-A)
        defi_trader_risk = (defi_score * 0.4 + trader_score * 0.3 + risk_score * 0.3)
        user_traits['defi_trader_risk'] = float(defi_trader_risk)
        
        # NFT-Holder-Community (N-H-C)
        nft_holder_community = (nft_score * 0.4 + holder_score * 0.3 + community_score * 0.3)
        user_traits['nft_holder_community'] = float(nft_holder_community)
        
        # DeFi-Holder-Security (D-H-S)
        defi_holder_security = (defi_score * 0.4 + holder_score * 0.3 + security_score * 0.3)
        user_traits['defi_holder_security'] = float(defi_holder_security)
        
        # NFT-Trader-Risk (N-T-A)
        nft_trader_risk = (nft_score * 0.4 + trader_score * 0.3 + risk_score * 0.3)
        user_traits['nft_trader_risk'] = float(nft_trader_risk)
        
        # DeFi-Holder-Community (D-H-C)
        defi_holder_community = (defi_score * 0.4 + holder_score * 0.3 + community_score * 0.3)
        user_traits['defi_holder_community'] = float(defi_holder_community)
        
        # NFT-Holder-Independent (N-H-I)
        nft_holder_independent = (nft_score * 0.4 + holder_score * 0.3 + independent_score * 0.3)
        user_traits['nft_holder_independent'] = float(nft_holder_independent)
        
        # DeFi-Trader-Independent (D-T-I)
        defi_trader_independent = (defi_score * 0.4 + trader_score * 0.3 + independent_score * 0.3)
        user_traits['defi_trader_independent'] = float(defi_trader_independent)
        
        # NFT-Trader-Community (N-T-C)
        nft_trader_community = (nft_score * 0.4 + trader_score * 0.3 + community_score * 0.3)
        user_traits['nft_trader_community'] = float(nft_trader_community)
        
        return user_traits
    
    def save_cluster_profiles(self, version: str = "v1"):
        """
        클러스터 프로필 저장
        
        Args:
            version: 프로필 버전
        """
        version_dir = os.path.join(self.output_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # 클러스터 프로필 저장
        profile_path = os.path.join(version_dir, "cluster_profiles.json")
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(self.cluster_profiles, f, ensure_ascii=False, indent=2)
        
        # 사용자 특성 설명 저장
        traits_path = os.path.join(version_dir, "user_traits.json")
        with open(traits_path, 'w', encoding='utf-8') as f:
            json.dump(self.USER_TRAITS, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"클러스터 프로필 저장 완료: {profile_path}")
    
    def load_cluster_profiles(self, version: str = "v1") -> bool:
        """
        저장된 클러스터 프로필 로드
        
        Args:
            version: 프로필 버전
            
        Returns:
            로드 성공 여부
        """
        profile_path = os.path.join(self.output_dir, version, "cluster_profiles.json")
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                self.cluster_profiles = json.load(f)
            
            self.logger.info(f"클러스터 프로필 로드 완료: {profile_path}")
            return True
        except Exception as e:
            self.logger.error(f"클러스터 프로필 로드 실패: {str(e)}")
            return False
    
    def get_user_traits(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        사용자 특성 점수 계산
        
        Args:
            features: 사용자 특성 벡터
            
        Returns:
            사용자 특성 점수 (0~1)
        """
        if not self.cluster_profiles:
            self.logger.error("클러스터 프로필이 로드되지 않았습니다.")
            return {}
        
        # 메트릭 추출
        metrics = {
            'txn_per_day': features.get('txn_per_day', 0),
            'token_count': features.get('token_count', 0),
            'nft_count': features.get('nft_count', 0),
            'nft_ratio': features.get('nft_ratio', 0),
            'defi_interaction_count': features.get('defi_interaction_count', 0),
            'unique_contracts': features.get('unique_contracts', 0),
            'unique_counterparties': features.get('unique_counterparties', 0),
            'avg_txn_value': features.get('avg_txn_value', 0),
            'value_std': features.get('value_std', 0),
            'gas_to_value_ratio': features.get('gas_to_value_ratio', 0),
            'account_age_days': features.get('account_age_days', 0),
            'days_since_last_activity': features.get('days_since_last_activity', 0)
        }
        
        # 1. 금융 vs 창작 축 (D-N)
        
        # DeFi 포커스 (D)
        defi_score = min(1.0, features.get('defi_interaction_count', 0) / 20)
        
        # NFT 포커스 (N)
        nft_score = min(1.0, features.get('nft_ratio', 0) * 5 + 
                       features.get('nft_count', 0) / 10)
        
        # 2. 트레이딩 vs 홀딩 축 (T-H)
        
        # 단기 트레이더 (T)
        trader_score = min(1.0, 
                          features.get('txn_per_day', 0) / 10 * 0.6 + 
                          features.get('total_txn_count', 0) / 100 * 0.4)
        
        # 장기 보유자 (H)
        holder_score = min(1.0, 
                          features.get('account_age_days', 0) / 365 * 0.5 + 
                          (1 - min(features.get('txn_per_day', 0) / 5, 1)) * 0.5)
        
        # 3. 위험 vs 안정성 축 (A-S)
        
        # 위험 감수 (A)
        risk_score = 0.0
        # 다양한 컨트랙트와 상호작용할수록 위험 감수 성향
        risk_score += min(features.get('unique_contracts', 0) / 30, 0.5)
        # 높은 변동성은 위험 감수 성향 암시
        if metrics['avg_txn_value'] > 0:
            volatility = metrics['value_std'] / metrics['avg_txn_value']
            risk_score += min(volatility / 5, 0.5)
        risk_score = min(1.0, risk_score)
        
        # 보안 중심 (S)
        security_score = 1.0 - risk_score * 0.7  # 완전한 반대는 아님
        
        # 4. 커뮤니티 vs 독립성 축 (C-I)
        
        # 커뮤니티 빌더 (C)
        community_score = min(1.0, 
                             features.get('unique_counterparties', 0) / 50 * 0.7 + 
                             features.get('unique_contracts', 0) / 30 * 0.3)
        
        # 독립적 행위자 (I)
        independent_score = 1.0 - community_score * 0.8  # 완전한 반대는 아님
        
        # 5. 복합 유형 계산
        
        # 결과 사용자 특성 점수
        user_traits = {
            # 기본 축
            'defi_focus': float(defi_score),
            'nft_focus': float(nft_score),
            'short_term_trader': float(trader_score),
            'long_term_holder': float(holder_score),
            'risk_taking': float(risk_score),
            'security_focused': float(security_score),
            'community_builder': float(community_score),
            'independent_actor': float(independent_score),
            
            # 복합 유형
            'defi_trader_risk': float((defi_score * 0.4 + trader_score * 0.3 + risk_score * 0.3)),
            'nft_holder_community': float((nft_score * 0.4 + holder_score * 0.3 + community_score * 0.3)),
            'defi_holder_security': float((defi_score * 0.4 + holder_score * 0.3 + security_score * 0.3)),
            'nft_trader_risk': float((nft_score * 0.4 + trader_score * 0.3 + risk_score * 0.3)),
            'defi_holder_community': float((defi_score * 0.4 + holder_score * 0.3 + community_score * 0.3)),
            'nft_holder_independent': float((nft_score * 0.4 + holder_score * 0.3 + independent_score * 0.3)),
            'defi_trader_independent': float((defi_score * 0.4 + trader_score * 0.3 + independent_score * 0.3)),
            'nft_trader_community': float((nft_score * 0.4 + trader_score * 0.3 + community_score * 0.3))
        }
        
        return user_traits
    
    def get_primary_traits(self, user_traits: Dict[str, float], threshold: float = 0.6, max_traits: int = 3) -> List[str]:
        """
        주요 사용자 특성 추출
        
        Args:
            user_traits: 사용자 특성 점수
            threshold: 최소 점수 기준
            max_traits: 최대 특성 수
            
        Returns:
            주요 사용자 특성 목록
        """
        # 점수 기준으로 정렬
        sorted_traits = sorted(user_traits.items(), key=lambda x: x[1], reverse=True)
        
        # 임계값 이상인 특성 선택
        primary_traits = [trait for trait, score in sorted_traits if score >= threshold]
        
        # 최대 특성 수 제한
        return primary_traits[:max_traits] 
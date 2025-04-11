"""
클러스터 분석 및 사용자 특성 생성 모듈
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import json
import os
import logging
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

logger = logging.getLogger(__name__)

class ClusterFeatureAnalyzer:
    """클러스터별 특성 분석 및 사용자 특징 추출 클래스"""
    
    # 사용자 특성 정의
    USER_TRAITS = {
        'nft_enthusiast': '높은 NFT 활동과 보유량을 가진 사용자',
        'defi_trader': 'DeFi 프로토콜과 상호작용이 많은 사용자',
        'heavy_trader': '높은 거래량과 빈번한 트랜잭션을 보이는 사용자',
        'light_user': '적은 활동량과 단순한 상호작용을 보이는 사용자',
        'hodler': '장기 보유 성향을 보이는 사용자',
        'token_collector': '다양한 토큰을 보유하는 사용자',
        'active_social': '다양한 상대방과 상호작용하는 사용자',
        'gas_optimizer': '가스비를 효율적으로 사용하는 사용자',
        'gas_spender': '가스비를 많이 사용하는 사용자',
        'new_user': '최근에 활동을 시작한 사용자'
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
        
        # NFT 애호가
        nft_score = 0.0
        if 'nft_activity' in specialized_traits:
            nft_score += specialized_traits['nft_activity'].get('percentile', 0) * 0.6
        nft_score += min(metrics.get('nft_ratio', 0) * 10, 0.4)  # NFT 비율
        user_traits['nft_enthusiast'] = float(nft_score)
        
        # DeFi 트레이더
        defi_score = 0.0
        if 'defi_activity' in specialized_traits:
            defi_score += specialized_traits['defi_activity'].get('percentile', 0) * 0.7
        defi_score += min(metrics.get('defi_interaction_count', 0) / 10, 0.3)  # DeFi 상호작용
        user_traits['defi_trader'] = float(defi_score)
        
        # 헤비 트레이더
        heavy_score = 0.0
        if 'transaction_volume' in specialized_traits:
            heavy_score += specialized_traits['transaction_volume'].get('percentile', 0) * 0.5
        if 'transaction_value' in specialized_traits:
            heavy_score += specialized_traits['transaction_value'].get('percentile', 0) * 0.3
        heavy_score += min(metrics.get('txn_per_day', 0) / 20, 0.2)  # 일일 트랜잭션
        user_traits['heavy_trader'] = float(heavy_score)
        
        # 라이트 유저 (헤비 트레이더와 반대)
        light_score = 1.0 - heavy_score
        user_traits['light_user'] = float(light_score)
        
        # 호들러 (장기 보유자)
        hodler_score = 0.0
        if 'holding_behavior' in specialized_traits:
            hodler_score += specialized_traits['holding_behavior'].get('percentile', 0) * 0.6
        # 트랜잭션 빈도가 낮을수록 hodler 점수 높음
        hodler_score += max(0, 0.4 * (1 - min(metrics.get('txn_per_day', 0) / 10, 1)))
        user_traits['hodler'] = float(hodler_score)
        
        # 토큰 컬렉터
        collector_score = 0.0
        if 'diversity' in specialized_traits:
            collector_score += specialized_traits['diversity'].get('percentile', 0) * 0.5
        collector_score += min(metrics.get('token_count', 0) / 20, 0.5)  # 토큰 수
        user_traits['token_collector'] = float(collector_score)
        
        # 활발한 사회적 상호작용
        social_score = 0.0
        social_score += min(metrics.get('unique_counterparties', 0) / 50, 0.7)  # 상대방 수
        social_score += min(metrics.get('unique_contracts', 0) / 30, 0.3)  # 컨트랙트 수
        user_traits['active_social'] = float(social_score)
        
        # 가스 최적화 사용자 (가스 효율성이 높을수록)
        gas_optimizer_score = 0.0
        if 'efficiency' in specialized_traits:
            gas_score = 1 - specialized_traits['efficiency'].get('percentile', 0.5)  # 낮을수록 효율적
            gas_optimizer_score += gas_score * 0.8
        gas_optimizer_score += 0.2 * (1 - min(metrics.get('gas_to_value_ratio', 0) / 0.1, 1))
        user_traits['gas_optimizer'] = float(gas_optimizer_score)
        
        # 가스 다량 사용자 (가스 비용이 높을수록)
        gas_spender_score = 1.0 - gas_optimizer_score
        user_traits['gas_spender'] = float(gas_spender_score)
        
        # 신규 사용자
        new_user_score = 0.0
        if 'recency' in specialized_traits:
            # 최근 활동이 가까울수록 점수 높음
            recency_percentile = 1 - specialized_traits['recency'].get('percentile', 0.5)
            new_user_score += recency_percentile * 0.7
        # 계정 나이가 짧을수록 신규 사용자
        days_factor = max(0, 1 - metrics.get('account_age_days', 365) / 365)
        new_user_score += days_factor * 0.3
        user_traits['new_user'] = float(new_user_score)
        
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
            'gas_to_value_ratio': features.get('gas_to_value_ratio', 0),
            'account_age_days': features.get('account_age_days', 0),
            'days_since_last_activity': features.get('days_since_last_activity', 0)
        }
        
        # 모든 특성에 대한 스코어 계산
        nft_score = min(1.0, features.get('nft_ratio', 0) * 5 + 
                        features.get('nft_count', 0) / 10)
        
        defi_score = min(1.0, features.get('defi_interaction_count', 0) / 20)
        
        heavy_score = min(1.0, 
                         features.get('txn_per_day', 0) / 10 * 0.5 + 
                         features.get('total_txn_count', 0) / 100 * 0.3 + 
                         min(features.get('avg_txn_value', 0) / 1000, 1.0) * 0.2)
        
        light_score = 1.0 - heavy_score
        
        hodler_score = min(1.0, 
                          features.get('account_age_days', 0) / 365 * 0.5 + 
                          (1 - min(features.get('txn_per_day', 0) / 5, 1)) * 0.5)
        
        collector_score = min(1.0, 
                             features.get('token_count', 0) / 20 * 0.6 + 
                             features.get('erc20_count', 0) / 15 * 0.4)
        
        social_score = min(1.0, 
                          features.get('unique_counterparties', 0) / 50 * 0.7 + 
                          features.get('unique_contracts', 0) / 30 * 0.3)
        
        gas_ratio = features.get('gas_to_value_ratio', 0)
        gas_optimizer_score = 1.0 - min(1.0, gas_ratio / 0.1)
        
        gas_spender_score = 1.0 - gas_optimizer_score
        
        new_user_score = min(1.0, 
                            (1 - min(features.get('account_age_days', 0) / 90, 1)) * 0.7 + 
                            (1 - min(features.get('days_since_last_activity', 0) / 30, 1)) * 0.3)
        
        # 결과 사용자 특성 점수
        user_traits = {
            'nft_enthusiast': float(nft_score),
            'defi_trader': float(defi_score),
            'heavy_trader': float(heavy_score),
            'light_user': float(light_score),
            'hodler': float(hodler_score),
            'token_collector': float(collector_score),
            'active_social': float(social_score),
            'gas_optimizer': float(gas_optimizer_score),
            'gas_spender': float(gas_spender_score),
            'new_user': float(new_user_score)
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
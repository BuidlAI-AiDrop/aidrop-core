"""
비지도 학습 클러스터링 알고리즘 모듈
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import pickle
import os
import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ClusteringModel:
    """유저 프로필 클러스터링 모델 클래스"""
    
    def __init__(self, output_dir: str = 'models'):
        """
        Args:
            output_dir: 모델 저장 디렉토리
        """
        self.logger = logger
        self.output_dir = output_dir
        self.models = {}
        self.features_for_clustering = None
        self.pca = None
        self.feature_importances = {}
        
        # 기본 저장 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def fit_models(self, features_df: pd.DataFrame, selected_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        여러 클러스터링 알고리즘 적용
        
        Args:
            features_df: 특성 DataFrame
            selected_features: 클러스터링에 사용할 특성 목록 (없으면 모든 숫자형 특성 사용)
            
        Returns:
            모델별 클러스터링 결과 딕셔너리
        """
        # 클러스터링에 사용할 특성 선택
        if selected_features:
            self.features_for_clustering = selected_features
        else:
            # 스케일링된 특성만 선택 (feature_scaled 접미사)
            self.features_for_clustering = [col for col in features_df.columns if col.endswith('_scaled')]
        
        self.logger.info(f"클러스터링에 {len(self.features_for_clustering)} 개 특성 사용")
        
        # 차원 축소 (PCA)
        self.pca = PCA(n_components=min(len(self.features_for_clustering), 30))
        pca_result = self.pca.fit_transform(features_df[self.features_for_clustering])
        
        # 주요 특성 중요도 계산
        for i, component in enumerate(self.pca.components_):
            if i >= 5:  # 상위 5개 성분만 기록
                break
            feature_importance = {}
            for j, feature in enumerate(self.features_for_clustering):
                feature_importance[feature] = abs(component[j])
            
            # 중요도별 정렬
            sorted_importance = {k: v for k, v in sorted(
                feature_importance.items(), key=lambda item: item[1], reverse=True)[:10]
            }
            
            self.feature_importances[f'pca_component_{i+1}'] = sorted_importance
        
        # 여러 클러스터링 알고리즘 적용
        clustering_results = {}
        
        # 1. K-Means 클러스터링
        kmeans_models = {}
        for n_clusters in [3, 5, 7, 10]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(features_df[self.features_for_clustering])
            
            kmeans_models[n_clusters] = {
                'model': kmeans,
                'labels': kmeans.labels_,
                'inertia': kmeans.inertia_
            }
        
        # 최적의 K 선택 (엘보우 메소드 대신 3개 클러스터 사용)
        optimal_k = 5  # 기본값
        self.models['kmeans'] = kmeans_models[optimal_k]['model']
        clustering_results['kmeans'] = {
            'labels': kmeans_models[optimal_k]['labels'],
            'num_clusters': optimal_k
        }
        
        # 2. DBSCAN 클러스터링
        dbscan = DBSCAN(eps=1.0, min_samples=5)
        dbscan_labels = dbscan.fit_predict(features_df[self.features_for_clustering])
        
        # DBSCAN에서 -1은 노이즈 포인트
        num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        self.models['dbscan'] = dbscan
        clustering_results['dbscan'] = {
            'labels': dbscan_labels,
            'num_clusters': num_clusters
        }
        
        # 3. 계층적 클러스터링
        agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
        agg_labels = agg_clustering.fit_predict(features_df[self.features_for_clustering])
        
        self.models['hierarchical'] = agg_clustering
        clustering_results['hierarchical'] = {
            'labels': agg_labels,
            'num_clusters': optimal_k
        }
        
        # 4. 가우시안 혼합 모델 (GMM)
        gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        gmm.fit(features_df[self.features_for_clustering])
        gmm_labels = gmm.predict(features_df[self.features_for_clustering])
        
        self.models['gmm'] = gmm
        clustering_results['gmm'] = {
            'labels': gmm_labels,
            'num_clusters': optimal_k
        }
        
        # 시각화용 차원 축소 (t-SNE)
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(pca_result)
        
        clustering_results['visualization'] = {
            'tsne': tsne_result,
            'pca': pca_result[:, :2]  # 처음 2개 주성분
        }
        
        self.logger.info(f"클러스터링 완료: K-Means({optimal_k}개), DBSCAN({num_clusters}개), 계층적({optimal_k}개), GMM({optimal_k}개)")
        return clustering_results
    
    def save_models(self, version: str = "v1"):
        """
        학습된 모델 저장
        
        Args:
            version: 모델 버전
        """
        version_dir = os.path.join(self.output_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(version_dir, f"{model_name}_model.pkl")
            
            with open(model_path, 'wb') as f:
                model_data = {
                    'model': model,
                    'features': self.features_for_clustering,
                    'feature_importances': self.feature_importances,
                    'pca': self.pca
                }
                pickle.dump(model_data, f)
            
            self.logger.info(f"모델 저장 완료: {model_path}")
    
    def load_model(self, model_name: str, version: str = "v1") -> bool:
        """
        저장된 모델 로드
        
        Args:
            model_name: 모델 이름 (kmeans, dbscan 등)
            version: 모델 버전
            
        Returns:
            로드 성공 여부
        """
        model_path = os.path.join(self.output_dir, version, f"{model_name}_model.pkl")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models[model_name] = model_data['model']
            self.features_for_clustering = model_data['features']
            self.feature_importances = model_data.get('feature_importances', {})
            self.pca = model_data.get('pca')
            
            self.logger.info(f"모델 로드 완료: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {str(e)}")
            return False
    
    def predict(self, features_df: pd.DataFrame, model_name: str = 'kmeans') -> np.ndarray:
        """
        새 데이터에 대한 클러스터 예측
        
        Args:
            features_df: 특성 DataFrame
            model_name: 사용할 모델 이름
            
        Returns:
            클러스터 레이블 배열
        """
        if model_name not in self.models:
            self.logger.error(f"모델 '{model_name}'이 로드되지 않았습니다.")
            return np.array([])
        
        # 클러스터링에 사용된 특성만 선택
        X = features_df[self.features_for_clustering]
        
        # 모델별 예측 방식
        if model_name == 'dbscan':
            labels = self.models[model_name].fit_predict(X)
        else:
            labels = self.models[model_name].predict(X)
        
        return labels
    
    def visualize_clusters(self, features_df: pd.DataFrame, labels: np.ndarray, title: str = 'Clusters') -> str:
        """
        클러스터 시각화
        
        Args:
            features_df: 특성 DataFrame
            labels: 클러스터 레이블
            title: 그래프 제목
            
        Returns:
            저장된 이미지 경로
        """
        # PCA로 차원 축소
        if self.pca is None:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(features_df[self.features_for_clustering])
        else:
            pca_result = self.pca.transform(features_df[self.features_for_clustering])[:, :2]
        
        # 시각화
        plt.figure(figsize=(12, 8))
        
        # 클러스터별 색상
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # DBSCAN 노이즈 포인트
                plt.scatter(
                    pca_result[labels == label, 0],
                    pca_result[labels == label, 1],
                    s=50, c='gray', alpha=0.5, marker='x', label='Noise'
                )
            else:
                plt.scatter(
                    pca_result[labels == label, 0],
                    pca_result[labels == label, 1],
                    s=50, c=[colors[i]], alpha=0.7, marker='o', label=f'Cluster {label}'
                )
        
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 이미지 저장
        os.makedirs('plots', exist_ok=True)
        filename = f"plots/{title.lower().replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()
        
        return filename 
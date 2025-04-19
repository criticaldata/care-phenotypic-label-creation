import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from care_phenotype_analyzer.phenotype_creator import CarePhenotypeCreator

def explore_clustering():
    """Explore the clustering functionality of CarePhenotypeCreator."""
    # Load synthetic data
    data_path = Path(__file__).parent / 'synthetic_healthcare_data.csv'
    if not data_path.exists():
        print("Please run data_generator.py first to create synthetic data")
        return
    
    data = pd.read_csv(data_path)
    
    # Identify clinical factors and care patterns
    clinical_cols = [col for col in data.columns if col.startswith('clinical_factor')]
    care_pattern_cols = [col for col in data.columns if col.startswith('care_pattern')]
    
    print(f"Loaded data with {len(data)} records")
    
    # Explore clustering with different numbers of clusters
    cluster_counts = [2, 3, 4, 5]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Clustering Results with Different Numbers of Clusters')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    for i, n_clusters in enumerate(cluster_counts):
        # Initialize CarePhenotypeCreator
        creator = CarePhenotypeCreator(
            data=data,
            clinical_factors=clinical_cols,
            n_clusters=n_clusters,
            random_state=42
        )
        
        # Preprocess data
        preprocessed_data = creator.preprocess_data()
        
        # Get phenotype labels
        labels = creator.create_phenotype_labels()
        
        print(f"\nClustering with {n_clusters} clusters:")
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        print(f"Cluster sizes: {cluster_counts.to_dict()}")
        
        # Visualize clusters using PCA for dimension reduction
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(preprocessed_data[care_pattern_cols])
        
        # Plot clusters
        scatter = axes[i].scatter(
            pca_result[:, 0], 
            pca_result[:, 1], 
            c=labels, 
            cmap='viridis', 
            alpha=0.6,
            s=50
        )
        
        # Add cluster centers if available
        try:
            centers = creator._kmeans.cluster_centers_
            centers_pca = pca.transform(centers)
            axes[i].scatter(
                centers_pca[:, 0],
                centers_pca[:, 1],
                c='red',
                marker='X',
                s=200,
                edgecolors='black'
            )
        except:
            # If centers not available, continue without them
            pass
        
        axes[i].set_title(f'{n_clusters} Clusters')
        axes[i].set_xlabel('PCA Component 1')
        axes[i].set_ylabel('PCA Component 2')
        
        # Add legend
        legend = axes[i].legend(*scatter.legend_elements(), 
                               title="Clusters")
        axes[i].add_artist(legend)
    
    plt.tight_layout()
    plt.savefig('clustering_results.png')
    print("\nSaved clustering visualization to clustering_results.png")
    
    # Explore the effect of random state
    random_states = [42, 100, 200, 300]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Effect of Random State on Clustering (3 Clusters)')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    for i, random_state in enumerate(random_states):
        # Initialize CarePhenotypeCreator
        creator = CarePhenotypeCreator(
            data=data,
            clinical_factors=clinical_cols,
            n_clusters=3,
            random_state=random_state
        )
        
        # Preprocess data
        preprocessed_data = creator.preprocess_data()
        
        # Get phenotype labels
        labels = creator.create_phenotype_labels()
        
        print(f"\nClustering with random_state={random_state}:")
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        print(f"Cluster sizes: {cluster_counts.to_dict()}")
        
        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(preprocessed_data[care_pattern_cols])
        
        # Plot clusters
        scatter = axes[i].scatter(
            pca_result[:, 0], 
            pca_result[:, 1], 
            c=labels, 
            cmap='viridis', 
            alpha=0.6,
            s=50
        )
        
        axes[i].set_title(f'Random State: {random_state}')
        axes[i].set_xlabel('PCA Component 1')
        axes[i].set_ylabel('PCA Component 2')
        
        # Add legend
        legend = axes[i].legend(*scatter.legend_elements(), 
                               title="Clusters")
        axes[i].add_artist(legend)
    
    plt.tight_layout()
    plt.savefig('random_state_effect.png')
    print("\nSaved random state visualization to random_state_effect.png")

if __name__ == "__main__":
    explore_clustering()
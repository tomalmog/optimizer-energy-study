#!/usr/bin/env python3
"""
Analysis script for optimizer energy efficiency results.

This script processes the experimental data and generates the figures
and statistical analyses reported in the paper.

Author: Tom Almog
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load experimental results."""
    df = pd.read_csv('../data/experimental_data/comprehensive_results.csv')
    return df

def statistical_analysis(df):
    """Perform statistical tests on the data."""
    results = []
    
    for dataset in df.dataset.unique():
        subset = df[df.dataset == dataset]
        
        # Test for accuracy differences
        acc_groups = [subset[subset.optimizer == opt]['final_accuracy'].values 
                     for opt in subset.optimizer.unique()]
        friedman_stat, p_val = stats.friedmanchisquare(*acc_groups)
        
        results.append({
            'dataset': dataset,
            'metric': 'accuracy', 
            'friedman_chi2': friedman_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        })
    
    return pd.DataFrame(results)

def create_efficiency_plots(df):
    """Generate efficiency comparison plots."""
    
    # Performance vs emissions plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, dataset in enumerate(['MNIST', 'CIFAR10', 'CIFAR100']):
        subset = df[df.dataset == dataset]
        grouped = subset.groupby('optimizer').agg({
            'final_accuracy': ['mean', 'std'],
            'emissions_kg': ['mean', 'std']
        })
        
        x = grouped[('emissions_kg', 'mean')]
        y = grouped[('final_accuracy', 'mean')]
        xerr = grouped[('emissions_kg', 'std')]
        yerr = grouped[('final_accuracy', 'std')]
        
        axes[i].errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', capsize=5)
        
        for opt in grouped.index:
            axes[i].annotate(opt, (x[opt], y[opt]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9)
        
        axes[i].set_xlabel('CO₂ Emissions (kg)')
        axes[i].set_ylabel('Final Accuracy')
        axes[i].set_title(f'{dataset}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/plots/performance_vs_emissions.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Load and analyze data
    df = load_data()
    
    print("Dataset overview:")
    print(f"Total experiments: {len(df)}")
    print(f"Datasets: {df.dataset.unique()}")
    print(f"Optimizers: {df.optimizer.unique()}")
    
    # Statistical analysis
    stats_results = statistical_analysis(df)
    print("\nStatistical significance:")
    print(stats_results)
    
    # Generate plots
    create_efficiency_plots(df)
    print("\nPlots saved to results/plots/")

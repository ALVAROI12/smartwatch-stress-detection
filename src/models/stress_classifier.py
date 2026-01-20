"""
Stress Classification Models
============================

This module contains Random Forest and XGBoost models for stress detection
using physiological signals from WESAD dataset.

Author: [Your Name]
Date: 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from scipy import stats
import warnings
import sys

warnings.filterwarnings('ignore')

class StressClassifier:
    """
    Main class for stress classification using physiological signals
    """
    
    def __init__(self, data_path="data/processed/wesad_features.json", results_path="results"):
        """
        Initialize the stress classifier
        
        Parameters:
        -----------
        data_path : str
            Path to processed features JSON file
        results_path : str
            Path to save results
        """
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        
        # Create results directory
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 Stress Classifier initialized")
        print(f"   Data path: {self.data_path}")
        print(f"   Results path: {self.results_path}")
    
    def load_data(self):
        """Load the extracted WESAD features"""
        
        print("\n📂 Loading WESAD Features")
        print("=" * 30)
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} windows")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Basic info
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📊 Subjects: {df['subject_id'].nunique()}")
        print(f"📊 Conditions: {df['condition'].value_counts().to_dict()}")
        print(f"📊 Class distribution:")
        print(df['label'].value_counts().rename({0: 'Baseline', 1: 'Stress'}))
        
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix and labels for ML"""
        
        print("\n🔧 Preparing Features and Labels")
        print("=" * 35)
        
        # Identify feature columns (exclude metadata)
        metadata_cols = ['subject_id', 'condition', 'label', 'window_start', 'window_end', 
                        'segment_idx', 'window_duration_sec', 'purity']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        print(f"📊 Feature columns: {len(feature_cols)}")
        print(f"📊 Features: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
        
        # Extract features and labels
        X = df[feature_cols].values
        y = df['label'].values
        subjects = df['subject_id'].values
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"📊 Labels shape: {y.shape}")
        print(f"📊 Unique subjects: {np.unique(subjects)}")
        
        # Check for missing values
        missing_values = np.isnan(X).sum()
        if missing_values > 0:
            print(f"⚠️  Found {missing_values} missing values - filling with feature means")
            # Simple imputation
            col_means = np.nanmean(X, axis=0)
            for i in range(X.shape[1]):
                X[np.isnan(X[:, i]), i] = col_means[i]
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"✅ Features prepared and scaled")
        
        return X_scaled, y, subjects, feature_cols, scaler
    
    def evaluate_random_forest(self, X, y, subjects, feature_cols):
        """Evaluate Random Forest with Leave-One-Subject-Out CV"""
        
        print("\n🌲 Random Forest Evaluation")
        print("=" * 35)
        
        # Create Random Forest classifier
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        
        # Leave-One-Subject-Out Cross-Validation
        logo = LeaveOneGroupOut()
        
        print("🔄 Performing Leave-One-Subject-Out Cross-Validation...")
        
        # Perform CV with multiple metrics
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = cross_validate(rf, X, y, groups=subjects, cv=logo, 
                                   scoring=scoring, return_train_score=False)
        
        # Print results
        print("\n📊 Random Forest LOSO CV Results:")
        for metric in scoring:
            scores = cv_results[f'test_{metric}']
            print(f"   {metric.upper():10s}: {scores.mean():.3f} ± {scores.std():.3f}")
        
        # Train on full dataset for feature importance
        rf.fit(X, y)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:25s}: {row['importance']:.3f}")
        
        return rf, cv_results, feature_importance
    
    def evaluate_xgboost(self, X, y, subjects, feature_cols):
        """Evaluate XGBoost with Leave-One-Subject-Out CV"""
        
        print("\n🚀 XGBoost Evaluation")
        print("=" * 25)
        
        # Create XGBoost classifier
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Calculate class weights for imbalanced dataset
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        sample_weights = np.array([class_weights[int(label)] for label in y])
        
        # Leave-One-Subject-Out Cross-Validation
        logo = LeaveOneGroupOut()
        
        print("🔄 Performing Leave-One-Subject-Out Cross-Validation...")
        
        # Manual CV loop for XGBoost with sample weights
        cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
        
        for train_idx, test_idx in logo.split(X, y, subjects):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            weights_train = sample_weights[train_idx]
            
            # Train XGBoost
            xgb_clf.fit(X_train, y_train, sample_weight=weights_train)
            
            # Predict
            y_pred = xgb_clf.predict(X_test)
            y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            cv_scores['accuracy'].append(accuracy_score(y_test, y_pred))
            cv_scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            cv_scores['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))
        
        # Print results
        print("\n📊 XGBoost LOSO CV Results:")
        for metric, scores in cv_scores.items():
            scores_array = np.array(scores)
            print(f"   {metric.upper():10s}: {scores_array.mean():.3f} ± {scores_array.std():.3f}")
        
        # Train on full dataset for feature importance
        xgb_clf.fit(X, y, sample_weight=sample_weights)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': xgb_clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:25s}: {row['importance']:.3f}")
        
        return xgb_clf, cv_scores, feature_importance
    
    def compare_models(self, rf_scores, xgb_scores):
        """Compare Random Forest and XGBoost performance"""
        
        print("\n⚖️  Model Comparison")
        print("=" * 25)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        comparison_df = pd.DataFrame({
            'Metric': metrics,
            'Random Forest': [np.mean(rf_scores[f'test_{m}']) for m in metrics],
            'XGBoost': [np.mean(xgb_scores[m]) for m in metrics],
            'RF_std': [np.std(rf_scores[f'test_{m}']) for m in metrics],
            'XGB_std': [np.std(xgb_scores[m]) for m in metrics]
        })
        
        # Calculate differences
        comparison_df['Difference'] = comparison_df['XGBoost'] - comparison_df['Random Forest']
        comparison_df['Better_Model'] = comparison_df['Difference'].apply(
            lambda x: 'XGBoost' if x > 0.01 else ('Random Forest' if x < -0.01 else 'Tie')
        )
        
        print("📊 Performance Comparison:")
        print(f"{'Metric':<12} {'RF':<8} {'XGB':<8} {'Diff':<8} {'Winner':<12}")
        print("-" * 55)
        
        for _, row in comparison_df.iterrows():
            print(f"{row['Metric']:<12} {row['Random Forest']:.3f}   {row['XGBoost']:.3f}   "
                  f"{row['Difference']:+.3f}   {row['Better_Model']:<12}")
        
        # Statistical significance testing
        print(f"\n📈 Statistical Significance (p < 0.05):")
        for metric in metrics:
            rf_metric = f'test_{metric}'
            rf_values = rf_scores[rf_metric]
            xgb_values = xgb_scores[metric]
            
            if len(rf_values) == len(xgb_values):
                statistic, p_value = stats.ttest_rel(xgb_values, rf_values)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"   {metric:<12}: p = {p_value:.4f} {significance}")
        
        return comparison_df
    
    def create_visualizations(self, df, rf_importance, xgb_importance, rf_scores, xgb_scores):
        """Create visualizations for the results"""
        
        print("\n📊 Creating Visualizations")
        print("=" * 30)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('WESAD Stress Detection - ML Results', fontsize=16, fontweight='bold')
        
        # 1. Class distribution
        ax1 = axes[0, 0]
        class_counts = df['condition'].value_counts()
        colors = ['lightblue', 'lightcoral']
        ax1.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=colors)
        ax1.set_title('Dataset Class Distribution')
        
        # 2. Feature importance comparison (top 10)
        ax2 = axes[0, 1]
        top_features = rf_importance.head(10)
        y_pos = np.arange(len(top_features))
        bars = ax2.barh(y_pos, top_features['importance'], alpha=0.7, color='forestgreen')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_features['feature'], fontsize=9)
        ax2.set_xlabel('Feature Importance')
        ax2.set_title('Top 10 Random Forest Features')
        ax2.invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        # 3. Model performance comparison
        ax3 = axes[1, 0]
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        rf_means = [np.mean(rf_scores[f'test_{m}']) for m in metrics]
        xgb_means = [np.mean(xgb_scores[m]) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, rf_means, width, label='Random Forest', 
                       alpha=0.8, color='skyblue')
        bars2 = ax3.bar(x + width/2, xgb_means, width, label='XGBoost', 
                       alpha=0.8, color='orange')
        
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Model Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics, rotation=45)
        ax3.legend()
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Subject distribution
        ax4 = axes[1, 1]
        subjects = sorted(df['subject_id'].unique())
        subject_counts = df['subject_id'].value_counts()[subjects]
        
        bars = ax4.bar(range(len(subjects)), subject_counts.values, 
                      alpha=0.7, color='mediumseagreen')
        ax4.set_xlabel('Subjects')
        ax4.set_ylabel('Number of Windows')
        ax4.set_title('Data Distribution by Subject')
        ax4.set_xticks(range(len(subjects)))
        ax4.set_xticklabels(subjects, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.results_path / 'wesad_ml_results.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved to: {viz_path}")
    
    def save_results(self, df, rf_model, xgb_model, rf_scores, xgb_scores, 
                    rf_importance, xgb_importance, comparison):
        """Save all results to files"""
        
        print("\n💾 Saving Results")
        print("=" * 20)
        
        # Prepare results dictionary
        results = {
            'dataset_info': {
                'total_windows': len(df),
                'stress_windows': len(df[df['label'] == 1]),
                'baseline_windows': len(df[df['label'] == 0]),
                'subjects': len(df['subject_id'].unique()),
                'features': len([col for col in df.columns if col not in 
                              ['subject_id', 'condition', 'label', 'window_start', 
                               'window_end', 'segment_idx', 'window_duration_sec', 'purity']])
            },
            'random_forest': {
                'cv_scores': {k: v.tolist() for k, v in rf_scores.items()},
                'feature_importance': rf_importance.to_dict('records')
            },
            'xgboost': {
                'cv_scores': xgb_scores,
                'feature_importance': xgb_importance.to_dict('records')
            },
            'comparison': comparison.to_dict('records')
        }
        
        # Save to JSON
        results_path = self.results_path / 'wesad_ml_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save feature importance to CSV
        rf_importance.to_csv(self.results_path / 'rf_feature_importance.csv', index=False)
        xgb_importance.to_csv(self.results_path / 'xgb_feature_importance.csv', index=False)
        
        print(f"✅ Results saved to: {results_path}")
        print(f"✅ Feature importance saved to CSV files")
        
        return results
    
    def run_complete_analysis(self):
        """Run the complete stress detection analysis"""
        
        print("🎯 WESAD STRESS DETECTION - COMPLETE ML ANALYSIS")
        print("=" * 60)
        
        # 1. Load data
        df = self.load_data()
        
        # 2. Prepare features
        X, y, subjects, feature_cols, scaler = self.prepare_features(df)
        
        # 3. Evaluate Random Forest
        rf_model, rf_scores, rf_importance = self.evaluate_random_forest(X, y, subjects, feature_cols)
        
        # 4. Evaluate XGBoost
        xgb_model, xgb_scores, xgb_importance = self.evaluate_xgboost(X, y, subjects, feature_cols)
        
        # 5. Compare models
        comparison = self.compare_models(rf_scores, xgb_scores)
        
        # 6. Create visualizations
        self.create_visualizations(df, rf_importance, xgb_importance, rf_scores, xgb_scores)
        
        # 7. Save results
        results = self.save_results(df, rf_model, xgb_model, rf_scores, xgb_scores, 
                                  rf_importance, xgb_importance, comparison)
        
        # 8. Final summary
        print(f"\n🎯 FINAL SUMMARY")
        print("=" * 20)
        print(f"✅ Dataset: {len(df)} windows from {len(df['subject_id'].unique())} subjects")
        print(f"✅ Random Forest: {np.mean(rf_scores['test_accuracy']):.3f} ± {np.std(rf_scores['test_accuracy']):.3f} accuracy")
        print(f"✅ XGBoost: {np.mean(xgb_scores['accuracy']):.3f} ± {np.std(xgb_scores['accuracy']):.3f} accuracy")
        
        # Determine best model
        rf_acc = np.mean(rf_scores['test_accuracy'])
        xgb_acc = np.mean(xgb_scores['accuracy'])
        best_model = "XGBoost" if xgb_acc > rf_acc else "Random Forest"
        best_acc = max(rf_acc, xgb_acc)
        
        print(f"\n🏆 BEST MODEL: {best_model} ({best_acc:.3f} accuracy)")
        print(f"🎉 Stress detection pipeline successfully completed!")
        
        return results

# Main execution function
def main():
    """Main function to run stress classification"""
    
    # Initialize classifier
    classifier = StressClassifier(
        data_path="data/processed/wesad_features.json",
        results_path="results"
    )
    
    # Run complete analysis
    results = classifier.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()
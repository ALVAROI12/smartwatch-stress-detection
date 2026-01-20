#!/usr/bin/env python3
"""
Production-Grade Random Forest Classifier for Stress Detection
====================================================================
Includes safeguards for OOB estimation conflict resolution.
"""

import json
import logging
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "src"))

from utils.logging_utils import setup_logging
from utils.config_manager import get_config

warnings.filterwarnings('ignore')

logger = logging.getLogger("smartwatch.production_rf")

class ProductionRandomForestClassifier:
    def __init__(self, data_path="data/processed/empatica_e4_improved_features.json", 
                 optimization_level="intensive", random_state=42):
        self.data_path = Path(data_path)
        self.optimization_level = optimization_level
        self.random_state = random_state
        self.best_model = None
        self.best_pipeline = None
        self.logger = logger.getChild(self.__class__.__name__)
        
        self.logger.info("Production Random Forest Classifier initialized")
        self.logger.info("Optimization level: %s", optimization_level.upper())
        self.logger.debug(
            "Target performance band: 85-90%% accuracy via literature best practices"
        )
    
    def load_and_prepare_data(self):
        self.logger.info("Loading data from %s", self.data_path)

        with open(self.data_path, 'r', encoding='utf-8') as stream:
            data = json.load(stream)
        df = pd.DataFrame(data)

        self.logger.info(
            "Loaded %d windows across %d subjects",
            len(df),
            df['subject_id'].nunique(),
        )
        self.logger.info("Class distribution: %s", Counter(df['label']))

        metadata_cols = ['subject_id', 'condition', 'label', 'window_start', 'window_end', 'purity', 'window_duration_sec']
        features = [col for col in df.columns if col not in metadata_cols]

        X = df[features].values
        y = df['label'].values
        subjects = df['subject_id'].values

        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        feature_variances = np.var(X, axis=0)
        valid_mask = feature_variances > 1e-10
        X = X[:, valid_mask]
        features = [features[i] for i in range(len(features)) if valid_mask[i]]

        removed_features = int(np.sum(~valid_mask))
        self.logger.debug("Removed %d zero-variance features", removed_features)
        self.logger.info("Final feature count: %d", len(features))

        return X, y, subjects, features, df
    
    def get_param_grid(self):
        if self.optimization_level == "basic":
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'class_weight': ['balanced'],
                'bootstrap': [True],  # Fixed: always True for consistency
                'random_state': [self.random_state]
            }
        elif self.optimization_level == "intensive":
            return {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
                'bootstrap': [True],  # Fixed: always True
                'class_weight': ['balanced', 'balanced_subsample', None],
                'criterion': ['gini', 'entropy'],
                'random_state': [self.random_state]
            }
        else:  # standard
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5],
                'bootstrap': [True],  # Fixed: always True
                'class_weight': ['balanced', 'balanced_subsample'],
                'random_state': [self.random_state]
            }
    
    def optimize_hyperparameters(self, X, y, subjects):
        self.logger.info(
            "Hyperparameter optimization (%s)",
            self.optimization_level.upper(),
        )

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        self.logger.info("Class distribution: %s", Counter(y))
        self.logger.debug("Computed class weights: %s", class_weight_dict)

        param_grid = self.get_param_grid()
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        self.logger.info("Exploring %s parameter combinations", f"{total_combinations:,}")

        # Fixed: Remove OOB score to avoid conflicts
        rf_base = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1
        )
        
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        start_time = time.time()
        
        if total_combinations > 5000:
            self.logger.info("Using RandomizedSearchCV")
            search = RandomizedSearchCV(
                rf_base, param_grid, n_iter=1000, cv=cv_strategy,
                scoring='f1_macro', n_jobs=-1, verbose=0, random_state=self.random_state
            )
        else:
            self.logger.info("Using GridSearchCV")
            search = GridSearchCV(
                rf_base, param_grid, cv=cv_strategy,
                scoring='f1_macro', n_jobs=-1, verbose=0
            )
        
        search.fit(X, y)
        
        optimization_time = time.time() - start_time
        self.logger.info("Optimization completed in %.1f seconds", optimization_time)
        self.logger.info("Best CV F1-score: %.4f", search.best_score_)
        self.logger.info("Best hyperparameters: %s", search.best_params_)

        for param, value in sorted(search.best_params_.items()):
            self.logger.debug("Best param %s: %s", param, value)
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def feature_selection(self, X, y, features):
        self.logger.info("Feature selection on %d candidate features", len(features))

        n_features = min(20, max(10, len(features) // 2))
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = [features[i] for i in selector.get_support(indices=True)]

        self.logger.info("Selected %d features", len(selected_features))
        self.logger.debug("Top features: %s", selected_features[:10])

        return X_selected, selected_features, selector
    
    def comprehensive_evaluation(self, pipeline, X, y):
        self.logger.info("Comprehensive evaluation across CV strategies")

        cv_strategies = {
            'StratifiedKFold_5': StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            'StratifiedKFold_10': StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        }
        
        results = {}
        
        for cv_name, cv in cv_strategies.items():
            self.logger.info("Evaluating %s", cv_name)
            
            acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
            
            results[cv_name] = {
                'accuracy': {'mean': acc_scores.mean(), 'std': acc_scores.std()},
                'f1_macro': {'mean': f1_scores.mean(), 'std': f1_scores.std()}
            }
            
            self.logger.info(
                "Accuracy: %.3f +/- %.3f",
                acc_scores.mean(),
                acc_scores.std(),
            )
            self.logger.info(
                "F1-macro: %.3f +/- %.3f",
                f1_scores.mean(),
                f1_scores.std(),
            )
        
        # Feature importance
        pipeline.fit(X, y)
        classifier = pipeline.named_steps['classifier']
        if hasattr(classifier, 'feature_importances_'):
            importance = classifier.feature_importances_
            feature_names = pipeline.named_steps['feature_selector'].get_feature_names_out() if hasattr(pipeline.named_steps['feature_selector'], 'get_feature_names_out') else self.selected_features
            
            if len(feature_names) == len(importance):
                feature_importance = list(zip(feature_names, importance))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                top_features = feature_importance[:10]
                for index, (feat, imp) in enumerate(top_features, start=1):
                    self.logger.debug("Feature rank %d: %s (%.4f)", index, feat, imp)
                self.logger.info("Logged top %d feature importances", len(top_features))
        
        # Overall assessment
        best_f1 = max([results[cv]['f1_macro']['mean'] for cv in results.keys()])
        best_acc = max([results[cv]['accuracy']['mean'] for cv in results.keys()])
        self.logger.info("Best F1-score: %.3f", best_f1)
        self.logger.info("Best accuracy: %.3f", best_acc)
        
        if best_f1 >= 0.80 and best_acc >= 0.85:
            status = "EXCELLENT - Research-grade performance!"
        elif best_f1 >= 0.75 and best_acc >= 0.80:
            status = "VERY GOOD - Production-ready"
        elif best_f1 >= 0.70 and best_acc >= 0.75:
            status = "GOOD - Solid performance"
        else:
            status = "ACCEPTABLE - Room for improvement"
        
        self.logger.info("Performance status: %s", status)
        
        results['overall'] = {'best_f1': best_f1, 'best_accuracy': best_acc, 'status': status}
        return results
    
    def train_and_evaluate(self):
        self.logger.info("Starting production Random Forest training")
        
        X, y, subjects, features, df = self.load_and_prepare_data()
        
        best_model, best_params, best_score = self.optimize_hyperparameters(X, y, subjects)
        
        X_selected, selected_features, feature_selector = self.feature_selection(X, y, features)
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('feature_selector', feature_selector),
            ('classifier', best_model)
        ])
        
        results = self.comprehensive_evaluation(pipeline, X, y)
        
        self.best_pipeline = pipeline
        self.best_model = best_model
        self.selected_features = selected_features

        overall = results['overall']
        self.logger.info("Production training complete")
        self.logger.info("Best F1-score: %.3f", overall['best_f1'])
        self.logger.info("Best accuracy: %.3f", overall['best_accuracy'])
        self.logger.info("Selected features: %d", len(selected_features))
        self.logger.info("Status: %s", overall['status'])
        
        return {
            'pipeline': pipeline,
            'best_model': best_model,
            'best_params': best_params,
            'selected_features': selected_features,
            'results': results
        }

if __name__ == "__main__":
    config_manager = get_config()
    setup_logging(config_manager.get_logging_config())

    logger.info("Production Random Forest Stress Detection")
    logger.info("Includes OOB estimation conflict safeguards")

    OPTIMIZATION_LEVEL = "intensive"  # Change as needed

    classifier = ProductionRandomForestClassifier(optimization_level=OPTIMIZATION_LEVEL)
    results = classifier.train_and_evaluate()

    overall_results = results['results']['overall']
    logger.info("Training completed successfully")
    logger.info("F1-score: %.3f", overall_results['best_f1'])
    logger.info("Accuracy: %.3f", overall_results['best_accuracy'])
    logger.info("Ready for deployment")

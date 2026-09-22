
import joblib
import numpy as np

class StressDetector:
    def __init__(self, package_path):
        pkg = joblib.load(package_path)
        self.model = pkg['model']
        self.scaler = pkg['scaler']
        self.le = pkg['label_encoder']
        self.feature_names = pkg['feature_names']
        self.classes = pkg['classes']
    
    def extract_features(self, bvp, eda, temp, acc_x, acc_y, acc_z):
        """Extract features from raw signals"""
        features = {}
        
        # HR features (from BVP)
        features['hr_mean'] = np.mean(bvp)
        features['hr_std'] = np.std(bvp)
        features['hr_min'] = np.min(bvp)
        features['hr_max'] = np.max(bvp)
        
        # EDA features
        features['eda_mean'] = np.mean(eda)
        features['eda_std'] = np.std(eda)
        features['eda_min'] = np.min(eda)
        features['eda_max'] = np.max(eda)
        features['eda_range'] = np.max(eda) - np.min(eda)
        
        # Temperature features
        features['temp_mean'] = np.mean(temp)
        features['temp_std'] = np.std(temp)
        features['temp_min'] = np.min(temp)
        features['temp_max'] = np.max(temp)
        features['temp_range'] = np.max(temp) - np.min(temp)
        
        # ACC features
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        features['acc_x_mean'] = np.mean(acc_x)
        features['acc_y_mean'] = np.mean(acc_y)
        features['acc_z_mean'] = np.mean(acc_z)
        features['acc_mag_mean'] = np.mean(acc_mag)
        features['acc_mag_std'] = np.std(acc_mag)
        features['acc_sma'] = np.sum(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z)) / len(acc_x)
        
        return features
    
    def predict(self, features_dict):
        """Predict stress class from features"""
        X = np.array([[features_dict.get(f, 0) for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        return self.classes[pred], dict(zip(self.classes, proba))

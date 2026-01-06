"""
Model prediction utilities.
"""
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FEATURE_COLUMNS, MODELS_DIR


def load_model(filename: str = "codeforensics_model.pkl"):
    """Load trained model from disk."""
    path = MODELS_DIR / filename
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_risk(features: Dict[str, Any], model=None) -> Dict[str, Any]:
    """
    Predict bug risk for a commit given its features.
    
    Args:
        features: Dict with keys matching FEATURE_COLUMNS
        model: Pre-loaded model (loads from disk if None)
        
    Returns:
        Dict with risk_score and risk_level
    """
    if model is None:
        model = load_model()
    
    # Create feature vector
    feature_vector = pd.DataFrame([{col: features.get(col, 0) for col in FEATURE_COLUMNS}])
    
    # Predict
    risk_score = model.predict_proba(feature_vector)[0][1]
    
    # Determine risk level
    if risk_score > 0.7:
        risk_level = "HIGH"
    elif risk_score > 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "features_used": features
    }


def batch_predict(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Predict risk for multiple commits.
    
    Args:
        df: DataFrame with feature columns
        model: Pre-loaded model
        
    Returns:
        DataFrame with added risk_score and risk_level columns
    """
    if model is None:
        model = load_model()
    
    df = df.copy()
    
    # Get predictions
    X = df[FEATURE_COLUMNS].fillna(0)
    df['risk_score'] = model.predict_proba(X)[:, 1]
    df['risk_level'] = df['risk_score'].apply(
        lambda x: "HIGH" if x > 0.7 else "MEDIUM" if x > 0.4 else "LOW"
    )
    
    return df


if __name__ == "__main__":
    # Test prediction
    test_features = {
        'hour_of_day': 23,  # Late night commit
        'day_of_week': 4,   # Friday
        'files_changed': 15,
        'lines_added': 500,
        'lines_deleted': 100,
        'net_lines': 400,
        'complexity_delta': 10,
        'avg_file_churn': 50
    }
    
    result = predict_risk(test_features)
    print(f"Risk Score: {result['risk_score']:.1%}")
    print(f"Risk Level: {result['risk_level']}")

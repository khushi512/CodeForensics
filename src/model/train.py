"""
LightGBM model training for bug-inducing commit prediction.
"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FEATURE_COLUMNS, MODELS_DIR, DATASETS_DIR


def load_dataset(filename: str = "training_dataset.csv") -> pd.DataFrame:
    """Load the training dataset."""
    path = DATASETS_DIR / filename
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} samples")
    return df


def prepare_data(df: pd.DataFrame, test_size: float = None):
    """
    Prepare features and labels for training.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    from config import TEST_SIZE, RANDOM_STATE, logger
    
    if test_size is None:
        test_size = TEST_SIZE
    
    # Filter to only rows with all features
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ['is_bug_inducing'])
    
    logger.info(f"Cleaned dataset: {len(df_clean)} samples (dropped {len(df) - len(df_clean)} with missing values)")
    
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean['is_bug_inducing'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Bug-inducing rate: {y.mean():.1%}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test):
    """
    Train LightGBM classifier using config parameters.
    
    Returns:
        Trained model
    """
    from config import MODEL_PARAMS, logger
    
    model = lgb.LGBMClassifier(**MODEL_PARAMS)
    
    logger.info("\nTraining LightGBM model...")
    logger.info(f"Parameters: {MODEL_PARAMS}")
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    logger.info(f"Best iteration: {model.best_iteration_}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {auc:.3f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    for _, row in importance.iterrows():
        bar = "█" * int(row['importance'] / importance['importance'].max() * 20)
        print(f"  {row['feature']:20} {bar} ({row['importance']:.0f})")
    
    return auc


def save_model(model, filename: str = "codeforensics_model.pkl"):
    """Save trained model to disk."""
    path = MODELS_DIR / filename
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {path}")
    return path


def run_training_pipeline(dataset_file: str = "training_dataset.csv"):
    """Run the full training pipeline."""
    # Load data
    df = load_dataset(dataset_file)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate
    auc = evaluate_model(model, X_test, y_test)
    
    # Save
    save_model(model)
    
    return model, auc


if __name__ == "__main__":
    model, auc = run_training_pipeline()

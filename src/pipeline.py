"""
Main data processing pipeline with resume capability and validation.
Orchestrates cloning, extraction, features, and labeling.
"""
import pandas as pd
from pathlib import Path
import traceback

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (REPOS, DATASETS_DIR, CLONE_DEPTH, MAX_COMMITS, 
                    FEATURE_COLUMNS, logger)
from src.collection.clone import clone_repo_shallow, cleanup_repo
from src.collection.extract import extract_commit_data
from src.collection.features import extract_all_features
from src.labeling.blame import create_labeled_dataset


def print_repo_summary(df: pd.DataFrame, repo_url: str):
    """Print detailed summary for a repo"""
    logger.info(f"\n📊 {repo_url} Summary:")
    logger.info(f"  Total commits: {len(df)}")
    logger.info(f"  Bug fixes: {df['is_bug_fix'].sum()}")
    logger.info(f"  Bug-inducing: {df['is_bug_inducing'].sum()}")
    logger.info(f"  Bug rate: {df['is_bug_inducing'].mean():.1%}")
    logger.info(f"  Avg complexity delta: {df['complexity_delta'].mean():.1f}")
    logger.info(f"  Avg file churn: {df['avg_file_churn'].mean():.1f}")


def validate_dataset(df: pd.DataFrame) -> bool:
    """Validate dataset quality"""
    issues = []
    
    # Check for nulls in critical columns
    critical_cols = ['sha', 'is_bug_inducing']
    if 'repo' in df.columns:
        critical_cols.append('repo')
        
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"{col} has {null_count} null values")
    
    # Check class balance
    bug_rate = df['is_bug_inducing'].mean()
    if bug_rate < 0.01:
        issues.append(f"Very low bug-inducing rate: {bug_rate:.1%} - may indicate labeling issues")
    elif bug_rate > 0.5:
        issues.append(f"Very high bug-inducing rate: {bug_rate:.1%} - check labeling logic")
    
    # Check feature distributions
    for feat in FEATURE_COLUMNS:
        if feat in df.columns and df[feat].std() == 0:
            issues.append(f"No variance in {feat}")
    
    if issues:
        logger.warning(f"\n⚠️  Dataset Validation Issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info(f"\n✓ Dataset validation passed")
    return True


def process_single_repo(repo_url: str, delete_after: bool = True) -> pd.DataFrame:
    """
    Process a single repository through the full pipeline.
    
    Args:
        repo_url: GitHub repo in format 'owner/repo'
        delete_after: Whether to delete repo after processing
        
    Returns:
        Labeled dataset for this repository
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {repo_url}")
    logger.info('='*60)
    
    # Clone
    repo_path = clone_repo_shallow(repo_url, depth=CLONE_DEPTH)
    
    try:
        # Extract commits
        df_commits = extract_commit_data(repo_path, max_commits=MAX_COMMITS)
        
        # Extract features
        df_features = extract_all_features(repo_path, df_commits, max_commits=MAX_COMMITS)
        
        # Create labeled dataset
        df_labeled = create_labeled_dataset(repo_path, df_commits, df_features)
        
        # Add repo column for tracking
        df_labeled['repo'] = repo_url
        
        return df_labeled
        
    finally:
        if delete_after:
            cleanup_repo(repo_path)


def run_full_pipeline(repos: list = None, output_filename: str = "training_dataset.csv") -> pd.DataFrame:
    """
    Run the full pipeline on all configured repositories with resume capability.
    
    Args:
        repos: List of repos to process (uses config if None)
        output_filename: Name of output CSV file
        
    Returns:
        Combined dataset from all repositories
    """
    repos = repos or REPOS
    all_data = []
    
    logger.info(f"\nCodeForensics Data Pipeline")
    logger.info(f"{'='*60}")
    logger.info(f"Target repos: {len(repos)}")
    logger.info(f"Clone depth: {CLONE_DEPTH}, Max commits: {MAX_COMMITS}")
    
    # Check for existing checkpoint
    checkpoint_path = DATASETS_DIR / "checkpoint.csv"
    processed_repos = []
    
    if checkpoint_path.exists():
        logger.info(f"📂 Found checkpoint, loading...")
        try:
            existing_df = pd.read_csv(checkpoint_path)
            all_data.append(existing_df)
            if 'repo' in existing_df.columns:
                processed_repos = existing_df['repo'].unique().tolist()
                repos = [r for r in repos if r not in processed_repos]
                logger.info(f"📍 Resuming from {len(processed_repos)} already processed repos")
                logger.info(f"   Remaining: {len(repos)} repos")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
    
    # Process each repo
    for i, repo_url in enumerate(repos, 1):
        logger.info(f"\n[{i}/{len(repos)}] Processing {repo_url}")
        try:
            df = process_single_repo(repo_url, delete_after=True)
            
            # Show summary
            print_repo_summary(df, repo_url)
            
            all_data.append(df)
            
            # Save checkpoint after each repo
            checkpoint_df = pd.concat(all_data, ignore_index=True)
            checkpoint_df.to_csv(checkpoint_path, index=False)
            logger.info(f"💾 Checkpoint saved ({len(checkpoint_df)} total commits)")
            
        except Exception as e:
            logger.error(f"✗ {repo_url}: Failed - {e}")
            traceback.print_exc()
            continue
    
    if not all_data:
        logger.error("No data collected!")
        return pd.DataFrame()
    
    # Combine all datasets
    df_final = pd.concat(all_data, ignore_index=True)
    
    # Validate dataset
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATING DATASET")
    logger.info(f"{'='*60}")
    is_valid = validate_dataset(df_final)
    
    # Save final dataset
    output_path = DATASETS_DIR / output_filename
    df_final.to_csv(output_path, index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total commits: {len(df_final)}")
    logger.info(f"Repositories: {df_final['repo'].nunique()}")
    logger.info(f"Bug-inducing commits: {df_final['is_bug_inducing'].sum()}")
    logger.info(f"Bug-inducing rate: {df_final['is_bug_inducing'].mean():.1%}")
    logger.info(f"Saved to: {output_path}")
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("🗑️  Checkpoint cleaned up")
    
    if not is_valid:
        logger.warning("⚠️  Proceeding with validation warnings")
    
    return df_final


def show_dataset_stats(filename: str = "training_dataset.csv"):
    """Quick stats on existing dataset"""
    path = DATASETS_DIR / filename
    if not path.exists():
        logger.error(f"Dataset not found: {path}")
        return
    
    df = pd.read_csv(path)
    
    logger.info(f"\n📊 Dataset Statistics: {filename}")
    logger.info(f"{'='*60}")
    logger.info(f"Total commits: {len(df)}")
    
    if 'repo' in df.columns:
        logger.info(f"Repositories: {df['repo'].nunique()}")
        logger.info(f"Repos: {', '.join(df['repo'].unique())}")
    
    if 'timestamp' in df.columns:
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    logger.info(f"\n🐛 Labels:")
    logger.info(f"  Bug-inducing: {df['is_bug_inducing'].sum()} ({df['is_bug_inducing'].mean():.1%})")
    
    if 'is_bug_fix' in df.columns:
        logger.info(f"  Bug fixes: {df['is_bug_fix'].sum()}")
    
    logger.info(f"\n📈 Features:")
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            logger.info(f"  {col:20} mean={df[col].mean():8.2f}  std={df[col].std():8.2f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        # Show stats for existing dataset
        show_dataset_stats()
    else:
        # Run pipeline - start with just 1 repo for testing
        test_repos = ["pallets/click"]
        df = run_full_pipeline(repos=test_repos)
        
        if len(df) > 0:
            logger.info(f"\n📋 Sample data:")
            print(df[['sha', 'repo', 'is_bug_inducing', 'complexity_delta']].head())

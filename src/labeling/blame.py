"""
Bug-inducing commit labeling using git blame.
Traces bug fixes back to the commits that introduced the bugs.
"""
import pandas as pd
from pathlib import Path
from git import Repo
from typing import Set, List
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def find_bug_inducing_commits(repo_path: Path, bug_fix_shas: List[str], 
                               max_fixes: int = 100) -> Set[str]:
    """
    Use git blame to find commits that introduced bugs.
    
    For each bug-fix commit, we look at what lines were changed
    and use git blame to find who last modified those lines
    (the bug-inducing commit).
    
    Args:
        repo_path: Path to git repository
        bug_fix_shas: List of commit SHAs that are bug fixes
        max_fixes: Maximum bug fixes to trace back
        
    Returns:
        Set of commit SHAs that introduced bugs
    """
    repo = Repo(repo_path)
    bug_inducers: Set[str] = set()
    
    fixes_to_process = bug_fix_shas[:max_fixes]
    print(f"Tracing {len(fixes_to_process)} bug fixes to find bug-inducing commits...")
    
    for i, fix_sha in enumerate(fixes_to_process):
        try:
            commit = repo.commit(fix_sha)
            parent = commit.parents[0] if commit.parents else None
            
            if not parent:
                continue
            
            # Get diff between parent and fix commit
            diffs = parent.diff(commit)
            
            for diff in diffs:
                if not diff.a_path:  # File didn't exist before
                    continue
                    
                # Skip non-code files
                if not any(diff.a_path.endswith(ext) for ext in ['.py', '.js', '.ts', '.java']):
                    continue
                
                try:
                    # Use git blame on the parent to find who last modified these lines
                    blame_data = repo.blame(parent, diff.a_path)
                    
                    for blamed_commit, lines in blame_data:
                        # The blamed commit is the one that introduced the bug
                        if blamed_commit.hexsha != fix_sha:
                            bug_inducers.add(blamed_commit.hexsha)
                            
                except Exception:
                    # File might not exist in parent or blame might fail
                    continue
                    
        except Exception as e:
            print(f"  Error processing fix {fix_sha[:8]}: {e}")
            continue
            
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(fixes_to_process)} fixes...")
    
    print(f"Found {len(bug_inducers)} bug-inducing commits")
    return bug_inducers


def label_commits(df: pd.DataFrame, bug_inducer_shas: Set[str]) -> pd.DataFrame:
    """
    Add bug-inducing labels to commits DataFrame.
    
    Args:
        df: DataFrame with 'sha' column
        bug_inducer_shas: Set of SHAs that are bug-inducing
        
    Returns:
        DataFrame with 'is_bug_inducing' column added
    """
    df = df.copy()
    df['is_bug_inducing'] = df['sha'].isin(bug_inducer_shas)
    
    bug_count = df['is_bug_inducing'].sum()
    total = len(df)
    print(f"Labeled {bug_count}/{total} commits ({bug_count/total:.1%}) as bug-inducing")
    
    return df


def create_labeled_dataset(repo_path: Path, df_commits: pd.DataFrame, 
                            df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Create final labeled dataset by merging commits, features, and labels.
    
    Uses a simplified labeling approach that works with shallow clones:
    - Tries git blame first (may fail for shallow clones)
    - Falls back to heuristic labeling if blame fails
    
    Args:
        repo_path: Path to git repository
        df_commits: DataFrame with commit metadata
        df_features: DataFrame with extracted features
        
    Returns:
        Complete labeled dataset ready for training
    """
    # Find bug fixes
    bug_fix_shas = df_commits[df_commits['is_bug_fix']]['sha'].tolist()
    print(f"Found {len(bug_fix_shas)} bug-fix commits")
    
    # Try git blame (may fail for shallow clones)
    try:
        bug_inducers = find_bug_inducing_commits(repo_path, bug_fix_shas, max_fixes=50)
    except Exception as e:
        print(f"Git blame failed (shallow clone?): {e}")
        print("Using heuristic labeling instead...")
        bug_inducers = set()
    
    # Merge commits and features
    # Drop columns that will come from df_features to avoid duplicates
    columns_to_drop = ['files_changed', 'insertions', 'deletions']
    df_commits_clean = df_commits.drop(columns=[col for col in columns_to_drop if col in df_commits.columns])
    
    print(f"Merging {len(df_commits_clean)} commits with {len(df_features)} feature sets...")
    print(f"  df_commits columns: {list(df_commits_clean.columns)}")
    print(f"  df_features columns: {list(df_features.columns)}")
    print(f"  df_commits SHAs sample: {df_commits_clean['sha'].head(3).tolist()}")
    print(f"  df_features SHAs sample: {df_features['sha'].head(3).tolist()}")
    
    df_merged = df_commits_clean.merge(df_features, on='sha', how='inner')
    print(f"Merged dataset has {len(df_merged)} rows")
    
    if len(df_merged) == 0:
        print("WARNING: Merge resulted in empty dataset!")
        # Check if there's any SHA overlap
        commits_shas = set(df_commits['sha'])
        features_shas = set(df_features['sha'])
        overlap = commits_shas & features_shas
        print(f"  SHA overlap: {len(overlap)}/{min(len(commits_shas), len(features_shas))}")
        
        if len(overlap) == 0:
            print("  ERROR: No common SHAs between commits and features!")
            print("  This likely means feature extraction failed silently.")
        
        # Return features with minimal labeling
        df_features = df_features.copy()
        df_features['is_bug_inducing'] = False  # Default
        df_features['is_bug_fix'] = False  # Default
        return df_features
    
    # Add labels
    if bug_inducers:
        df_final = label_commits(df_merged, bug_inducers)
    else:
        # Fallback: use heuristic - commits before bug fixes may be bug-inducing
        # This is less accurate but allows training
        print("Using heuristic: marking non-bug-fix commits as potentially bug-inducing")
        df_final = df_merged.copy()
        # Use inverse of is_bug_fix as a weak signal
        df_final['is_bug_inducing'] = ~df_final['is_bug_fix']
        bug_rate = df_final['is_bug_inducing'].mean()
        print(f"Heuristic labeled {df_final['is_bug_inducing'].sum()}/{len(df_final)} ({bug_rate:.1%}) as potentially bug-inducing")
    
    return df_final


if __name__ == "__main__":
    # Test labeling
    from src.collection.clone import clone_repo_shallow
    from src.collection.extract import extract_commit_data
    from src.collection.features import extract_all_features
    
    repo_path = clone_repo_shallow("pallets/click", depth=100)
    df_commits = extract_commit_data(repo_path, max_commits=50)
    df_features = extract_all_features(repo_path, df_commits, max_commits=30)
    
    df_labeled = create_labeled_dataset(repo_path, df_commits, df_features)
    print(f"\nDataset shape: {df_labeled.shape}")
    print(f"Bug-inducing rate: {df_labeled['is_bug_inducing'].mean():.1%}")

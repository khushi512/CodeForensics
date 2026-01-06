"""
Feature engineering for commit analysis.
Extracts code complexity and churn metrics using tree-sitter.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from git import Repo
from typing import Dict, Any, Optional
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FEATURE_COLUMNS

# Setup tree-sitter for Python
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

# Decision nodes for cyclomatic complexity
DECISION_NODES = [
    'if_statement', 'while_statement', 'for_statement',
    'except_clause', 'elif_clause', 'and', 'or',
    'conditional_expression', 'match_statement'
]


def calculate_complexity(code: str) -> int:
    """
    Calculate cyclomatic complexity from Python code using AST.
    
    Args:
        code: Python source code
        
    Returns:
        Cyclomatic complexity score
    """
    try:
        tree = parser.parse(bytes(code, "utf8"))
        
        def count_nodes(node) -> int:
            count = 1 if node.type in DECISION_NODES else 0
            for child in node.children:
                count += count_nodes(child)
            return count
        
        return count_nodes(tree.root_node)
    except Exception:
        return 0


def get_file_history(repo: Repo, file_path: str, max_commits: int = 50) -> int:
    """
    Get the number of times a file has been modified (churn).
    
    Args:
        repo: GitPython Repo object
        file_path: Path to file relative to repo root
        max_commits: Maximum commits to look back
        
    Returns:
        Number of commits that modified this file
    """
    try:
        commits = list(repo.iter_commits(paths=file_path, max_count=max_commits))
        return len(commits)
    except Exception:
        return 0


def extract_features_for_commit(repo_path: Path, commit_sha: str) -> Optional[Dict[str, Any]]:
    """
    Extract all features for a single commit.
    
    Args:
        repo_path: Path to git repository
        commit_sha: SHA of commit to analyze
        
    Returns:
        Dictionary of features or None if extraction fails
    """
    try:
        repo = Repo(repo_path)
        commit = repo.commit(commit_sha)
        
        # Basic features
        features: Dict[str, Any] = {
            'sha': commit_sha,
            'hour_of_day': commit.committed_datetime.hour,
            'day_of_week': commit.committed_datetime.weekday(),
            'is_weekend': commit.committed_datetime.weekday() >= 5,  # Sat/Sun
            'files_changed': len(commit.stats.files),
            'lines_added': commit.stats.total.get('insertions', 0),
            'lines_deleted': commit.stats.total.get('deletions', 0),
        }
        features['net_lines'] = features['lines_added'] - features['lines_deleted']
        
        # Count Python files modified
        python_files = [f for f in commit.stats.files.keys() if f.endswith('.py')]
        features['py_files_modified'] = len(python_files)
        
        # Calculate complexity delta for Python files
        complexity_delta = 0
        
        for file_path in python_files[:10]:  # Limit to avoid slowdown
            try:
                parent = commit.parents[0] if commit.parents else None
                if parent:
                    # Get file content before and after
                    try:
                        old_content = (parent.tree / file_path).data_stream.read().decode('utf-8', errors='ignore')
                        old_complexity = calculate_complexity(old_content)
                    except (KeyError, Exception):
                        old_complexity = 0
                    
                    try:
                        new_content = (commit.tree / file_path).data_stream.read().decode('utf-8', errors='ignore')
                        new_complexity = calculate_complexity(new_content)
                    except (KeyError, Exception):
                        new_complexity = 0
                    
                    complexity_delta += (new_complexity - old_complexity)
            except Exception:
                continue
        
        features['complexity_delta'] = complexity_delta
        
        # File churn (average change frequency)
        total_churn = 0
        file_count = min(len(commit.stats.files), 10)  # Limit files
        
        for file_path in list(commit.stats.files.keys())[:file_count]:
            total_churn += get_file_history(repo, file_path, max_commits=30)
        
        features['avg_file_churn'] = total_churn / max(file_count, 1)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features for {commit_sha[:8]}: {e}")
        return None


def extract_all_features(repo_path: Path, df_commits: pd.DataFrame, 
                         max_commits: int = 400) -> pd.DataFrame:
    """
    Extract features for all commits in a DataFrame.
    
    Args:
        repo_path: Path to git repository
        df_commits: DataFrame with commit data (must have 'sha' column)
        max_commits: Maximum commits to process
        
    Returns:
        DataFrame with extracted features
    """
    features_list = []
    shas = df_commits['sha'].tolist()[:max_commits]
    
    print(f"Extracting features for {len(shas)} commits...")
    
    for i, sha in enumerate(shas):
        features = extract_features_for_commit(repo_path, sha)
        if features:
            features_list.append(features)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(shas)} commits...")
    
    df_features = pd.DataFrame(features_list)
    print(f"Extracted features for {len(df_features)} commits")
    return df_features


if __name__ == "__main__":
    # Test feature extraction
    from clone import clone_repo_shallow
    from extract import extract_commit_data
    
    repo_path = clone_repo_shallow("pallets/click", depth=50)
    df_commits = extract_commit_data(repo_path, max_commits=20)
    df_features = extract_all_features(repo_path, df_commits, max_commits=10)
    
    print("\nFeature sample:")
    print(df_features[FEATURE_COLUMNS].head())

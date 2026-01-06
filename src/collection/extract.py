"""
Commit data extraction from git repositories.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
from git import Repo
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BUG_KEYWORDS, MAX_COMMITS


def is_bug_fix(message: str) -> bool:
    """
    Identify if a commit message indicates a bug fix.
    
    Args:
        message: Commit message text
        
    Returns:
        True if message contains bug-related keywords
    """
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in BUG_KEYWORDS)


def extract_commit_data(repo_path: Path, max_commits: int = MAX_COMMITS) -> pd.DataFrame:
    """
    Extract commit metadata from a repository.
    
    Args:
        repo_path: Path to git repository
        max_commits: Maximum number of commits to extract
        
    Returns:
        DataFrame with commit data
    """
    repo = Repo(repo_path)
    commits_data: List[Dict[str, Any]] = []
    
    print(f"Extracting commits from {repo_path}...")
    
    for i, commit in enumerate(repo.iter_commits('HEAD')):
        if i >= max_commits:
            break
            
        try:
            # Get commit stats
            stats = commit.stats.total
            
            data = {
                'sha': commit.hexsha,
                'message': commit.message.strip()[:500],  # Limit message length
                'author': commit.author.email if commit.author else 'unknown',
                'timestamp': datetime.fromtimestamp(commit.committed_date),
                'files_changed': len(commit.stats.files),
                'insertions': stats.get('insertions', 0),
                'deletions': stats.get('deletions', 0),
                'is_bug_fix': is_bug_fix(commit.message),
                'repo_name': repo_path.name
            }
            commits_data.append(data)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} commits...")
                
        except Exception as e:
            print(f"  Error processing commit {commit.hexsha[:8]}: {e}")
            continue
    
    df = pd.DataFrame(commits_data)
    print(f"Extracted {len(df)} commits, {df['is_bug_fix'].sum()} bug fixes identified")
    return df


def save_commits(df: pd.DataFrame, output_path: Path) -> None:
    """Save commits DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    # Test extraction
    from clone import clone_repo_shallow
    
    repo_path = clone_repo_shallow("pallets/click", depth=100)
    df = extract_commit_data(repo_path, max_commits=50)
    print(df.head())
    print(f"\nBug fix rate: {df['is_bug_fix'].mean():.1%}")

"""
Repository cloning utilities with shallow clone support.
Optimized for laptop storage constraints.
"""
import shutil
from pathlib import Path
from git import Repo
from git.exc import GitCommandError

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import REPOS_DIR, CLONE_DEPTH


def clone_repo_shallow(repo_url: str, depth: int = CLONE_DEPTH) -> Path:
    """
    Clone a repository with limited history (shallow clone).
    
    Args:
        repo_url: GitHub repo in format 'owner/repo'
        depth: Number of commits to fetch
        
    Returns:
        Path to cloned repository
    """
    repo_name = repo_url.split('/')[-1]
    repo_path = REPOS_DIR / repo_name
    
    if repo_path.exists():
        print(f"Repository {repo_name} already exists at {repo_path}")
        return repo_path
    
    print(f"Cloning {repo_url} (depth={depth})...")
    try:
        Repo.clone_from(
            f"https://github.com/{repo_url}.git",
            repo_path,
            depth=depth
        )
        print(f"Successfully cloned to {repo_path}")
        return repo_path
    except GitCommandError as e:
        print(f"Error cloning {repo_url}: {e}")
        raise


def cleanup_repo(repo_path: Path) -> None:
    """
    Delete a repository to free up disk space.
    Handles Windows file lock issues gracefully.
    
    Args:
        repo_path: Path to repository to delete
    """
    if not repo_path.exists():
        print(f"Repository {repo_path} does not exist")
        return
        
    print(f"Cleaning up {repo_path}...")
    try:
        shutil.rmtree(repo_path)
        print(f"✓ Deleted {repo_path}")
    except PermissionError as e:
        print(f"⚠️  Could not delete {repo_path} (Windows file lock)")
        print(f"   You can manually delete it later to free up space")
        # Don't raise - allow pipeline to continue
    except Exception as e:
        print(f"⚠️  Cleanup failed for {repo_path}: {e}")
        # Don't raise - allow pipeline to continue


def get_repo_size_mb(repo_path: Path) -> float:
    """Get repository size in megabytes."""
    total_size = sum(f.stat().st_size for f in repo_path.rglob('*') if f.is_file())
    return total_size / (1024 * 1024)


if __name__ == "__main__":
    # Test with a small repo
    test_repo = "pallets/click"
    repo_path = clone_repo_shallow(test_repo, depth=100)
    print(f"Repo size: {get_repo_size_mb(repo_path):.2f} MB")

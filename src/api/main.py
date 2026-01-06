"""
CodeForensics FastAPI Application.
Analyzes git commits to predict bug risk.
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.db import get_db, init_db, Analysis, TrackedRepo
from src.collection.clone import clone_repo_shallow, cleanup_repo
from src.collection.extract import extract_commit_data
from src.collection.features import extract_features_for_commit
from src.model.predict import predict_risk, load_model
from config import CLONE_DEPTH

# Initialize FastAPI app
app = FastAPI(
    title="CodeForensics API",
    description="ML-powered bug risk prediction for git commits",
    version="1.0.0"
)

# CORS for VSCode extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None


@app.on_event("startup")
async def startup():
    global model
    init_db()
    try:
        model = load_model()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Train a model first using: python -m src.model.train")


# Request/Response models
class CommitAnalysisRequest(BaseModel):
    repo_url: str  # e.g., "pallets/flask"
    commit_sha: str


class CommitAnalysisResponse(BaseModel):
    commit_sha: str
    risk_score: float
    risk_level: str
    features: dict
    analyzed_at: datetime


class RepoTrackRequest(BaseModel):
    repo_url: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/analyze", response_model=CommitAnalysisResponse)
async def analyze_commit(
    request: CommitAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze a specific commit for bug risk.
    
    Clones the repo temporarily, extracts features, and returns prediction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")
    
    # Validate input format
    if not request.repo_url or request.repo_url.lower() == "string":
        raise HTTPException(
            status_code=400, 
            detail="Invalid repo_url. Expected format: 'owner/repo' (e.g., 'pallets/flask')"
        )
    
    if not request.commit_sha or request.commit_sha.lower() == "string":
        raise HTTPException(
            status_code=400,
            detail="Invalid commit_sha. Expected a 40-character Git SHA hash"
        )
    
    # Validate repo_url format
    if '/' not in request.repo_url or len(request.repo_url.split('/')) != 2:
        raise HTTPException(
            status_code=400,
            detail="Invalid repo_url format. Expected 'owner/repo' (e.g., 'pallets/flask')"
        )
    
    # Validate commit SHA format (basic check)
    if len(request.commit_sha) not in [7, 40] or not all(c in '0123456789abcdef' for c in request.commit_sha.lower()):
        raise HTTPException(
            status_code=400,
            detail="Invalid commit_sha format. Expected a Git SHA hash (7-40 hex characters)"
        )
    
    # Check if already analyzed
    existing = db.query(Analysis).filter(
        Analysis.repo_url == request.repo_url,
        Analysis.commit_sha == request.commit_sha
    ).first()
    
    if existing:
        return CommitAnalysisResponse(
            commit_sha=existing.commit_sha,
            risk_score=existing.risk_score,
            risk_level=existing.risk_level,
            features={
                "hour_of_day": existing.hour_of_day,
                "day_of_week": existing.day_of_week,
                "files_changed": existing.files_changed,
                "lines_added": existing.lines_added,
                "lines_deleted": existing.lines_deleted,
                "complexity_delta": existing.complexity_delta,
                "avg_file_churn": existing.avg_file_churn
            },
            analyzed_at=existing.analyzed_at
        )
    
    # Clone and analyze
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            from git import Repo, GitCommandError
            repo_path = Path(tmpdir) / request.repo_url.split('/')[-1]
            
            print(f"Cloning {request.repo_url} for analysis...")
            
            # Clone with better error handling
            try:
                Repo.clone_from(
                    f"https://github.com/{request.repo_url}.git",
                    repo_path,
                    depth=100
                )
            except GitCommandError as e:
                error_msg = str(e)
                if "not found" in error_msg.lower() or "repository not found" in error_msg.lower():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Repository '{request.repo_url}' not found on GitHub. Check the repo_url format."
                    )
                elif "authentication" in error_msg.lower() or "permission" in error_msg.lower():
                    raise HTTPException(
                        status_code=403,
                        detail=f"Cannot access repository '{request.repo_url}'. It may be private."
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Git clone failed: {error_msg}"
                    )
            
            # Extract features
            try:
                features = extract_features_for_commit(repo_path, request.commit_sha)
            except Exception as e:
                error_msg = str(e)
                if "unknown revision" in error_msg.lower() or "does not exist" in error_msg.lower():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Commit '{request.commit_sha}' not found in repository '{request.repo_url}'"
                    )
                raise HTTPException(
                    status_code=500,
                    detail=f"Feature extraction failed: {error_msg}"
                )
            
            if not features:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Could not extract features for commit '{request.commit_sha}'. It may not exist in this repository."
                )
            
            # Predict
            result = predict_risk(features, model)
            
            # Store in database
            analysis = Analysis(
                repo_url=request.repo_url,
                commit_sha=request.commit_sha,
                risk_score=result['risk_score'],
                risk_level=result['risk_level'],
                hour_of_day=features.get('hour_of_day'),
                day_of_week=features.get('day_of_week'),
                files_changed=features.get('files_changed'),
                lines_added=features.get('lines_added'),
                lines_deleted=features.get('lines_deleted'),
                complexity_delta=features.get('complexity_delta'),
                avg_file_churn=features.get('avg_file_churn')
            )
            db.add(analysis)
            db.commit()
            db.refresh(analysis)
            
            return CommitAnalysisResponse(
                commit_sha=request.commit_sha,
                risk_score=result['risk_score'],
                risk_level=result['risk_level'],
                features=features,
                analyzed_at=analysis.analyzed_at
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions (already formatted)
            raise
        except Exception as e:
            # Catch any other unexpected errors
            raise HTTPException(
                status_code=500, 
                detail=f"Unexpected error during analysis: {str(e)}"
            )


@app.get("/history/{repo_url:path}", response_model=List[CommitAnalysisResponse])
async def get_analysis_history(
    repo_url: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get analysis history for a repository."""
    analyses = db.query(Analysis).filter(
        Analysis.repo_url == repo_url
    ).order_by(Analysis.analyzed_at.desc()).limit(limit).all()
    
    return [
        CommitAnalysisResponse(
            commit_sha=a.commit_sha,
            risk_score=a.risk_score,
            risk_level=a.risk_level,
            features={
                "hour_of_day": a.hour_of_day,
                "day_of_week": a.day_of_week,
                "files_changed": a.files_changed,
                "lines_added": a.lines_added,
                "lines_deleted": a.lines_deleted,
                "complexity_delta": a.complexity_delta,
                "avg_file_churn": a.avg_file_churn
            },
            analyzed_at=a.analyzed_at
        )
        for a in analyses
    ]


@app.post("/repos/track")
async def track_repo(request: RepoTrackRequest, db: Session = Depends(get_db)):
    """Add a repository to tracking."""
    existing = db.query(TrackedRepo).filter(TrackedRepo.repo_url == request.repo_url).first()
    
    if existing:
        existing.is_active = True
        db.commit()
        return {"message": f"Repository {request.repo_url} reactivated"}
    
    repo = TrackedRepo(repo_url=request.repo_url)
    db.add(repo)
    db.commit()
    
    return {"message": f"Repository {request.repo_url} now being tracked"}


@app.get("/repos")
async def list_tracked_repos(db: Session = Depends(get_db)):
    """List all tracked repositories."""
    repos = db.query(TrackedRepo).filter(TrackedRepo.is_active == True).all()
    return [{"repo_url": r.repo_url, "added_at": r.added_at} for r in repos]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

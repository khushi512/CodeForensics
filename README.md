# CodeForensics

Machine learning system for predicting bug-inducing commits in Git repositories using LightGBM.

**Live API**: https://codeforensics.onrender.com  
**Documentation**: https://codeforensics.onrender.com/docs

## Overview

CodeForensics analyzes Git commit metadata and code changes to predict whether a commit is likely to introduce bugs. The system uses tree-sitter for code complexity analysis and git blame for labeling bug-inducing commits.

## Architecture

- **Data Pipeline**: Shallow clones repositories, extracts commit metadata, and computes features
- **Feature Engineering**: Extracts 10 features including cyclomatic complexity delta, file churn, and temporal patterns
- **ML Model**: LightGBM binary classifier trained on labeled commit data
- **API**: FastAPI REST service with PostgreSQL backend, deployed on Render

## Tech Stack

- **ML/Data**: Python, LightGBM, pandas, tree-sitter
- **Backend**: FastAPI, SQLAlchemy, PostgreSQL
- **Infrastructure**: Docker, Render
- **Version Control**: GitPython

## Features Extracted

| Feature | Description |
|---------|-------------|
| `hour_of_day` | Commit timestamp hour (0-23) |
| `day_of_week` | Day of week (0-6) |
| `is_weekend` | Boolean flag for weekend commits |
| `files_changed` | Number of files modified |
| `py_files_modified` | Number of Python files modified |
| `lines_added` | Lines added in commit |
| `lines_deleted` | Lines deleted in commit |
| `net_lines` | Net line change |
| `complexity_delta` | Change in cyclomatic complexity |
| `avg_file_churn` | Average modification frequency of changed files |

## API Usage

### Analyze a Commit

```bash
curl -X POST "https://codeforensics.onrender.com/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "pallets/flask",
    "commit_sha": "abc123..."
  }'
```

Response:
```json
{
  "commit_sha": "abc123...",
  "risk_score": 0.074,
  "risk_level": "LOW",
  "features": { ... },
  "analyzed_at": "2026-01-06T10:00:00"
}
```

### Get Analysis History

```bash
curl "https://codeforensics.onrender.com/history/pallets/flask?limit=10"
```

## Local Development

### Setup

```bash
# Clone repository
git clone https://github.com/khushi512/CodeForensics.git
cd CodeForensics

# Install dependencies
pip install -r requirements.txt

# Run data pipeline
python -m src.pipeline

# Train model
python -m src.model.train

# Start API
uvicorn src.api.main:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

### Configuration

Edit `config.py` to modify:
- Repository list for data collection
- Model hyperparameters
- Feature extraction settings
- Database connection

## Dataset

Current dataset: 400 commits from pallets/click repository with 18.5% bug-inducing rate.

## Deployment

Deployed on Render with:
- Docker containerization
- PostgreSQL database
- Automatic deployments from main branch

## Future Enhancements

- Expand dataset to 50+ repositories
- Add JavaScript/TypeScript support via tree-sitter
- Develop VS Code extension for real-time commit analysis
- Improve feature engineering with code review metrics
- Implement A/B testing framework for model improvements

## License

MIT

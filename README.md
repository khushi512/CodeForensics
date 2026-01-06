# CodeForensics

ML-powered bug risk prediction for git commits.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect training data
```bash
python -m src.pipeline
```
This will clone repos, extract features, and create `data/datasets/training_dataset.csv`.

### 3. Train model
```bash
python -m src.model.train
```
This saves the model to `models/codeforensics_model.pkl`.

### 4. Run API
```bash
uvicorn src.api.main:app --reload
```
API docs at: http://localhost:8000/docs

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/analyze` | Analyze a commit |
| GET | `/history/{repo}` | Get analysis history |
| POST | `/repos/track` | Track a repository |
| GET | `/repos` | List tracked repos |

## Example Request

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "pallets/flask", "commit_sha": "abc123"}'
```

## Deployment

Deploy to Render:
1. Push to GitHub
2. Connect repo in Render Dashboard
3. Use `render.yaml` blueprint

## Features
- Tree-sitter code complexity analysis
- Git blame bug tracing
- LightGBM prediction model
- PostgreSQL storage (SQLite for local dev)

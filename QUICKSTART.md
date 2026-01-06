# CodeForensics - Quick Start Guide

## What You Have Now

✅ ML-powered bug prediction system
✅ 400 commits analyzed from click repo  
✅ LightGBM model trained (81% risk prediction works!)
✅ FastAPI backend ready

---

## Test It Right Now (3 Commands)

### 1. Start the API
```bash
uvicorn src.api.main:app --reload
```

### 2. Visit Swagger UI
Open browser: `http://localhost:8000/docs`

### 3. Test Prediction
Click "Try it out" on `/analyze` endpoint

---

## Commands Reference

```bash
# Run complete data pipeline
python -m src.pipeline

# Show dataset statistics  
python -m src.pipeline stats

# Train model
python -m src.model.train

# Test prediction
python test_prediction.py

# Start API server
uvicorn src.api.main:app --reload

# Initialize database
python -c "from src.api.db import init_db; init_db()"
```

---

## What's Next?

**Choose one:**
- **A.** Test API locally → I'll guide you
- **B.** Collect more data (3-5 repos) → Better model
- **C.** Deploy to Render → Production!

Just tell me A, B, or C! 🚀

# Two-Tower Recommender (MovieLens Small)

End-to-end implementation of a two-tower recommendation system using the MovieLens dataset. This project demonstrates core concepts in modern recommendation systems: embedding-based collaborative filtering, in-batch negative sampling, temporal evaluation, and efficient retrieval using approximate nearest neighbor search.

## Features
- **Two-tower architecture**: Separate user and item embedding towers for scalable retrieval
- **In-batch negative sampling**: Efficient training without explicit negative generation
- **Temporal evaluation**: Time-based train/validation splits with proper seen-item filtering
- **FAISS integration**: Fast similarity search for real-time recommendations
- **REST API serving**: FastAPI-based recommendation service

## Tech Stack
PyTorch • FAISS • FastAPI • Pandas • NumPy

## Performance
Evaluated on 438 active users with temporal splits:
- **Hit Rate@10: 28.3%** (users receiving relevant recommendations)
- Recall@10: 2.8%
- NDCG@10: 4.5%

## Train
```bash
# Basic training
python -m src.train --config configs/movielens-small.yaml

# Training with evaluation for best metrics
python -m src.train_eval --config configs/movielens-small-max-recall.yaml
```

## Serve
```bash
export ARTIFACTS_DIR=artifacts/movielens_small_twotower
uvicorn src.serve:app --reload --port 8000
```

## Test
```bash
curl "http://127.0.0.1:8000/healthz"
curl "http://127.0.0.1:8000/recommend?user_id=1&k=10"
```

## Evaluate + Plot
```bash
python -m src.train_eval --config configs/movielens-small-eval.yaml
open artifacts/movielens_small_twotower_eval/recall_at_k.png  # macOS
```

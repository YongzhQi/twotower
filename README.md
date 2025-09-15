# Two-Tower Recommender (MovieLens Small)

End-to-end demo: train a two-tower model, build a FAISS index, and serve recommendations via FastAPI.

## Train
```bash
python -m src.train --config configs/movielens-small.yaml
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

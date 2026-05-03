# 🎬 CineAI — AI-Powered Movie Recommendation Engine

A production-grade movie recommendation system that goes beyond basic collaborative filtering by implementing and benchmarking **4 distinct ML models** — from a TF-IDF baseline to gradient-boosted re-rankers — with a live Streamlit interface powered by the TMDB API.

---

## 🤖 Models

| Model | Approach | Badge |
|-------|----------|-------|
| **Cosine Similarity** | TF-IDF vectors + cosine distance on movie tags | Baseline |
| **SVD / Latent Semantic** | Truncated SVD (300-dim) to capture hidden thematic structure | Semantic |
| **XGBoost Re-ranker** | Gradient boosting on 6 pairwise similarity features over top-60 cosine candidates | ML |
| **CatBoost Re-ranker** | Yandex's ordered boosting for non-linear re-ranking with structured features | ML+ |

The XGBoost and CatBoost re-rankers are trained on pairwise features including cosine similarity, SVD dot product, Jaccard overlap, tag length ratio, and positional distance — enabling non-linear re-ranking beyond what vector similarity alone can achieve.

---

## ✨ Features

- **4-model recommendation engine** with switchable model selection in the sidebar
- **Model benchmarking dashboard** showing live Precision, Recall, and NDCG per model
- **Mood-based filtering** across 7 genre categories (Feel-Good, Thriller, Sci-Fi, etc.)
- **TMDB API integration** for real-time posters, cast, director, runtime, and ratings (TTL-cached)
- **Session-state watchlist** and search history across the app lifecycle
- Animated movie cards with genre pills, star ratings, and expandable detail panels

---

## 🛠️ Tech Stack

`Python` · `Streamlit` · `XGBoost` · `CatBoost` · `scikit-learn` · `Truncated SVD` · `TF-IDF` · `TMDB API` · `Pickle`

---

## 🚀 Getting Started

```bash
git clone https://github.com/Viresh2408/ai-movie-recommendation-engine
cd ai-movie-recommendation-engine
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 How the Re-rankers Work

```
Input movie
    │
    ▼
Top-60 candidates via Cosine Similarity
    │
    ▼
Build 6 pairwise features per candidate:
  [cosine_sim, svd_dot, jaccard, tag_overlap, length_ratio, positional_dist]
    │
    ▼
XGBoost / CatBoost scores candidates
    │
    ▼
Top-N re-ranked recommendations
```

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application (1200+ lines) |
| `train_models.py` | Model training pipeline for all 4 models |
| `movierecommendor.py` | Core recommendation logic |
| `model.pkl` / `model_xgb.pkl` / `model_catboost.pkl` | Serialised model artifacts |
| `moviedata_svd.pkl` / `svd_matrix.pkl` | SVD similarity matrix and normalised embeddings |
| `tfidf.pkl` | TF-IDF vectoriser |

---

Built with ❤️ by [Viresh Kumbhar](https://github.com/Viresh2408)

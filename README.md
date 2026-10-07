# Netflix-Recommendation-Engine----Singular-Value-Decomposition
# 🎬 Netflix Movie Recommender (SVD Collaborative Filtering)

A movie recommendation system built on the **Netflix Prize dataset**. It cleans and filters millions of raw ratings, trains an **SVD matrix-factorization model**, and serves personalized top-N recommendations through a live **Streamlit** web app.

**🔗 Live demo:** https://netflix-recommendation-engine----singular-value-decomposition.streamlit.app/

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Dataset](#dataset)
5. [How It Works](#how-it-works)
6. [Project Structure](#project-structure)
7. [Setup and Run Locally](#setup-and-run-locally)
8. [Retraining the Model](#retraining-the-model)
9. [Deployment](#deployment)
10. [Results](#results)
11. [Limitations](#limitations)
12. [Future Improvements](#future-improvements)
13. [Author](#author)

---

## Overview

Streaming platforms have millions of users and thousands of titles, but each user rates only a tiny fraction of them. This project predicts how a user would rate movies they have not seen, then recommends the highest-scoring ones.

It uses **collaborative filtering**: recommendations come purely from rating patterns (who rated what), with no need for genre or plot information.

## Features

- Personalized **top-N recommendations** for any customer in the trained sample (3 to 20 results)
- Option to **hide movies the customer has already rated**
- **Random customer** button to explore the model quickly
- Predicted ratings shown on the original 1-5 scale
- Fast inference: scoring all movies is a single matrix multiplication, with no heavy ML library needed at serve time

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data processing | pandas, NumPy |
| Visualization (EDA) | Matplotlib, Seaborn |
| Modeling | scikit-surprise (SVD) |
| Web app | Streamlit |
| Hosting | Streamlit Community Cloud + GitHub |
| Training environment | Google Colab |

## Dataset

- **Source:** Netflix Prize dataset (`combined_data_1.txt` and `movie_titles.csv`)
- **Size:** about 24 million ratings (1-5 stars) from roughly 470,000 customers
- **Raw format:** a movie ID line (e.g. `1:`) followed by `CustomerID,Rating,Date` lines for that movie

> The raw data (about 500 MB) is **not** included in this repository. Only the small exported model files are.

## How It Works

### 1. Data loading and restructuring
The raw file has movie IDs on their own lines, so ratings are not directly linked to movies. The notebook loops through the file, detects the movie header rows, and assigns each rating its `Movie_Id`. The result is a clean table:

| Cust_Id | Rating | Movie_Id |
|---|---|---|

### 2. Exploratory analysis
- Count of movies, customers, and total ratings
- Rating distribution (1 to 5 stars) plotted as a bar chart

### 3. Filtering for quality
To reduce sparsity and noise, rows are filtered using the **60th percentile** of rating counts:
- Movies with too few ratings are removed
- Customers who rated too few movies are removed

### 4. Model: SVD (matrix factorization)
Each user and each movie is represented by a vector of latent factors. The predicted rating is:

```
predicted rating = global mean + user bias + movie bias + (user factors · movie factors)
```

- **Global mean:** the average rating overall
- **User bias:** some users rate generously, others harshly
- **Movie bias:** some movies are rated higher by everyone
- **Dot product:** the personalized part, matching user taste to movie traits

### 5. Evaluation
3-fold cross-validation with **RMSE** as the metric.

### 6. From notebook to web app
The notebook's raw data is too large to host and `scikit-surprise` is hard to install on hosting platforms. So the project uses a **train once, serve lightweight** approach:

1. Train the final SVD model in Colab on a **random sample of 20,000 users** (so every movie is represented).
2. Export the learned parameters (`pu`, `qi`, `bu`, `bi`, global mean) and ID mappings to `model_artifacts.npz`.
3. The Streamlit app loads these files and recomputes predictions with NumPy:

```python
scores = mu + bu[u] + bi + qi @ pu[u]   # scores every movie at once
```

## Project Structure

```
netflix-app/
├── app.py                  # Streamlit web app
├── export_artifacts.py     # Colab script: trains the final model and exports files
├── model_artifacts.npz     # Exported SVD parameters and ID mappings
├── movies.csv              # Movie_Id, Year, Name
├── requirements.txt        # Python dependencies for the app
└── README.md
```

## Setup and Run Locally

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

If the `streamlit` command is not recognized (common on Windows), run it through Python:

```bash
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Retraining the Model

1. Run the original notebook cells up to loading `df_title` (data loading, cleaning, filtering).
2. Add `export_artifacts.py` as a new cell and run it. It:
   - samples 20,000 random users (a fixed seed keeps results reproducible),
   - runs 3-fold cross-validation,
   - trains the final SVD model (`n_factors=50`) on all sampled data,
   - downloads `model_artifacts.npz` and `movies.csv`.
3. Replace the two files in this repository.

To change the model size, edit the `size=20000` sample or `n_factors`.

## Deployment

The app is deployed on **Streamlit Community Cloud**:

1. Push `app.py`, `requirements.txt`, `model_artifacts.npz`, and `movies.csv` to a GitHub repository.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **Create app**, select the repository, set the main file to `app.py`, and click **Deploy**.
4. Every push to GitHub automatically redeploys the app.

## Results

| Metric | Value |
|---|---|
| Model | SVD (`n_factors=50`) |
| Validation | 3-fold cross-validation |
| RMSE | ~1.02 *(update with your own final number)* |

An RMSE near 1.0 means predictions are typically off by about one star on the 1-5 scale. This is a reasonable baseline for an untuned model.

## Limitations

- **Cold start:** new users and new movies have no ratings, so they cannot be scored.
- **Filtering** removes rarely rated movies and low-activity users, so those cannot be recommended.
- **Sampled users:** the deployed model covers 20,000 users, not the full customer base, because of repository and memory size limits.
- Only `combined_data_1.txt` is used (one of four files in the full dataset).
- Evaluated with RMSE only, which measures rating accuracy, not ranking quality.

## Future Improvements

- Tune hyperparameters (`n_factors`, `lr_all`, `reg_all`) with `GridSearchCV`
- Add ranking metrics such as precision@k and recall@k
- Use all four data files and split train/test by time
- Build a hybrid model that adds genre or metadata for the cold-start problem
- Add a "popular movies" fallback for unknown users
- Replace the ID input with a searchable dropdown and add movie posters via an external API

## Author

**Naman Shah**
Computer Science student (Data Science and Machine Learning)

---

*Dataset credit: Netflix Prize dataset. This project is for educational purposes.*

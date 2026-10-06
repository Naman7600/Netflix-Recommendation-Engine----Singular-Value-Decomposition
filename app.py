import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Netflix Recommender", page_icon="🎬", layout="centered")


@st.cache_resource
def load_artifacts():
    z = np.load("model_artifacts.npz")
    movies = pd.read_csv("movies.csv").set_index("Movie_Id")
    uid_to_idx = {int(u): i for i, u in enumerate(z["user_ids"])}
    return {
        "pu": z["pu"], "qi": z["qi"], "bu": z["bu"], "bi": z["bi"],
        "mu": float(z["mu"]),
        "user_ids": z["user_ids"], "item_ids": z["item_ids"],
        "indptr": z["rated_indptr"], "indices": z["rated_indices"],
        "uid_to_idx": uid_to_idx, "movies": movies,
    }


def recommend(a, customer_id, n, hide_seen):
    u = a["uid_to_idx"][customer_id]
    # same formula SVD uses: mean + user bias + movie bias + user·movie
    scores = a["mu"] + a["bu"][u] + a["bi"] + a["qi"] @ a["pu"][u]
    scores = np.clip(scores, 1, 5)
    seen = a["indices"][a["indptr"][u]:a["indptr"][u + 1]]
    if hide_seen:
        scores = scores.copy()
        scores[seen] = -np.inf
    top = np.argsort(scores)[::-1][:n]
    movie_ids = a["item_ids"][top]
    out = pd.DataFrame({
        "Movie": [a["movies"].loc[m, "Name"] if m in a["movies"].index else f"Movie {m}" for m in movie_ids],
        "Year": [int(a["movies"].loc[m, "Year"]) if m in a["movies"].index else 0 for m in movie_ids],
        "Predicted rating": scores[top].round(2),
    })
    out["Year"] = out["Year"].replace(0, "")
    out.index = np.arange(1, len(out) + 1)
    return out, len(seen)


a = load_artifacts()

st.title("🎬 Netflix Movie Recommender")
st.caption("Collaborative filtering with SVD, trained on the Netflix Prize dataset.")

if "cid" not in st.session_state:
    st.session_state.cid = int(a["user_ids"][0])

if st.button("🎲 Pick a random customer"):
    st.session_state.cid = int(np.random.choice(a["user_ids"]))

customer_id = st.number_input("Customer ID", step=1, key="cid")
c1, c2 = st.columns(2)
n = c1.slider("Number of recommendations", 3, 20, 5)
hide_seen = c2.checkbox("Hide movies already rated", value=True)

if st.button("Recommend", type="primary"):
    if int(customer_id) not in a["uid_to_idx"]:
        st.error("This Customer ID isn't in the trained sample. Try the random-customer button.")
    else:
        recs, n_seen = recommend(a, int(customer_id), n, hide_seen)
        st.success(f"Customer {int(customer_id)} has rated {n_seen} movies in the data.")
        st.table(recs)

with st.expander("About this project"):
    st.write(
        "Ratings are loaded, cleaned, and filtered to active users and popular "
        "movies. An SVD matrix-factorisation model (scikit-surprise) learns latent "
        "taste factors, and predicted rating = global mean + user bias + movie bias "
        "+ user·movie factors."
    )

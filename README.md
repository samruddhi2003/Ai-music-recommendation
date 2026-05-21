# 🎵 Music Recommendation System

An AI-powered music recommendation web app built with Streamlit that suggests similar songs based on audio features using Machine Learning.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
music-recommendation-system/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── data/
│   └── dataset.csv         # Spotify dataset (114K songs)
└── notebooks/
    └── analysis.ipynb      # EDA and model experimentation
```

---

## 📊 Dataset

- **Source:** [Spotify Tracks Dataset on Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Download:** Dataset manually download karke `data/dataset.csv` pe rakh do
- **Size:** 114,000 songs × 21 columns
- **Key columns used:**

| Column | Description |
|---|---|
| `track_name` | Name of the song |
| `artists` | Artist(s), semicolon separated |
| `track_genre` | Genre of the song |
| `danceability` | How suitable for dancing (0.0 – 1.0) |
| `energy` | Intensity and activity level (0.0 – 1.0) |
| `valence` | Musical positiveness / happiness (0.0 – 1.0) |
| `tempo` | Beats per minute (BPM) |
| `loudness` | Overall loudness in dB |
| `popularity` | Spotify popularity score (0 – 100) |

---

## 🧠 Concepts Used

### 1. Data Cleaning
- **Removing remixes** — songs with "Remix", "Mix", "Edit", "Version" in name are filtered out using `str.contains()` with regex
- **Removing duplicates** — `drop_duplicates()` on `track_name` + `artists` to avoid same song appearing twice

```python
df = df[~df['track_name'].str.contains('Remix|Mix|Edit|Version', case=False)]
df = df.drop_duplicates(subset=['track_name', 'artists'])
```

---

### 2. Feature Selection
5 audio features were selected that best represent the "sound" of a song:
- `danceability`, `energy`, `valence`, `tempo`, `loudness`

These capture mood, intensity, and rhythm — enough to find sonically similar songs.

---

### 3. MinMax Scaling
**What:** Normalizes all features to the same range [0, 1] so no single feature dominates.

**Why needed:** `tempo` ranges from 0–250 BPM while `danceability` is 0–1. Without scaling, tempo would dominate the similarity calculation.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
genre_features = scaler.fit_transform(genre_df[features])
```

---

### 4. Cosine Similarity
**What:** Measures the angle between two vectors. Value ranges from 0 (completely different) to 1 (identical).

**Why cosine over euclidean:** Cosine similarity focuses on the direction/pattern of features, not the magnitude. Two songs with similar "shape" of audio features will score high even if their raw values differ.

**Formula:**
```
similarity = (A · B) / (||A|| × ||B||)
```

```python
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(song_vector, genre_features)[0]
```

---

### 5. Genre Filtering (Hybrid Approach)
Instead of comparing a song against all 114K songs, we first filter to the **same genre** and then apply cosine similarity within that genre.

**Why:** This makes recommendations more relevant and reduces computation. A pop song should not be compared to death metal.

This is what makes it a **Hybrid Recommendation System** — combining content-based filtering (audio features) with genre-based filtering.

---

### 6. Artist Boosting
Songs by the same artist get a 1.2× score boost so they appear higher in recommendations.

```python
if selected_artist in current_artist:
    similarities[i] *= 1.2
```

**Why:** Users who like a song often want more from the same artist.

---

### 7. Content-Based Filtering
**What:** Recommends items similar to what the user already likes, based on item features (not user history).

**How it works here:**
1. Take the searched song's audio feature vector
2. Compare it with all songs in the same genre using cosine similarity
3. Return top 10 most similar songs

**Advantage over Collaborative Filtering:** Does not need user data or ratings — works purely on song properties.

---

### 8. iTunes API Integration
Used Apple's free iTunes Search API to fetch:
- **Album cover art** (600×600 resolution)
- **30-second audio preview**

```python
response = requests.get(
    "https://itunes.apple.com/search",
    params={"term": f"{track_name} {artist}", "media": "music", "limit": 1}
)
```

**Why iTunes over Spotify API:** Spotify API requires premium subscription for certain operations. iTunes API is completely free with no authentication.

---

### 9. Streamlit
**What:** Python framework for building interactive web apps with pure Python — no HTML/CSS/JS required (though custom HTML can be injected).

**Key Streamlit concepts used:**

| Concept | Usage |
|---|---|
| `st.session_state` | Persist data across reruns (prefill search bar) |
| `st.rerun()` | Force page refresh when state changes |
| `st.empty()` | Placeholder for dynamic content (loading messages) |
| `st.columns()` | Side-by-side layout |
| `st.html()` | Render raw HTML safely |
| `st.markdown(..., unsafe_allow_html=True)` | Inject CSS globally |
| `st.audio()` | Native audio player |
| `st.image()` | Display images from URL |

---

### 10. Recommendation Analytics
After recommendations are generated, real metrics are calculated:

- **Avg Similarity** — mean cosine similarity score of all 10 recommendations
- **Artist Diversity** — how many unique artists appear in recommendations
- **Mood Consistency** — compares `valence` of seed song vs recommendations
  - `|seed_valence - avg_rec_valence| < 0.15` → High
  - `< 0.30` → Medium
  - `>= 0.30` → Low

---

## 🔧 Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| Streamlit | Web app framework |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | MinMaxScaler, Cosine Similarity |
| Requests | iTunes API calls |

---

## 💡 Key Design Decisions

1. **Genre filtering before similarity** — reduces search space from 114K to ~1K songs per genre, making it faster and more relevant
2. **MinMax scaling per genre** — scaler is fit on genre subset, not entire dataset, so scaling is relative to that genre's range
3. **iTunes over Spotify** — avoids API authentication and premium subscription issues
4. **`st.html()` for cards, `st.markdown()` for CSS** — `st.html()` renders in an iframe (safe sandbox), `st.markdown` with `unsafe_allow_html=True` injects CSS globally into the page

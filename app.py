import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Spotify AI Recommender",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    background-color: #121212 !important;
    color: #FFFFFF;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding: 2rem 3rem !important;
    max-width: 1200px;
}

section[data-testid="stSidebar"] {
    background-color: #000000 !important;
    border-right: 1px solid #282828;
}

section[data-testid="stSidebar"] * {
    color: #b3b3b3 !important;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }

/* Input */
.stTextInput > div > div > input {
    background-color: #2a2a2a !important;
    color: white !important;
    border-radius: 4px !important;
    border: 1px solid #535353 !important;
    padding: 14px 18px !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    transition: border 0.2s;
}
.stTextInput > div > div > input:focus {
    border: 1px solid #1DB954 !important;
    outline: none !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: #6a6a6a !important;
}

/* Button */
.stButton > button {
    background-color: #1DB954 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 500px !important;
    padding: 14px 32px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    transition: all 0.15s ease !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    background-color: #1ed760 !important;
    transform: scale(1.04) !important;
}

/* Link button */
.stLinkButton > a {
    background-color: transparent !important;
    color: #FFFFFF !important;
    border: 1px solid #535353 !important;
    border-radius: 500px !important;
    padding: 8px 20px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    transition: all 0.15s ease !important;
    text-decoration: none !important;
}
.stLinkButton > a:hover {
    border-color: #FFFFFF !important;
    transform: scale(1.02) !important;
}

/* Audio player */
audio {
    width: 100%;
    height: 36px;
    border-radius: 4px;
    filter: invert(1) hue-rotate(90deg);
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #282828;
    margin: 24px 0;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #1DB954 !important;
}
/* Mobile Responsive */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem !important;
    }
    h1 {
        font-size: 32px !important;
    }
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD & CLEAN DATA
# =====================================================

df = pd.read_csv("data/dataset.csv")

df = df[~df['track_name'].str.contains('Remix|Mix|Edit|Version', case=False, na=False)]
df = df.drop_duplicates(subset=['track_name', 'artists'])

features = ['danceability', 'energy', 'valence', 'tempo', 'loudness']
scaler = MinMaxScaler()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; padding:20px 0 20px 0; border-bottom:1px solid #282828;">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="#1DB954">
            <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
        </svg>
        <span style="color:white; font-size:16px; font-weight:700;">Music Recommender</span>
    </div>
    """, unsafe_allow_html=True)

    # DATASET STATS — real numbers from df
    total_songs = len(df)
    total_genres = df['track_genre'].nunique()
    total_artists = df['artists'].nunique()

    st.markdown("<div style='padding-top:16px;'></div>", unsafe_allow_html=True)
    st.html(
        '<p style="color:#b3b3b3; font-size:10px; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;">Dataset Stats</p>'
        '<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:4px;">'
        '<div style="background:#181818; border:1px solid #282828; border-radius:8px; padding:12px; text-align:center;">'
        '<div style="color:#1DB954; font-size:18px; font-weight:800;">' + f"{total_songs:,}" + '</div>'
        '<div style="color:#6a6a6a; font-size:10px; text-transform:uppercase; letter-spacing:1px; margin-top:2px;">Songs</div>'
        '</div>'
        '<div style="background:#181818; border:1px solid #282828; border-radius:8px; padding:12px; text-align:center;">'
        '<div style="color:#1DB954; font-size:18px; font-weight:800;">' + str(total_genres) + '</div>'
        '<div style="color:#6a6a6a; font-size:10px; text-transform:uppercase; letter-spacing:1px; margin-top:2px;">Genres</div>'
        '</div>'
        '<div style="background:#181818; border:1px solid #282828; border-radius:8px; padding:12px; text-align:center; grid-column:span 2;">'
        '<div style="color:#1DB954; font-size:18px; font-weight:800;">' + f"{total_artists:,}" + '</div>'
        '<div style="color:#6a6a6a; font-size:10px; text-transform:uppercase; letter-spacing:1px; margin-top:2px;">Artists</div>'
        '</div>'
        '</div>'
    )

    st.markdown("<div style='border-top:1px solid #282828; margin:16px 0 16px 0;'></div>", unsafe_allow_html=True)

    # TOP GENRES from dataset
    top_genres = df['track_genre'].value_counts().head(6).index.tolist()
    genre_html = '<p style="color:#b3b3b3; font-size:10px; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;">Top Genres</p><div style="display:flex; flex-wrap:wrap; gap:6px;">'
    for g in top_genres:
        genre_html += '<span style="background:#181818; border:1px solid #282828; color:#b3b3b3; font-size:11px; padding:4px 10px; border-radius:500px;">' + g + '</span>'
    genre_html += '</div>'
    st.html(genre_html)

    st.markdown("<div style='border-top:1px solid #282828; margin:16px 0 16px 0;'></div>", unsafe_allow_html=True)

    # RANDOM SONG SUGGESTIONS
    st.html('<p style="color:#b3b3b3; font-size:10px; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;">Try Searching</p>')
    suggestions = df.sample(5)['track_name'].tolist()
    for s in suggestions:
        st.markdown(
            f"<div style='background:#181818; border:1px solid #282828; border-radius:6px; padding:8px 12px; margin-bottom:6px; color:#b3b3b3; font-size:12px;'>🎵 {s}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='border-top:1px solid #282828; margin:16px 0 16px 0;'></div>", unsafe_allow_html=True)

    # HOW IT WORKS
    st.html(
        '<p style="color:#b3b3b3; font-size:10px; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;">How It Works</p>'
        '<div style="color:#6a6a6a; font-size:12px; line-height:2;">'
        '<div><span style="color:#1DB954; font-weight:700;">1.</span> Enter a song you love</div>'
        '<div><span style="color:#1DB954; font-weight:700;">2.</span> AI analyzes audio features</div>'
        '<div><span style="color:#1DB954; font-weight:700;">3.</span> Matches by mood, energy &amp; genre</div>'
        '<div><span style="color:#1DB954; font-weight:700;">4.</span> Returns top 10 similar songs</div>'
        '</div>'
    )

# =====================================================
# RECOMMEND FUNCTION
# =====================================================

def recommend(song_name, top_n=10):
    selected_song = df[df['track_name'].str.lower() == song_name.lower()].iloc[0]
    selected_artist = str(selected_song['artists']).split(";")[0]
    selected_genre = selected_song['track_genre']

    genre_df = df[df['track_genre'] == selected_genre].reset_index(drop=True)
    genre_features = scaler.fit_transform(genre_df[features])

    song_index = genre_df[genre_df['track_name'].str.lower() == song_name.lower()].index[0]
    song_vector = genre_features[song_index].reshape(1, -1)
    similarities = cosine_similarity(song_vector, genre_features)[0]

    for i in range(len(genre_df)):
        if selected_artist in str(genre_df.iloc[i]['artists']):
            similarities[i] *= 1.2

    similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
    recommendations = genre_df.iloc[similar_indices][['track_name', 'artists', 'track_genre']].copy()
    recommendations['score'] = similarities[similar_indices]
    return recommendations

# =====================================================
# ITUNES DATA
# =====================================================

def get_song_data(track_name, artist):
    try:
        query = f"{track_name} {artist.split(';')[0]}"
        response = requests.get(
            "https://itunes.apple.com/search",
            params={"term": query, "media": "music", "limit": 1},
            timeout=5
        )
        results = response.json().get("results", [])
        if not results:
            return {'album_cover': None, 'preview_url': None}
        item = results[0]
        album_cover = item.get("artworkUrl100", "").replace("100x100", "600x600")
        preview_url = item.get("previewUrl", None)
        return {'album_cover': album_cover or None, 'preview_url': preview_url}
    except:
        return {'album_cover': None, 'preview_url': None}

# =====================================================
# HERO
# =====================================================

st.markdown("""
<div style="text-align:center; padding: 48px 0 32px 0;">
    <div style="display:inline-flex; align-items:center; gap:14px; margin-bottom:16px;">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="#1DB954">
            <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
        </svg>
        <h1 style="color:#FFFFFF; font-size:clamp(28px, 5vw, 52px); font-weight:900; letter-spacing:-2px; margin:0;">
            Music <span style="color:#1DB954;">Recommender</span>
        </h1>
    </div>
    <p style="color:#b3b3b3; font-size:17px; font-weight:400; letter-spacing:0.2px;">
        Enter a song you love — we'll find 10 songs that sound just like it
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# STATS
# =====================================================

c1, c2, c3 = st.columns(3)
for col, number, label in [
    (c1, "114K+", "Songs in Library"),
    (c2, "100+", "Genres Covered"),
    (c3, "AI", "Hybrid Engine"),
]:
    with col:
        st.html(f"""
        <div style="
            background:#181818;
            border:1px solid #282828;
            border-radius:8px;
            padding:24px;
            text-align:center;
        ">
            <div style="color:#1DB954; font-size:36px; font-weight:800; letter-spacing:-1px;">{number}</div>
            <div style="color:#b3b3b3; font-size:13px; font-weight:500; margin-top:6px; text-transform:uppercase; letter-spacing:1px;">{label}</div>
        </div>
        """)

st.markdown("<div style='margin:32px 0 8px 0; border-top:1px solid #282828;'></div>", unsafe_allow_html=True)

# =====================================================
# SEARCH
# =====================================================

st.markdown("<p style='color:#b3b3b3; font-size:13px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px;'>Search</p>", unsafe_allow_html=True)

col_input, col_btn, col_surprise = st.columns([4, 1, 1])
with col_input:
    song_name = st.text_input("", value=st.session_state.get('prefill', ''), placeholder="Enter a song you like (e.g. Hey There Delilah)", label_visibility="collapsed", key="song_input")
with col_btn:
    st.write("")
    search = st.button("Find Similar")
with col_surprise:
    st.write("")
    surprise = st.button("🎲 Surprise Me")

if surprise:
    st.session_state['prefill'] = df.sample(1).iloc[0]['track_name']
    st.rerun()

if st.session_state.get('prefill') and not search:
    song_name = st.session_state['prefill']
    search = True

# =====================================================
# RESULTS
# =====================================================

if search and song_name:
    try:
        status = st.empty()

        status.markdown("<p style='color:#1DB954; font-size:14px; font-weight:600;'>🔍 Analyzing audio features...</p>", unsafe_allow_html=True)
        recommendations = recommend(song_name)

        status.markdown("<p style='color:#1DB954; font-size:14px; font-weight:600;'>🎯 Finding similar songs...</p>", unsafe_allow_html=True)
        seed_data = get_song_data(song_name, df[df['track_name'].str.lower() == song_name.lower()].iloc[0]['artists'])
        seed_cover = seed_data['album_cover']

        status.markdown("<p style='color:#1DB954; font-size:14px; font-weight:600;'>🎨 Fetching album art...</p>", unsafe_allow_html=True)
        all_song_data = [get_song_data(row['track_name'], row['artists']) for _, row in recommendations.iterrows()]

        status.empty()

        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        st.html(
            '<p style="color:#b3b3b3; font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;">Because you liked</p>'
            '<div style="display:flex; align-items:center; gap:16px; background:#181818; border:1px solid #1DB954; border-radius:10px; padding:16px 20px; margin-bottom:16px;">'
            + ('<img src="' + str(seed_cover) + '" style="width:64px; height:64px; border-radius:6px; object-fit:cover; flex-shrink:0;">' if seed_cover else '<div style="width:64px; height:64px; background:#282828; border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:24px;">🎵</div>')
            + '<div>'
            '<p style="color:#FFFFFF; font-size:18px; font-weight:800; margin:0 0 4px 0;">' + song_name.title() + '</p>'
            '<p style="color:#1DB954; font-size:12px; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin:0;">Seed Song · Finding similar tracks...</p>'
            '</div>'
            '</div>'
        )

        if seed_data.get('preview_url'):
            st.audio(seed_data['preview_url'])

        st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

        st.html('<p style="color:#b3b3b3; font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:16px;">Recommended for you</p>')

        for idx, (i, row) in enumerate(recommendations.iterrows()):
            song_data = all_song_data[idx]
            album_cover = song_data['album_cover']
            preview_url = song_data['preview_url']

            img_col, info_col = st.columns([1, 4])

            with img_col:
                if album_cover:
                    st.image(album_cover, width=160)
                else:
                    st.html("""
                    <div style="
                        width:160px; height:160px;
                        background:#282828;
                        border-radius:4px;
                        display:flex; align-items:center; justify-content:center;
                        font-size:40px;
                    ">🎵</div>
                    """)

            with info_col:
                score_pct = min(int(row['score'] * 100), 100)
                artists_display = str(row['artists']).replace(';', ', ')

                st.html(
                    '<div style="padding: 8px 0 4px 8px;">'
                    '<p style="color:#b3b3b3; font-size:12px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:6px;"># ' + str(idx + 1) + '</p>'
                    '<h3 style="color:#FFFFFF; font-size:22px; font-weight:800; letter-spacing:-0.3px; margin-bottom:6px;">' + str(row['track_name']) + '</h3>'
                    '<p style="color:#b3b3b3; font-size:14px; margin-bottom:4px;">' + artists_display + '</p>'
                    '<p style="color:#1DB954; font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">' + str(row['track_genre']) + '</p>'
                    '<div style="display:flex; align-items:center; gap:10px;">'
                    '<div style="background:#282828; border-radius:500px; height:4px; width:120px; overflow:hidden;">'
                    '<div style="background:#1DB954; height:4px; width:' + str(score_pct) + '%;"></div>'
                    '</div>'
                    '<span style="color:#b3b3b3; font-size:12px;">' + str(round(row['score'], 2)) + ' match</span>'
                    '</div>'
                    '</div>'
                )

                if preview_url:
                    st.audio(preview_url)

            st.markdown("<div style='border-top:1px solid #282828; margin: 8px 0 16px 0;'></div>", unsafe_allow_html=True)

        # ANALYTICS
        selected_song_row = df[df['track_name'].str.lower() == song_name.lower()].iloc[0]
        seed_genre = selected_song_row['track_genre']
        avg_similarity = round(recommendations['score'].mean(), 2)
        seed_valence = selected_song_row['valence']
        rec_valence = df[df['track_name'].isin(recommendations['track_name'])]['valence'].mean()
        mood_diff = abs(seed_valence - rec_valence)
        mood_consistency = 'High' if mood_diff < 0.15 else 'Medium' if mood_diff < 0.30 else 'Low'
        mood_color = '#1DB954' if mood_consistency == 'High' else '#f59e0b' if mood_consistency == 'Medium' else '#e22134'
        unique_artists = recommendations['artists'].apply(lambda x: str(x).split(';')[0]).nunique()
        artist_diversity_pct = int(unique_artists / len(recommendations) * 100)

        st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)
        st.html(
            '<p style="color:#b3b3b3; font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:16px;">Recommendation Analytics</p>'
            '<div style="background:#181818; border:1px solid #282828; border-radius:12px; padding:24px; display:grid; grid-template-columns:1fr 1fr; gap:20px;">'
            '<div><p style="color:#6a6a6a; font-size:11px; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;">Artist Diversity</p>'
            '<div style="display:flex; align-items:center; gap:10px;"><div style="background:#282828; border-radius:500px; height:6px; flex:1; overflow:hidden;"><div style="background:#1DB954; height:6px; width:' + str(artist_diversity_pct) + '%;"></div></div>'
            '<span style="color:#FFFFFF; font-size:16px; font-weight:700;">' + str(unique_artists) + ' artists</span></div></div>'
            '<div><p style="color:#6a6a6a; font-size:11px; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;">Avg Similarity</p>'
            '<div style="display:flex; align-items:center; gap:10px;"><div style="background:#282828; border-radius:500px; height:6px; flex:1; overflow:hidden;"><div style="background:#1DB954; height:6px; width:' + str(int(avg_similarity * 100)) + '%;"></div></div>'
            '<span style="color:#FFFFFF; font-size:16px; font-weight:700;">' + str(avg_similarity) + '</span></div></div>'
            '<div><p style="color:#6a6a6a; font-size:11px; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;">Mood Consistency</p>'
            '<span style="color:' + mood_color + '; font-size:16px; font-weight:700;">' + mood_consistency + '</span></div>'
            '<div><p style="color:#6a6a6a; font-size:11px; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;">Recommendation Type</p>'
            '<span style="color:#FFFFFF; font-size:14px; font-weight:600;">Hybrid AI</span></div>'
            '</div>'
        )

    except Exception as e:
        st.markdown(
            "<div style='background:#2a1a1a; border:1px solid #e22134; border-radius:8px; padding:16px 20px; color:#e22134; font-size:14px; margin-bottom:16px;'>"
            "❌ Song not found in dataset. Please check the spelling."
            "</div>",
            unsafe_allow_html=True
        )

elif search and not song_name:
    st.markdown("""
    <div style="background:#2a2a1a; border:1px solid #b3b300; border-radius:8px; padding:16px 20px; color:#b3b300; font-size:14px;">
        ⚠️ Please enter a song name.
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div style="text-align:center; padding:48px 0 24px 0; border-top:1px solid #282828; margin-top:48px;">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="#1DB954" style="margin-bottom:8px;">
        <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
    </svg>
    <p style="color:#535353; font-size:12px; margin-top:8px;">Built with Machine Learning · NLP · iTunes API · Streamlit</p>
</div>
""", unsafe_allow_html=True)

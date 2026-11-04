import sys
from flask import Flask, request, render_template_string
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.movie_engine import search_movie, get_movie, recommend_movies, get_movie_vector
import pickle
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR/"models/"
with open(MODEL_DIR /"movie_pca.pkl", "rb") as f:
    pca_model = pickle.load(f)
with open(MODEL_DIR /"movie_kmeans.pkl", "rb") as f:
    kmeans_model = pickle.load(f)
with open(MODEL_DIR / "movie_cluster_scaler.pkl", "rb") as f:
    cluster_scaler = pickle.load(f)
with open(MODEL_DIR / "movie_decision_tree_gini.pkl", "rb") as f:
    gini_model = pickle.load(f)
with open(MODEL_DIR / "movie_decision_tree_entropy.pkl", "rb") as f:
    entropy_model = pickle.load(f)
CLASS_FEATURES = ["popularity", "runtime", "vote_average", "release_year"]

# Prepare PCA coordinates for all movies; keep the app usable if the CSV is absent.
PCA_POINTS = None
try:
    movie_data = pd.read_csv(BASE_DIR / "datasets" / "processed" / "movies_ml.csv")
    pca_features = movie_data[CLASS_FEATURES].apply(pd.to_numeric, errors="coerce").dropna()
    PCA_POINTS = pca_model.transform(cluster_scaler.transform(pca_features))
except Exception as e:
    print("PCA overview unavailable:", e)
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Movie Recommendation System</title>
<style>
*{box-sizing:border-box}
body{margin:0;font:15px Arial,sans-serif;background:#f5f6fa;color:#202638}
header{background:#fff;border-bottom:1px solid #e2e5eb;padding:18px 7%;font-weight:700}
main{width:88%;max-width:1100px;margin:35px auto 60px}
.search{display:flex;max-width:720px;margin:0 auto 35px}
.search input{flex:1;min-width:0;padding:14px;border:1px solid #cfd4df;border-radius:8px 0 0 8px;font-size:15px}
button{padding:0 24px;border:0;border-radius:0 8px 8px 0;background:#5556c9;color:white;font-weight:600;cursor:pointer}
button:hover{background:#4445b5}
.results{max-width:720px;margin:-25px auto 35px;background:white;border:1px solid #e1e4eb;border-radius:8px;overflow:hidden}
.results a{display:block;padding:12px 16px;color:#202638;text-decoration:none;border-bottom:1px solid #eee}
.results a:hover{background:#f3f4ff}
.results small{display:block;margin-top:4px;color:#788195}
h2{font-size:22px;margin:0 0 16px}
h3{font-size:16px;margin:0 0 10px}
.panel{background:white;border:1px solid #e2e5eb;border-radius:10px;padding:22px;margin-bottom:25px}
.movie{display:flex;gap:25px;align-items:center}
.poster{width:115px;min-height:145px;flex-shrink:0;display:grid;place-items:center;text-align:center;padding:12px;border-radius:7px;background:#e7e9f7;color:#454d72;font-weight:700}
.meta{color:#687286;margin:8px 0}
.rating{color:#d79a00;font-weight:700}
.analysis-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.metric{padding:15px;background:#f6f7fb;border-radius:8px}
.metric span{display:block;color:#778094;font-size:12px;margin-bottom:7px}
.metric strong{font-size:18px}
.pca-formula{
    display:block;
    margin-top:10px;
    padding-top:8px;
    border-top:1px solid #e2e5eb;
    font-size:11px;
    line-height:1.7;
    color:#50586b;
    font-family:monospace;
    overflow-wrap:anywhere
}
.pca-variables{
    margin-top:16px;
    padding:14px 16px;
    background:#f6f7fb;
    border-radius:8px;
    font-size:13px;
    line-height:1.8;
    color:#50586b
}
.pca-variables b{
    display:block;
    margin-bottom:4px;
    color:#202638;
    font-size:14px
}
@media(max-width:850px){
    .analysis-grid{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:550px){
    .pca-formula{font-size:10px}
    .pca-variables{font-size:12px}
}
details{margin-top:14px;border-top:1px solid #eceef3;padding-top:12px}
summary{cursor:pointer;color:#5052bd;font-weight:600}
pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:280px;overflow:auto;background:#f6f7fb;padding:12px;border-radius:7px;font-size:11px}
table{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px}
th,td{text-align:left;padding:10px;border-bottom:1px solid #e8eaf0;vertical-align:top}
th{color:#687286}
.score{color:#5556c9;font-weight:700}
.plot{width:100%;max-width:760px;display:block;margin:15px auto 0}
.pca-note{max-width:760px;margin:8px auto 0;font-size:11px;line-height:1.7;color:#687286}
.pca-note-vars{margin-top:4px;font-size:10px;color:#778094}
.analysis-help{font-size:13px;line-height:1.5;color:#687286;margin:14px 0 0}
.movie-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px}
.movie-card{background:#fff;border:1px solid #e2e5eb;border-radius:9px;padding:15px;min-width:0}
.movie-card h3{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.movie-card p{font-size:12px;color:#778094;min-height:30px}
.similarity{font-size:12px;color:#5556c9;font-weight:700}
.message{padding:18px;background:white;border:1px solid #e2e5eb;border-radius:8px;color:#687286}
@media(max-width:850px){.movie-grid{grid-template-columns:repeat(3,1fr)}.analysis-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:550px){main{width:92%;margin-top:22px}.movie{align-items:flex-start;flex-direction:column}.movie-grid{grid-template-columns:repeat(2,1fr)}.analysis-grid{grid-template-columns:1fr 1fr}.panel{padding:16px}}

/* Homepage landing-page styling */
body{background:linear-gradient(135deg,#f5f6ff,#eef0ff,#f8f8fc);min-height:100vh}
header{background:#fff;border-bottom:1px solid #e2e5eb;padding:20px 7%;font-size:18px}
.hero{text-align:center;padding:85px 15px 65px;position:relative}
.badge{display:inline-block;padding:8px 16px;border-radius:30px;background:#e8e9ff;color:#5556c9;font-size:11px;font-weight:bold;letter-spacing:1px;margin-bottom:22px}
.hero h1{font-size:46px;margin:0 0 15px;color:#202344;letter-spacing:-1.5px}
.hero h1 span{color:#5556c9}
.hero p{color:#737b91;font-size:16px;line-height:1.7;max-width:550px;margin:0 auto 30px}
.search{max-width:700px;margin:0 auto 18px;box-shadow:0 8px 25px #5556c91a;border-radius:10px}
.search input{padding:18px;border:1px solid #dfe1ef;border-radius:10px 0 0 10px}
.search button{border-radius:0 10px 10px 0;padding:0 30px}
@media(max-width:550px){.hero{padding:60px 10px}.hero h1{font-size:32px}.hero p{font-size:14px}}


.compact-search{
    max-width:840px;
    margin:0 auto 35px;
    box-shadow:none
}
.compact-search input{padding:16px}


.site-header{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:16px
}
.brand{
    color:#202638;
    text-decoration:none;
    font-weight:700
}
.home-link{
    display:inline-block;
    padding:10px 16px;
    border:1px solid #dedff2;
    border-radius:8px;
    background:#f7f7ff;
    color:#5556c9;
    text-decoration:none;
    font-size:14px;
    font-weight:600;
    white-space:nowrap
}
.home-link:hover{
    background:#e8e9ff
}
@media(max-width:550px){
    .site-header{padding:16px 4%;font-size:14px}
    .home-link{padding:8px 10px;font-size:12px}
}

</style>
</head>
<body>
<header class="site-header">
    <a class="brand" href="/">Movie Recommendation System</a>
    {% if movie %}
    <a class="home-link" href="/">← Back to Home</a>
    {% endif %}
</header>
<main>
{% if not movie %}
<section class="hero">
    <div class="badge">CONTENT-BASED MOVIE DISCOVERY</div>
    <h1>Find Your Next <span>Favourite Movie</span></h1>
    <p>
        Find movies with similar genres, keywords, directors and storylines.
        Search for a title to discover recommendations based on its content.
    </p>
    <form class="search" method="GET" action="/">
        <input type="text" name="movie" value="{{ query }}"
               placeholder="Search for a movie..." autocomplete="off">
        <button type="submit">Search →</button>
    </form>
</section>
{% else %}
<form class="search compact-search" method="GET" action="/">
    <input type="text" name="movie" placeholder="Search for another movie..."
           autocomplete="off">
    <button type="submit">Search →</button>
</form>
{% endif %}

{% if search_results is not none %}
<div class="results">
    {% if search_results %}
        {% for item in search_results %}
        <a href="/movie/{{ item['title'] | urlencode }}">
            <strong>{{ item['title'] }}</strong>
            <small>{{ item.get('genres', 'Genre unavailable') }}</small>
        </a>
        {% endfor %}
    {% else %}
        <div style="padding:14px">No movies found. Try another title.</div>
    {% endif %}
</div>
{% endif %}

{% if movie %}
<section class="panel movie">
    <div class="poster">{{ movie.get('title', 'Movie') }}</div>
    <div>
        <h2>{{ movie.get('title', 'Unknown Movie') }}</h2>
        <div class="meta">{{ movie.get('genres', 'Genre information unavailable') }}</div>
        {% if movie.get('release_year') %}<div class="meta">Release year: {{ movie['release_year'] }}</div>{% endif %}
        {% if movie.get('vote_average') is not none %}<div class="rating">★ {{ "%.1f"|format(movie['vote_average']) }}/10</div>{% endif %}
    </div>
</section>

<section class="panel">
    <h2>Recommended Movies</h2>
    {% if recommendations %}
    <div class="movie-grid">
        {% for item in recommendations %}
        <article class="movie-card">
            <h3>{{ item.get('title', 'Unknown Movie') }}</h3>
            <div class="meta">{{ item.get('genres', 'Genre unavailable') }}</div>
            <p>{% if item.get('vote_average') is not none %}★ {{ "%.1f"|format(item['vote_average']) }}{% endif %} {% if item.get('release_year') %} · {{ item['release_year'] }}{% endif %}</p>
            {% if item.get('similarity') is not none %}<div class="similarity">Similarity: {{ "%.4f"|format(item['similarity']) }}</div>{% endif %}
        </article>
        {% endfor %}
    </div>
    <details>
        <summary>View recommendation TF-IDF vectors and similarity scores</summary>
        <table>
            <thead><tr><th>#</th><th>Movie</th><th>Cosine similarity</th><th>TF-IDF vector</th></tr></thead>
            <tbody>
            {% for item in recommendation_details %}
            <tr>
                <td>{{ item.rank }}</td>
                <td>{{ item.title }}</td>
                <td class="score">{{ "%.4f"|format(item.similarity) }}</td>
                <td><details><summary>View vector</summary><pre>{{ item.tfidf }}</pre></details></td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
    </details>
    {% else %}
    <div class="message">No recommendations were found for this movie.</div>
    {% endif %}
</section>

<section class="panel">
    <h2>Data Mining Analysis</h2>
    <div class="analysis-grid">
        <div class="metric">
            <span>PCA — PC1</span>
            <strong>{% if pca_coordinates %}{{ "%.4f"|format(pca_coordinates[0]) }}{% else %}—{% endif %}</strong>
        </div>
        <div class="metric">
            <span>PCA — PC2</span>
            <strong>{% if pca_coordinates %}{{ "%.4f"|format(pca_coordinates[1]) }}{% else %}—{% endif %}</strong>
        </div>
        <div class="metric"><span>K-Means Cluster</span><strong>{{ cluster if cluster is not none else '—' }}</strong></div>
        <div class="metric"><span>Number of Clusters Formed</span><strong>3</strong></div>
        <div class="metric"><span>Gini Tree — Predicted Cluster</span><strong>{{ gini_prediction if gini_prediction is not none else '—' }}</strong></div>
        <div class="metric"><span>Entropy Tree — Predicted Cluster</span><strong>{{ entropy_prediction if entropy_prediction is not none else '—' }}</strong></div>
        <div class="metric"><span>Input features</span><strong>4 features</strong></div>
    </div>
    <p class="analysis-help">
        <b>How to read these results:</b>
        PC1 and PC2 are the movie's coordinates after reducing four features to two.
        K-Means assigns the movie to a group. The Gini and Entropy decision trees
        independently predict the group label learned during training.
        Cluster numbers are labels, not ratings or rankings.
    </p>
    {% if pca_plot %}
    <img class="plot" src="data:image/png;base64,{{ pca_plot }}" alt="PCA visualization">
    <div class="pca-note">
        <div><b>PC1</b> = 0.612126P + 0.511189R + 0.503003V + 0.333129Y</div>
        <div><b>PC2</b> = 0.200192P − 0.266588R − 0.501446V + 0.798377Y</div>
        <div class="pca-note-vars">
            P = Standardized Popularity &nbsp;&nbsp; R = Standardized Runtime 
           <br>
           V = Standardized Vote Average &nbsp;&nbsp;Y = Standardized Release Year
        </div>
    </div>
    {% endif %}
    <details>
        <summary>View selected movie TF-IDF vector</summary>
        <pre>{{ selected_tfidf }}</pre>
    </details>
    <details>
        <summary>View input features used by PCA, K-Means and Decision Trees</summary>
        <pre>{% if feature_values is not none %}popularity   = {{ movie.get('popularity', 0) }}
runtime      = {{ movie.get('runtime', 0) }}
vote_average = {{ movie.get('vote_average', 0) }}
release_year = {{ movie.get('release_year', 0) }}{% endif %}</pre>
    </details>
</section>
{% endif %}
</main>
</body>
</html>
"""
@app.route("/")
def home():
    query = request.args.get(
        "movie",
        ""
    ).strip()
    search_results = None
    if query:
        try:
            results = search_movie(
                query,
                limit=10
            )
            if results is not None and not results.empty:
                search_results = results.to_dict(
                    orient="records"
                )
            else:
                search_results = []
        except Exception as e:
            print("Search error:", e)
            search_results = []
    return render_template_string(
        HTML,
        query=query,
        search_results=search_results,
        movie=None,
        recommendations=[],
        recommendation_details=[],
        selected_tfidf="",
        pca_coordinates=None,
        cluster=None,
        gini_prediction=None,
        entropy_prediction=None,
        pca_plot=None,
        feature_values=None
    )
@app.route("/movie/<path:title>")
def movie_page(title):
    movie = None
    recommendations = []
    recommendation_details = []
    selected_tfidf = "Unavailable"
    pca_coordinates = None
    cluster = None
    gini_prediction = None
    entropy_prediction = None
    pca_plot = None
    feature_values = None
    try:
        movie_data = get_movie(title)
        if movie_data is not None:
            movie = movie_data.to_dict() if hasattr(movie_data, "to_dict") else movie_data
            results = recommend_movies(title, top_n=10)
            if results is not None and not results.empty:
                if "similarity" in results.columns:
                    results = results.sort_values("similarity", ascending=False, kind="stable")
                recommendations = results.to_dict(orient="records")
            vec = get_movie_vector(title).toarray().ravel()
            selected_tfidf = f"Vector length: {len(vec)}\n\n" + np.array2string(vec, precision=6, separator=", ", threshold=vec.size, max_line_width=140)
            for rank, item in enumerate(recommendations, 1):
                rv = get_movie_vector(item["title"]).toarray().ravel()
                recommendation_details.append({
                    "rank": rank,
                    "title": item.get("title", "Movie"),
                    "similarity": float(item.get("similarity", 0.0)),
                    "tfidf": f"Vector length: {len(rv)}\n\n" + np.array2string(rv, precision=6, separator=", ", threshold=rv.size, max_line_width=140)
                })
            feature_values = np.array([[float(movie.get(c, 0) or 0) for c in CLASS_FEATURES]], dtype=float)
            scaled = cluster_scaler.transform(feature_values)
            pca_coordinates = pca_model.transform(scaled)[0].tolist()
            cluster = int(kmeans_model.predict(scaled)[0])
            gini_prediction = gini_model.predict(feature_values)[0]
            entropy_prediction = entropy_model.predict(feature_values)[0]
            fig, ax = plt.subplots(figsize=(8, 5))
            if PCA_POINTS is not None and len(PCA_POINTS):
                ax.scatter(PCA_POINTS[:, 0], PCA_POINTS[:, 1], s=12, alpha=0.25,
                           label="Other movies")
            ax.scatter(pca_coordinates[0], pca_coordinates[1], s=150, marker="*",
                       edgecolors="black", linewidths=0.6, label="Selected movie", zorder=3)
            ax.annotate(movie.get("title", "Selected Movie"),
                        (pca_coordinates[0], pca_coordinates[1]),
                        xytext=(8, 8), textcoords="offset points", fontsize=9)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("Movie Distribution in PCA Feature Space")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130)
            plt.close(fig)
            pca_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print("Movie analysis error:", e)
    return render_template_string(
        HTML,
        query="",
        search_results=None,
        movie=movie,
        recommendations=recommendations,
        recommendation_details=recommendation_details,
        selected_tfidf=selected_tfidf,
        pca_coordinates=pca_coordinates,
        cluster=cluster,
        gini_prediction=gini_prediction,
        entropy_prediction=entropy_prediction,
        pca_plot=pca_plot,
        feature_values=feature_values
    )
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

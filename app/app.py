import sys
from flask import Flask, request, render_template_string
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.movie_engine import search_movie, get_movie, recommend_movies, get_movie_vector, tfidf as tfidf_model, tfidf_matrix
import pickle
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score
)
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
try:
    with open(MODEL_DIR / "movie_pca_full.pkl", "rb") as f:
        pca_full_model = pickle.load(f)
except FileNotFoundError:
    pca_full_model = None
CLASS_FEATURES = ["popularity", "runtime", "vote_average", "release_year"]

# ============================================================
# NOTEBOOK ANALYTICS
# Recreates the useful evaluation outputs from notebooks 04/05.
# The recommendation engine (TF-IDF + cosine similarity) remains
# separate from the numerical clustering/classification pipeline.
# ============================================================

def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def calculate_entropy(y_values):
    if len(y_values) == 0:
        return 0.0
    counts = pd.Series(y_values).value_counts()
    probabilities = counts / len(y_values)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def calculate_information_gain(parent_y, left_y, right_y):
    if len(parent_y) == 0:
        return 0.0
    left_weight = len(left_y) / len(parent_y)
    right_weight = len(right_y) / len(parent_y)
    child_entropy = (
        left_weight * calculate_entropy(left_y)
        + right_weight * calculate_entropy(right_y)
    )
    return calculate_entropy(parent_y) - child_entropy


def find_best_information_gain_split(X_data, y_data, feature_names):
    best_feature, best_threshold, best_gain = None, None, -1.0
    for feature_index, feature_name in enumerate(feature_names):
        values = X_data[:, feature_index]
        thresholds = np.percentile(values, [20, 40, 60, 80])
        for threshold in thresholds:
            left_mask = values <= threshold
            right_mask = values > threshold
            if not left_mask.any() or not right_mask.any():
                continue
            gain = calculate_information_gain(
                y_data, y_data[left_mask], y_data[right_mask]
            )
            if gain > best_gain:
                best_feature = feature_name
                best_threshold = float(threshold)
                best_gain = float(gain)
    return best_feature, best_threshold, best_gain


PCA_POINTS = None

def build_notebook_analytics():
    """Build dashboard results using the same settings as the notebooks."""
    global PCA_POINTS
    result = {
        "available": False,
        "error": None,
        "pca_plot": None,
        "cluster_plot": None,
        "elbow_plot": None,
        "silhouette_plot": None,
        "gini_cm_plot": None,
        "entropy_cm_plot": None,
        "importance_plot": None,
        "pca_variance": None,
        "pca_variance_rows": [],
        "pca_variance_plot": None,
        "feature_statistics_rows": [],
        "movie_count": None,
        "feature_count": len(CLASS_FEATURES),
        "cluster_count": int(kmeans_model.n_clusters),
        "cluster_metrics": [],
        "elbow_rows": [],
        "silhouette_rows": [],
        "cluster_rows": [],
        "profile_rows": [],
        "deviation_rows": [],
        "sample_movies": [],
        "class_rows": [],
        "classification_rows": [],
        "tree_rows": [],
        "feature_importance_rows": [],
        "class_labels": [],
        "id3": None,
        "test_count": None,
        "train_count": None,
        "preprocessing_rows": [],
        "tfidf_info": {},
    }

    try:
        clustered_path = BASE_DIR / "datasets" / "processed" / "movies_clustered.csv"
        ml_path = BASE_DIR / "datasets" / "processed" / "movies_ml.csv"
        data_path = clustered_path if clustered_path.exists() else ml_path
        if not data_path.exists():
            raise FileNotFoundError(
                "Neither movies_clustered.csv nor movies_ml.csv was found "
                "inside datasets/processed."
            )

        movies_df = pd.read_csv(data_path)
        original_row_count = len(movies_df)
        duplicate_ids = (
            int(movies_df["movieId"].duplicated().sum())
            if "movieId" in movies_df.columns else None
        )
        missing_numeric_before = (
            int(movies_df[CLASS_FEATURES].apply(
                pd.to_numeric, errors="coerce"
            ).isna().sum().sum())
            if all(c in movies_df.columns for c in CLASS_FEATURES) else None
        )
        empty_combined = (
            int(movies_df["combined_features"].fillna("").astype(str).str.strip().eq("").sum())
            if "combined_features" in movies_df.columns else None
        )
        result["preprocessing_rows"] = [
            {"item": "Processed movie records", "value": f"{original_row_count:,}"},
            {"item": "Duplicate movie IDs", "value": str(duplicate_ids) if duplicate_ids is not None else "Column unavailable"},
            {"item": "Missing numerical feature values", "value": str(missing_numeric_before) if missing_numeric_before is not None else "Not checked"},
            {"item": "Empty combined text records", "value": str(empty_combined) if empty_combined is not None else "Column unavailable"},
        ]
        vectorizer_params = tfidf_model.get_params()
        result["tfidf_info"] = {
            "rows": int(tfidf_matrix.shape[0]),
            "columns": int(tfidf_matrix.shape[1]),
            "vocabulary": int(len(tfidf_model.vocabulary_)),
            "max_features": vectorizer_params.get("max_features"),
            "ngram_range": str(vectorizer_params.get("ngram_range")),
            "min_df": vectorizer_params.get("min_df"),
            "stop_words": str(vectorizer_params.get("stop_words")),
        }
        missing = [c for c in CLASS_FEATURES if c not in movies_df.columns]
        if missing:
            raise ValueError(f"Dataset is missing numerical features: {missing}")

        numeric = movies_df[CLASS_FEATURES].apply(pd.to_numeric, errors="coerce")
        valid_mask = numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1)
        movies_df = movies_df.loc[valid_mask].reset_index(drop=True)
        numeric = numeric.loc[valid_mask].reset_index(drop=True)
        if movies_df.empty:
            raise ValueError("No complete numerical movie rows are available.")

        scaled = cluster_scaler.transform(numeric)
        pca_points = pca_model.transform(scaled)
        PCA_POINTS = pca_points
        labels = (
            movies_df["cluster"].to_numpy()
            if "cluster" in movies_df.columns
            else kmeans_model.predict(scaled)
        )
        # If a clustered CSV is present, its labels are the notebook target.
        # Otherwise the saved K-Means model recreates those labels.
        if len(labels) != len(numeric):
            raise ValueError("Cluster labels are not aligned with the movie rows.")

        result["movie_count"] = int(len(movies_df))
        result["cluster_count"] = int(len(np.unique(labels)))
        result["pca_variance"] = float(np.sum(pca_model.explained_variance_ratio_))
        if pca_full_model is not None:
            variance_values = pca_full_model.explained_variance_ratio_
        else:
            variance_values = pca_model.explained_variance_ratio_
        cumulative_values = np.cumsum(variance_values)
        result["pca_variance_rows"] = [
            {
                "component": index + 1,
                "explained": f"{float(value) * 100:.2f}%",
                "cumulative": f"{float(cumulative_values[index]) * 100:.2f}%"
            }
            for index, value in enumerate(variance_values)
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        component_numbers = list(range(1, len(cumulative_values) + 1))
        ax.plot(component_numbers, cumulative_values, marker="o")
        ax.set_xlabel("Number of Principal Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("PCA Cumulative Explained Variance")
        ax.set_xticks(component_numbers)
        ax.grid(alpha=0.25)
        result["pca_variance_plot"] = fig_to_base64(fig)

        stats = numeric.describe().T
        result["feature_statistics_rows"] = [
            {
                "feature": feature,
                "mean": f"{stats.loc[feature, 'mean']:.2f}",
                "std": f"{stats.loc[feature, 'std']:.2f}",
                "min": f"{stats.loc[feature, 'min']:.2f}",
                "max": f"{stats.loc[feature, 'max']:.2f}",
            }
            for feature in CLASS_FEATURES
        ]

        # PCA distribution and cluster-coloured PCA view.
        rng_plot = np.random.default_rng(42)
        plot_idx = rng_plot.choice(
            len(pca_points), size=min(5000, len(pca_points)), replace=False
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(pca_points[plot_idx, 0], pca_points[plot_idx, 1],
                   s=10, alpha=0.28, label="Movies")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Movies in 2D PCA Space")
        ax.grid(alpha=0.2)
        result["pca_plot"] = fig_to_base64(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for cluster_id in sorted(np.unique(labels)):
            mask = labels[plot_idx] == cluster_id
            ax.scatter(
                pca_points[plot_idx[mask], 0],
                pca_points[plot_idx[mask], 1],
                s=11, alpha=0.45, label=f"Cluster {cluster_id}"
            )
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("K-Means Clusters Visualized in 2D PCA Space")
        ax.legend()
        ax.grid(alpha=0.2)
        result["cluster_plot"] = fig_to_base64(fig)

        # Memory-efficient K-Means model selection on a representative sample.
        # Inertia and silhouette values are sample-based estimates.
        k_values = list(range(2, 11))
        inertias, silhouette_values = [], []

        rng = np.random.default_rng(42)
        model_selection_size = min(3000, len(scaled))
        sample_idx = rng.choice(
            len(scaled), size=model_selection_size, replace=False
        )
        X_model_selection = scaled[sample_idx]

        for k in k_values:
            candidate = KMeans(
                n_clusters=k, random_state=42, n_init=5, algorithm="lloyd"
            )
            sample_labels = candidate.fit_predict(X_model_selection)
            inertias.append(float(candidate.inertia_))
            silhouette_values.append(
                float(silhouette_score(X_model_selection, sample_labels))
            )

        result["elbow_rows"] = [
            {"k": k, "inertia": value}
            for k, value in zip(k_values, inertias)
        ]
        result["silhouette_rows"] = [
            {"k": k, "score": value}
            for k, value in zip(k_values, silhouette_values)
        ]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(k_values, inertias, marker="o")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Inertia")
        ax.set_title(f"K-Means Elbow Analysis ({model_selection_size:,}-row sample)")
        ax.grid(alpha=0.25)
        result["elbow_plot"] = fig_to_base64(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(k_values, silhouette_values, marker="o")
        best_k = k_values[int(np.argmax(silhouette_values))]
        ax.axvline(best_k, linestyle="--", alpha=0.6,
                   label=f"Best sampled K = {best_k}")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title(f"Silhouette Analysis ({model_selection_size:,}-row sample)")
        ax.legend()
        ax.grid(alpha=0.25)
        result["silhouette_plot"] = fig_to_base64(fig)
        result["best_silhouette_k"] = int(best_k)
        result["best_silhouette_score"] = float(max(silhouette_values))

        # Sample-based silhouette avoids a full-dataset pairwise-distance calculation.
        # Davies-Bouldin and Calinski-Harabasz are still calculated on all valid rows.
        final_silhouette = float(
            silhouette_score(
                scaled, labels,
                sample_size=min(2000, len(scaled)),
                random_state=42
            )
        )
        db_index = float(davies_bouldin_score(scaled, labels))
        ch_score = float(calinski_harabasz_score(scaled, labels))
        result["cluster_metrics"] = [
            {"name": "Silhouette Score", "value": f"{final_silhouette:.4f}",
             "meaning": "Higher is generally better."},
            {"name": "Davies-Bouldin Index", "value": f"{db_index:.4f}",
             "meaning": "Lower is better."},
            {"name": "Calinski-Harabasz Score", "value": f"{ch_score:.4f}",
             "meaning": "Higher is generally better."},
        ]

        # Cluster size, profiles, deviations and representative movies.
        counts = pd.Series(labels).value_counts().sort_index()
        for cluster_id, count in counts.items():
            result["cluster_rows"].append({
                "cluster": int(cluster_id),
                "count": int(count),
                "percentage": f"{count / len(labels) * 100:.2f}%"
            })

        profile_source = movies_df.copy()
        profile_source["_cluster"] = labels
        profiles = profile_source.groupby("_cluster")[CLASS_FEATURES].mean()
        overall_means = numeric.mean()
        for cluster_id, row in profiles.iterrows():
            result["profile_rows"].append({
                "cluster": int(cluster_id),
                **{feature: f"{row[feature]:.2f}" for feature in CLASS_FEATURES}
            })
            result["deviation_rows"].append({
                "cluster": int(cluster_id),
                **{feature: f"{row[feature] - overall_means[feature]:+.2f}"
                   for feature in CLASS_FEATURES}
            })

        for cluster_id in sorted(np.unique(labels)):
            subset = profile_source[profile_source["_cluster"] == cluster_id]
            columns = [c for c in ["title", "genres", "vote_average",
                                    "popularity", "release_year"]
                       if c in subset.columns]
            if "vote_average" in subset.columns:
                subset = subset.sort_values("vote_average", ascending=False)
            result["sample_movies"].append({
                "cluster": int(cluster_id),
                "movies": subset[columns].head(5).fillna("").to_dict("records")
            })

        # Classification evaluation: same 80/20 stratified split and seed as notebook.
        X = numeric.copy()
        y = pd.Series(labels)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        pred_gini = gini_model.predict(X_test)
        pred_entropy = entropy_model.predict(X_test)
        result["train_count"] = int(len(X_train))
        result["test_count"] = int(len(X_test))
        class_labels = sorted(pd.Series(y).unique().tolist())
        result["class_labels"] = [int(x) for x in class_labels]

        def model_metrics(name, predictions):
            return {
                "model": name,
                "accuracy": f"{accuracy_score(y_test, predictions):.4f}",
                "precision": f"{precision_score(y_test, predictions, average='macro', zero_division=0):.4f}",
                "recall": f"{recall_score(y_test, predictions, average='macro', zero_division=0):.4f}",
                "f1": f"{f1_score(y_test, predictions, average='macro', zero_division=0):.4f}",
            }

        result["classification_rows"] = [
            model_metrics("Gini", pred_gini),
            model_metrics("Entropy", pred_entropy),
        ]

        def confusion_chart(y_true, predictions, title):
            cm = confusion_matrix(y_true, predictions, labels=class_labels)
            fig, ax = plt.subplots(figsize=(5, 4))
            image = ax.imshow(cm)
            ax.set_title(title)
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")
            ax.set_xticks(range(len(class_labels)), class_labels)
            ax.set_yticks(range(len(class_labels)), class_labels)
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    ax.text(col, row, str(cm[row, col]),
                            ha="center", va="center")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            return fig_to_base64(fig)

        result["gini_cm_plot"] = confusion_chart(
            y_test, pred_gini, "Gini Decision Tree Confusion Matrix"
        )
        result["entropy_cm_plot"] = confusion_chart(
            y_test, pred_entropy, "Entropy Decision Tree Confusion Matrix"
        )

        # Tree structure and Gini feature importance.
        result["tree_rows"] = [
            {"model": "Gini", "depth": int(gini_model.get_depth()),
             "leaves": int(gini_model.get_n_leaves())},
            {"model": "Entropy", "depth": int(entropy_model.get_depth()),
             "leaves": int(entropy_model.get_n_leaves())},
        ]
        importances = pd.DataFrame({
            "feature": CLASS_FEATURES,
            "importance": gini_model.feature_importances_
        }).sort_values("importance", ascending=False)
        result["feature_importance_rows"] = [
            {"feature": row.feature, "importance": f"{row.importance:.4f}"}
            for row in importances.itertuples(index=False)
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(importances["feature"], importances["importance"])
        ax.set_ylabel("Importance")
        ax.set_title("Gini Decision Tree — Feature Importance")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        result["importance_plot"] = fig_to_base64(fig)

        # Manual entropy and candidate best root split: same 10,000 sample method
        # and percentile thresholds as the notebook. This is NOT a full ID3 tree.
        sample_size = min(10000, len(X_train))
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(X_train), size=sample_size, replace=False)
        X_id3 = X_train.iloc[sample_indices].to_numpy()
        y_id3 = y_train.iloc[sample_indices].to_numpy()
        best_feature, best_threshold, best_gain = find_best_information_gain_split(
            X_id3, y_id3, CLASS_FEATURES
        )
        result["id3"] = {
            "sample_size": int(sample_size),
            "parent_entropy": f"{calculate_entropy(y_id3):.6f}",
            "best_feature": best_feature,
            "best_threshold": f"{best_threshold:.4f}" if best_threshold is not None else "—",
            "information_gain": f"{best_gain:.6f}" if best_gain >= 0 else "—",
            "note": "Manual Information Gain identifies the best candidate root split; this notebook does not build a complete recursive ID3 tree."
        }

        # Class distribution in the classification target.
        class_counts = pd.Series(labels).value_counts().sort_index()
        result["class_rows"] = [
            {"class": int(label), "count": int(count),
             "percentage": f"{count / len(labels) * 100:.2f}%"}
            for label, count in class_counts.items()
        ]

        result["available"] = True
        return result

    except Exception as exc:
        result["error"] = str(exc)
        print("Notebook analytics unavailable:", exc)
        return result


import os

if os.getenv("ENABLE_ANALYTICS", "true").lower() == "true":
    ANALYTICS = build_notebook_analytics()
else:
    ANALYTICS = {
        "available": False,
        "error": "Extended analytics disabled to conserve server memory."
    }

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
.metric strong{font-size:16px}
details{margin-top:14px;border-top:1px solid #eceef3;padding-top:12px}
summary{cursor:pointer;color:#5052bd;font-weight:600}
pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:280px;overflow:auto;background:#f6f7fb;padding:12px;border-radius:7px;font-size:11px}
table{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px}
th,td{text-align:left;padding:10px;border-bottom:1px solid #e8eaf0;vertical-align:top}
th{color:#687286}
.score{color:#5556c9;font-weight:700}
.plot{width:100%;max-width:760px;display:block;margin:15px auto 0}.analysis-help{font-size:13px;line-height:1.5;color:#687286;margin:14px 0 0}
.movie-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px}
.movie-card{background:#fff;border:1px solid #e2e5eb;border-radius:9px;padding:15px;min-width:0}
.movie-card h3{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.movie-card p{font-size:12px;color:#778094;min-height:30px}
.similarity{font-size:12px;color:#5556c9;font-weight:700}
.message{padding:18px;background:white;border:1px solid #e2e5eb;border-radius:8px;color:#687286}

.dashboard-dropdown{margin:28px 0 35px;border:1px solid #dfe1f2;border-radius:12px;background:#f8f8ff;overflow:hidden}
.dashboard-summary{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:20px 22px;cursor:pointer;list-style:none;color:#292d4a}
.dashboard-summary::-webkit-details-marker{display:none}
.dashboard-summary strong{display:block;font-size:19px;margin-bottom:5px}
.dashboard-summary small{display:block;color:#737b91;font-size:12px;font-weight:400;line-height:1.5}
.dashboard-toggle{flex-shrink:0;padding:9px 13px;border-radius:7px;background:#5556c9;color:#fff;font-size:12px;font-weight:700}
.dashboard-dropdown[open]>.dashboard-summary{border-bottom:1px solid #e2e5eb;background:#f0f0ff}
.dashboard-dropdown[open]>.dashboard-summary .dashboard-toggle{background:#e6e7fa;color:#5556c9}
.dashboard-dropdown>.analysis-section{margin:18px 20px}
.dashboard-dropdown>.analysis-section:last-child{margin-bottom:20px}
.analysis-section{background:#fff;border:1px solid #e2e5eb;border-radius:10px;padding:22px;margin:25px 0}
.analysis-section h2{margin-bottom:8px}
.section-note{font-size:13px;color:#687286;line-height:1.6;margin:5px 0 16px}
.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.chart-box{min-width:0;background:#fafaff;border:1px solid #eceef5;border-radius:8px;padding:12px}
.chart-box img{width:100%;height:auto;display:block}
.stat-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:16px 0}
.stat-card{background:#f6f7fb;border-radius:8px;padding:14px}
.stat-card span{display:block;color:#778094;font-size:12px;margin-bottom:6px}
.stat-card strong{font-size:20px;color:#252b45}
.table-wrap{overflow-x:auto}
.analysis-section table{min-width:520px}
.pill{display:inline-block;padding:4px 8px;border-radius:20px;background:#eeefff;color:#5556c9;font-size:11px;font-weight:700}
.analysis-section details{margin-top:15px}
.analysis-section summary{padding:7px 0}
@media(max-width:750px){.chart-grid{grid-template-columns:1fr}.stat-grid{grid-template-columns:1fr 1fr}}

@media(max-width:850px){.movie-grid{grid-template-columns:repeat(3,1fr)}.analysis-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:550px){main{width:92%;margin-top:22px}.movie{align-items:flex-start;flex-direction:column}.movie-grid{grid-template-columns:repeat(2,1fr)}.analysis-grid{grid-template-columns:1fr 1fr}.panel{padding:16px}}


/* Homepage and navigation styling from the homepage version */

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
        <div class="metric"><span>PCA — PC1</span><strong>{% if pca_coordinates %}{{ "%.4f"|format(pca_coordinates[0]) }}{% else %}—{% endif %}</strong></div>
        <div class="metric"><span>PCA — PC2</span><strong>{% if pca_coordinates %}{{ "%.4f"|format(pca_coordinates[1]) }}{% else %}—{% endif %}</strong></div>
        <div class="metric"><span>K-Means Cluster</span><strong>{{ cluster if cluster is not none else '—' }}</strong></div>
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
    {% if pca_plot %}<img class="plot" src="data:image/png;base64,{{ pca_plot }}" alt="PCA visualization">{% endif %}
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

<details class="dashboard-dropdown">
    <summary class="dashboard-summary">
        <span>
            <strong>Data Mining Dashboard</strong>
            <small>Open to explore preprocessing, TF-IDF, PCA, K-Means, classification and ID3 analysis</small>
        </span>
        <span class="dashboard-toggle">View analysis ↓</span>
    </summary>
{% if analytics and analytics.available %}
<section class="analysis-section">
    <h2>Complete Data Mining Dashboard</h2>
    <p class="section-note">
        These results reproduce the useful analysis from the data preprocessing,
        clustering and classification notebooks. Numerical clustering/classification
        is a separate pipeline from TF-IDF content-based recommendations.
    </p>
    <div class="stat-grid">
        <div class="stat-card"><span>Movies Analyzed</span><strong>{{ analytics.movie_count }}</strong></div>
        <div class="stat-card"><span>Numerical Features</span><strong>{{ analytics.feature_count }}</strong></div>
        <div class="stat-card"><span>K-Means Clusters</span><strong>{{ analytics.cluster_count }}</strong></div>
    </div>
</section>

<section class="analysis-section">
    <h2>1. Data Preparation &amp; Feature Engineering</h2>
    <p class="section-note">
        The notebooks integrate MovieLens and TMDB information, clean and merge movie
        metadata, keywords and credits, and combine genres, keywords, genome tags,
        director and overview into text used by the content-based recommendation model.
    </p>
    <div class="table-wrap"><table>
        <thead><tr><th>Data Check</th><th>Result</th></tr></thead>
        <tbody>{% for row in analytics.preprocessing_rows %}<tr><td>{{ row.item }}</td><td>{{ row.value }}</td></tr>{% endfor %}</tbody>
    </table></div>
    <h3>TF-IDF Configuration</h3>
    <div class="stat-grid">
        <div class="stat-card"><span>TF-IDF Matrix</span><strong>{{ analytics.tfidf_info.rows }} × {{ analytics.tfidf_info.columns }}</strong></div>
        <div class="stat-card"><span>Vocabulary Size</span><strong>{{ analytics.tfidf_info.vocabulary }}</strong></div>
        <div class="stat-card"><span>Maximum Features</span><strong>{{ analytics.tfidf_info.max_features }}</strong></div>
    </div>
    <p class="section-note">
        N-gram range: {{ analytics.tfidf_info.ngram_range }} · Minimum document frequency:
        {{ analytics.tfidf_info.min_df }} · Stop words: {{ analytics.tfidf_info.stop_words }}.
        TF-IDF converts movie text into weighted vectors. Cosine similarity compares
        those vectors to rank similar movies.
    </p>
</section>

<section class="analysis-section">
    <h2>2. PCA — Dimensionality Reduction</h2>
    <p class="section-note">
        The four standardized numerical attributes are projected into two principal
        components for visualization. PCA is used for visualization; K-Means is
        trained on the four standardized features, not on these two plotted coordinates.
    </p>
    <div class="stat-grid">
        <div class="stat-card"><span>Original Dimensions</span><strong>4</strong></div>
        <div class="stat-card"><span>Reduced Dimensions</span><strong>2</strong></div>
        <div class="stat-card"><span>Variance Retained</span><strong>{{ "%.2f"|format(analytics.pca_variance * 100) }}%</strong></div>
    </div>
    <div class="chart-grid">
        {% if analytics.pca_plot %}<div class="chart-box"><img src="data:image/png;base64,{{ analytics.pca_plot }}" alt="PCA movie distribution"></div>{% endif %}
        {% if analytics.cluster_plot %}<div class="chart-box"><img src="data:image/png;base64,{{ analytics.cluster_plot }}" alt="Clusters in PCA space"></div>{% endif %}
        {% if analytics.pca_variance_plot %}<div class="chart-box"><img src="data:image/png;base64,{{ analytics.pca_variance_plot }}" alt="PCA explained variance"></div>{% endif %}
    </div>
    <details>
        <summary>View PCA explained and cumulative variance by component</summary>
        <div class="table-wrap"><table>
            <thead><tr><th>Component</th><th>Explained Variance</th><th>Cumulative Variance</th></tr></thead>
            <tbody>{% for row in analytics.pca_variance_rows %}<tr><td>PC{{ row.component }}</td><td>{{ row.explained }}</td><td>{{ row.cumulative }}</td></tr>{% endfor %}</tbody>
        </table></div>
    </details>
    <details>
        <summary>View numerical feature statistics</summary>
        <div class="table-wrap"><table>
            <thead><tr><th>Feature</th><th>Mean</th><th>Standard Deviation</th><th>Minimum</th><th>Maximum</th></tr></thead>
            <tbody>{% for row in analytics.feature_statistics_rows %}<tr><td>{{ row.feature }}</td><td>{{ row.mean }}</td><td>{{ row.std }}</td><td>{{ row.min }}</td><td>{{ row.max }}</td></tr>{% endfor %}</tbody>
        </table></div>
    </details>
</section>

<section class="analysis-section">
    <h2>3. K-Means Clustering Analysis</h2>
    <p class="section-note">
        K-Means was tested for K = 2 to 10. The notebook selected the final K using
        the highest Silhouette Score from a 10,000-row sample. The saved model currently
        assigns {{ analytics.cluster_count }} clusters.
    </p>
    <div class="stat-grid">
        <div class="stat-card"><span>Best K in Silhouette Search</span><strong>{{ analytics.best_silhouette_k }}</strong></div>
        <div class="stat-card"><span>Best Sampled Silhouette</span><strong>{{ "%.4f"|format(analytics.best_silhouette_score) }}</strong></div>
        <div class="stat-card"><span>Saved Model K</span><strong>{{ analytics.cluster_count }}</strong></div>
    </div>
    <div class="chart-grid">
        {% if analytics.elbow_plot %}<div class="chart-box"><img src="data:image/png;base64,{{ analytics.elbow_plot }}" alt="K-Means elbow curve"></div>{% endif %}
        {% if analytics.silhouette_plot %}<div class="chart-box"><img src="data:image/png;base64,{{ analytics.silhouette_plot }}" alt="Silhouette analysis"></div>{% endif %}
    </div>
    <h3>Final Cluster Quality Metrics</h3>
    <div class="stat-grid">
        {% for metric in analytics.cluster_metrics %}
        <div class="stat-card"><span>{{ metric.name }}</span><strong>{{ metric.value }}</strong><p class="section-note">{{ metric.meaning }}</p></div>
        {% endfor %}
    </div>
    <h3>Cluster Distribution</h3>
    <div class="table-wrap"><table>
        <thead><tr><th>Cluster</th><th>Movie Count</th><th>Percentage</th></tr></thead>
        <tbody>{% for row in analytics.cluster_rows %}<tr><td>{{ row.cluster }}</td><td>{{ row.count }}</td><td>{{ row.percentage }}</td></tr>{% endfor %}</tbody>
    </table></div>
    <details>
        <summary>View cluster profiles (average original feature values)</summary>
        <div class="table-wrap"><table>
            <thead><tr><th>Cluster</th><th>Popularity</th><th>Runtime</th><th>Vote Average</th><th>Release Year</th></tr></thead>
            <tbody>{% for row in analytics.profile_rows %}<tr><td>{{ row.cluster }}</td><td>{{ row.popularity }}</td><td>{{ row.runtime }}</td><td>{{ row.vote_average }}</td><td>{{ row.release_year }}</td></tr>{% endfor %}</tbody>
        </table></div>
    </details>
    <details>
        <summary>View cluster deviation from overall feature averages</summary>
        <div class="table-wrap"><table>
            <thead><tr><th>Cluster</th><th>Popularity</th><th>Runtime</th><th>Vote Average</th><th>Release Year</th></tr></thead>
            <tbody>{% for row in analytics.deviation_rows %}<tr><td>{{ row.cluster }}</td><td>{{ row.popularity }}</td><td>{{ row.runtime }}</td><td>{{ row.vote_average }}</td><td>{{ row.release_year }}</td></tr>{% endfor %}</tbody>
        </table></div>
    </details>
    {% for group in analytics.sample_movies %}
    <details>
        <summary>Top-rated sample movies — Cluster {{ group.cluster }}</summary>
        <div class="table-wrap"><table>
            {% if group.movies %}
            <thead><tr>{% for key in group.movies[0].keys() %}<th>{{ key }}</th>{% endfor %}</tr></thead>
            <tbody>{% for movie_row in group.movies %}<tr>{% for value in movie_row.values() %}<td>{{ value }}</td>{% endfor %}</tr>{% endfor %}</tbody>
            {% endif %}
        </table></div>
    </details>
    {% endfor %}
</section>

<section class="analysis-section">
    <h2>4. Classification — Gini vs Entropy</h2>
    <p class="section-note">
        The target labels are generated by K-Means. Both decision trees learn to
        reproduce those cluster labels using the four original numerical features.
        Evaluation uses the notebook's stratified 80/20 train-test split
        (random_state = 42): {{ analytics.train_count }} training rows and
        {{ analytics.test_count }} test rows.
    </p>
    <div class="table-wrap"><table>
        <thead><tr><th>Model</th><th>Accuracy</th><th>Macro Precision</th><th>Macro Recall</th><th>Macro F1</th></tr></thead>
        <tbody>{% for row in analytics.classification_rows %}<tr><td><b>{{ row.model }}</b></td><td>{{ row.accuracy }}</td><td>{{ row.precision }}</td><td>{{ row.recall }}</td><td>{{ row.f1 }}</td></tr>{% endfor %}</tbody>
    </table></div>
    <div class="chart-grid">
        {% if analytics.gini_cm_plot %}<div class="chart-box"><img src="data:image/png;base64,{{ analytics.gini_cm_plot }}" alt="Gini confusion matrix"></div>{% endif %}
        {% if analytics.entropy_cm_plot %}<div class="chart-box"><img src="data:image/png;base64,{{ analytics.entropy_cm_plot }}" alt="Entropy confusion matrix"></div>{% endif %}
    </div>
    <p class="section-note">
        Confusion matrix rows represent actual cluster labels; columns represent
        predicted cluster labels. These scores measure agreement with K-Means labels,
        not independent ground-truth movie categories.
    </p>
    <h3>Decision Tree Structure</h3>
    <div class="table-wrap"><table>
        <thead><tr><th>Tree</th><th>Depth</th><th>Number of Leaves</th></tr></thead>
        <tbody>{% for row in analytics.tree_rows %}<tr><td>{{ row.model }}</td><td>{{ row.depth }}</td><td>{{ row.leaves }}</td></tr>{% endfor %}</tbody>
    </table></div>
    <h3>Gini Tree — Feature Importance</h3>
    {% if analytics.importance_plot %}<div class="chart-box"><img src="data:image/png;base64,{{ analytics.importance_plot }}" alt="Feature importance"></div>{% endif %}
    <div class="table-wrap"><table>
        <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
        <tbody>{% for row in analytics.feature_importance_rows %}<tr><td>{{ row.feature }}</td><td>{{ row.importance }}</td></tr>{% endfor %}</tbody>
    </table></div>
</section>

<section class="analysis-section">
    <h2>5. Manual Entropy &amp; ID3 Information Gain</h2>
    <p class="section-note">
        The notebook manually calculates entropy and tests percentile-based thresholds
        (20th, 40th, 60th and 80th) to identify the best candidate root split.
    </p>
    <div class="stat-grid">
        <div class="stat-card"><span>ID3 Demonstration Sample</span><strong>{{ analytics.id3.sample_size }}</strong></div>
        <div class="stat-card"><span>Parent Entropy</span><strong>{{ analytics.id3.parent_entropy }}</strong></div>
        <div class="stat-card"><span>Best Information Gain</span><strong>{{ analytics.id3.information_gain }}</strong></div>
    </div>
    <div class="table-wrap"><table>
        <thead><tr><th>Best Candidate Feature</th><th>Threshold</th><th>Information Gain</th></tr></thead>
        <tbody><tr><td>{{ analytics.id3.best_feature }}</td><td>{{ analytics.id3.best_threshold }}</td><td>{{ analytics.id3.information_gain }}</td></tr></tbody>
    </table></div>
    <p class="section-note"><b>Important:</b> {{ analytics.id3.note }}</p>
</section>
{% else %}
<section class="analysis-section">
    <h2>Additional Data Mining Analysis</h2>
    <p class="section-note">
        The extended notebook analysis could not be loaded: {{ analytics.error or "Unknown error" }}.
        Check that the processed dataset and saved model files are present in the project folders.
    </p>
</section>
{% endif %}
</details>

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
        feature_values=None,
        analytics=ANALYTICS
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
            # Keep TF-IDF vectors sparse and display only the highest-weight
            # non-zero features instead of expanding dense 10,000-value arrays.
            vec = get_movie_vector(title).tocsr()
            top_terms = np.argsort(vec.data)[::-1][:40]
            selected_tfidf = (
                f"Vector length: {vec.shape[1]}\\n"
                f"Non-zero terms: {vec.nnz}\\n"
                "Top TF-IDF feature indices and weights:\\n"
                + "\\n".join(
                    f"{int(vec.indices[i])}: {float(vec.data[i]):.6f}"
                    for i in top_terms
                )
            )
            for rank, item in enumerate(recommendations, 1):
                rv = get_movie_vector(item["title"]).tocsr()
                rv_top = np.argsort(rv.data)[::-1][:20]
                vector_text = (
                    f"Vector length: {rv.shape[1]}\\n"
                    f"Non-zero terms: {rv.nnz}\\n"
                    "Top TF-IDF feature indices and weights:\\n"
                    + "\\n".join(
                        f"{int(rv.indices[i])}: {float(rv.data[i]):.6f}"
                        for i in rv_top
                    )
                )
                recommendation_details.append({
                    "rank": rank,
                    "title": item.get("title", "Movie"),
                    "similarity": float(item.get("similarity", 0.0)),
                    "tfidf": vector_text
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
        feature_values=feature_values,
        analytics=ANALYTICS
    )
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )

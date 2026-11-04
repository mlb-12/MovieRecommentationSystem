"""
==========================================================
Movie Recommendation Engine
==========================================================

Project:
Movie Recommendation System

Module:
Movie Feature Engineering & Content-Based Recommendation

Description:
Loads the processed movie dataset and TF-IDF models
and provides:

1. Movie search
2. Movie information lookup
3. TF-IDF feature-vector access
4. Content-based movie recommendations

Recommendation Method:
TF-IDF + Cosine Similarity

Public API
----------
search_movie(keyword, limit=20)
recommend_movies(movie_title, top_n=10)
recommend_movies_by_id(movie_id, top_n=10)
get_movie(movie_title)
get_movie_by_id(movie_id)
get_movie_vector(movie_title)

==========================================================
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


# ==========================================================
# Module Info
# ==========================================================

MODULE_VERSION = "2.0.0"


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "movies_ml.csv"
)

MODEL_DIR = BASE_DIR / "models"

TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MATRIX_PATH = MODEL_DIR / "movie_tfidf_matrix.pkl"
INDEX_PATH = MODEL_DIR / "movie_indices.pkl"


# ==========================================================
# Verify Required Files
# ==========================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

for model_file in [
    TFIDF_PATH,
    MATRIX_PATH,
    INDEX_PATH
]:

    if not model_file.exists():
        raise FileNotFoundError(
            f"Missing model file:\n{model_file}"
        )


# ==========================================================
# Load Dataset
# ==========================================================

movies = pd.read_csv(
    DATASET_PATH
)


# ==========================================================
# Load TF-IDF Models
# ==========================================================

with open(
    TFIDF_PATH,
    "rb"
) as f:

    tfidf = pickle.load(f)


with open(
    MATRIX_PATH,
    "rb"
) as f:

    tfidf_matrix = pickle.load(f)


with open(
    INDEX_PATH,
    "rb"
) as f:

    movie_indices = pickle.load(f)


# ==========================================================
# Validate Dataset / TF-IDF Alignment
# ==========================================================

if len(movies) != tfidf_matrix.shape[0]:

    raise ValueError(
        "Dataset and TF-IDF matrix are not aligned.\n"
        f"Dataset rows: {len(movies)}\n"
        f"TF-IDF rows: {tfidf_matrix.shape[0]}"
    )


# ==========================================================
# Constants
# ==========================================================

SEARCH_COLUMNS = [
    "movieId",
    "title",
    "genres",
    "vote_average",
    "release_year"
]


RECOMMEND_COLUMNS = [
    "movieId",
    "title",
    "genres",
    "vote_average",
    "release_year",
    "similarity"
]


# ==========================================================
# Search
# ==========================================================

def search_movie(keyword, limit=20):
    """
    Search movies by title.

    Parameters
    ----------
    keyword : str
        Text to search for in movie titles.

    limit : int
        Maximum number of results.

    Returns
    -------
    pandas.DataFrame
        Matching movies.
    """

    if not isinstance(keyword, str):
        raise TypeError(
            "keyword must be a string."
        )

    if limit <= 0:
        raise ValueError(
            "limit must be greater than 0."
        )

    keyword = keyword.strip()

    if not keyword:
        return pd.DataFrame(
            columns=SEARCH_COLUMNS
        )

    results = movies[
        movies["title"].str.contains(
            keyword,
            case=False,
            na=False,
            regex=False
        )
    ]

    return results[
        SEARCH_COLUMNS
    ].head(limit)


# ==========================================================
# Movie Information
# ==========================================================

def get_movie(movie_title):
    """
    Return movie metadata by exact title.

    Parameters
    ----------
    movie_title : str
        Exact movie title.

    Returns
    -------
    dict or None
        Movie information.
    """

    movie = movies[
        movies["title"] == movie_title
    ]

    if movie.empty:
        return None

    return movie.iloc[0].to_dict()


# ==========================================================
# Movie Information by ID
# ==========================================================

def get_movie_by_id(movie_id):
    """
    Return movie metadata using MovieLens movieId.

    Parameters
    ----------
    movie_id : int
        MovieLens movie ID.

    Returns
    -------
    dict or None
        Movie information.
    """

    movie = movies[
        movies["movieId"] == movie_id
    ]

    if movie.empty:
        return None

    return movie.iloc[0].to_dict()


# ==========================================================
# TF-IDF Feature Vector
# ==========================================================

def get_movie_vector(movie_title):
    """
    Return the TF-IDF vector for a movie.

    Parameters
    ----------
    movie_title : str
        Exact movie title.

    Returns
    -------
    scipy sparse matrix
        TF-IDF vector.
    """

    if movie_title not in movie_indices:

        raise ValueError(
            f"Movie '{movie_title}' not found."
        )

    idx = movie_indices[
        movie_title
    ]

    return tfidf_matrix[idx]


# ==========================================================
# Content-Based Recommendation
# ==========================================================

def recommend_movies(movie_title, top_n=10):
    """
    Recommend movies similar to the given movie.

    The recommendation is based on:

        Movie Text
            ↓
          TF-IDF
            ↓
      Cosine Similarity
            ↓
      Similar Movies

    Parameters
    ----------
    movie_title : str
        Exact title of the input movie.

    top_n : int
        Number of recommendations.

    Returns
    -------
    pandas.DataFrame
        Recommended movies with similarity scores.
    """

    if movie_title not in movie_indices:

        raise ValueError(
            f"Movie '{movie_title}' not found."
        )

    if top_n <= 0:

        raise ValueError(
            "top_n must be greater than 0."
        )

    idx = movie_indices[
        movie_title
    ]

    # ------------------------------------------------------
    # Calculate cosine similarity
    # ------------------------------------------------------

    similarity_scores = linear_kernel(
        tfidf_matrix[idx:idx + 1],
        tfidf_matrix
    ).flatten()

    # ------------------------------------------------------
    # Sort from highest similarity to lowest
    # ------------------------------------------------------

    similarity_indices = (
        similarity_scores.argsort()[::-1]
    )

    # ------------------------------------------------------
    # Remove the input movie itself
    # ------------------------------------------------------

    similarity_indices = [
        i
        for i in similarity_indices
        if i != idx
    ]

    similarity_indices = (
        similarity_indices[:top_n]
    )

    # ------------------------------------------------------
    # Build recommendation table
    # ------------------------------------------------------

    recommendations = movies.iloc[
        similarity_indices
    ].copy()

    recommendations[
        "similarity"
    ] = similarity_scores[
        similarity_indices
    ]

    return recommendations[
        RECOMMEND_COLUMNS
    ].reset_index(drop=True)


# ==========================================================
# Recommendation by Movie ID
# ==========================================================

def recommend_movies_by_id(
    movie_id,
    top_n=10
):
    """
    Recommend movies using MovieLens movieId.
    """

    movie = get_movie_by_id(
        movie_id
    )

    if movie is None:

        raise ValueError(
            f"Movie ID {movie_id} not found."
        )

    return recommend_movies(
        movie["title"],
        top_n
    )


# ==========================================================
# Self Test
# ==========================================================

if __name__ == "__main__":

    TEST_MOVIE = "Interstellar (2014)"

    print("=" * 70)
    print("MOVIE RECOMMENDATION ENGINE")
    print(f"Version: {MODULE_VERSION}")
    print("=" * 70)

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    print("\nDATASET")
    print("-" * 70)

    print("Dataset path:")
    print(DATASET_PATH)

    print("Number of movies:")
    print(len(movies))

    # ------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------

    print("\nTF-IDF")
    print("-" * 70)

    print(
        "TF-IDF matrix shape:",
        tfidf_matrix.shape
    )

    print(
        "Vocabulary size:",
        len(tfidf.vocabulary_)
    )

    # ------------------------------------------------------
    # Search
    # ------------------------------------------------------

    print("\nSEARCH TEST")
    print("-" * 70)

    print(
        search_movie(
            "interstellar"
        )
    )

    # ------------------------------------------------------
    # Movie Information
    # ------------------------------------------------------

    print("\nMOVIE INFORMATION")
    print("-" * 70)

    movie_info = get_movie(
        TEST_MOVIE
    )

    if movie_info is not None:

        print(
            "Title:",
            movie_info["title"]
        )

        print(
            "Genres:",
            movie_info["genres"]
        )

        print(
            "Rating:",
            movie_info["vote_average"]
        )

        print(
            "Release year:",
            movie_info["release_year"]
        )

    # ------------------------------------------------------
    # Vector
    # ------------------------------------------------------

    print("\nMOVIE VECTOR")
    print("-" * 70)

    vector = get_movie_vector(
        TEST_MOVIE
    )

    print(
        "Vector shape:",
        vector.shape
    )

    # ------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------

    print("\nRECOMMENDATIONS")
    print("-" * 70)

    recommendations = recommend_movies(
        TEST_MOVIE,
        top_n=10
    )

    print(
        recommendations.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # ID Recommendation Test
    # ------------------------------------------------------

    print("\nMOVIE ID RECOMMENDATION TEST")
    print("-" * 70)

    test_movie = get_movie(
        TEST_MOVIE
    )

    if test_movie is not None:

        test_movie_id = test_movie[
            "movieId"
        ]

        print(
            "Movie ID:",
            test_movie_id
        )

        print(
            recommend_movies_by_id(
                test_movie_id,
                top_n=5
            ).to_string(
                index=False
            )
        )

    print("\n" + "=" * 70)
    print("ENGINE SELF TEST COMPLETE")
    print("=" * 70)
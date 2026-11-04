# 🎬 Movie Recommendation System

An intelligent Movie Recommendation System that uses Data Mining and Machine Learning techniques to analyze movie data and recommend similar movies based on their content and characteristics.

The project combines content-based filtering, dimensionality reduction, clustering, and classification to provide movie recommendations and demonstrate different data mining techniques.

## 📌 Project Overview

The Movie Recommendation System allows users to:

* Search for movies by title.
* View movie information, including genres, release year, and ratings.
* Receive recommendations based on movie similarity.
* Explore TF-IDF feature vectors and cosine similarity scores.
* Visualize movie data using Principal Component Analysis (PCA).
* Analyze movie clusters using the K-Means clustering algorithm.
* Compare classification results using Decision Trees with Gini and Entropy criteria.

## 🛠️ Technologies Used

| Category                 | Technologies      |
| ------------------------ | ----------------- |
| Programming Language     | Python            |
| Web Framework            | Flask             |
| Data Processing          | Pandas, NumPy     |
| Machine Learning         | Scikit-learn      |
| Feature Extraction       | TF-IDF            |
| Similarity Measurement   | Cosine Similarity |
| Dimensionality Reduction | PCA               |
| Clustering               | K-Means           |
| Classification           | Decision Tree     |
| Data Visualization       | Matplotlib        |

## 🧠 Data Mining Techniques

### 1. TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF converts movie-related textual information into numerical feature vectors. These vectors represent the textual characteristics of movies and are used to identify similarities.

### 2. Cosine Similarity

Cosine Similarity measures the similarity between movie feature vectors. Movies with higher similarity scores are considered more relevant recommendations.

### 3. Principal Component Analysis (PCA)

PCA reduces the dimensionality of movie features into two principal components, allowing movie data to be represented and visualized in a lower-dimensional space.

### 4. K-Means Clustering

K-Means groups movies into clusters based on their numerical features. It helps identify groups of movies with similar characteristics.

### 5. Decision Tree Classification

Two Decision Tree classifiers are implemented:

* Gini Index
* Information Gain (Entropy)

Both classifiers are used to predict movie cluster labels and compare classification approaches.

## 📂 Project Structure

```text
DataMining/
│
├── app/
│   ├── app.py
│   └── README.md
│
├── datasets/
│   └── processed/
│       └── movies_ml.csv
│
├── models/
│   ├── classification_features.pkl
│   ├── clustering_features.pkl
│   ├── movie_cluster_scaler.pkl
│   ├── movie_decision_tree_entropy.pkl
│   ├── movie_decision_tree_gini.pkl
│   ├── movie_indices.pkl
│   ├── movie_kmeans.pkl
│   ├── movie_order.pkl
│   ├── movie_pca.pkl
│   ├── movie_tfidf_matrix.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_inspect_datasets.ipynb
│   ├── 02_data_cleaning_and_merging.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 09_data_reduction_clustering.ipynb
│   └── 10_classification.ipynb
│
├── src/
│   └── movie_engine.py
│
├── tests/
│   └── test_movie_engine.py
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

* Python 3.10 or later
* Git
* pip

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd DataMining
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

The current application uses Flask.

Navigate to the application directory:

```bash
cd app
```

Run:

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

## 🚀 Deployment

The project can be prepared for deployment using cloud hosting platforms.

The current application is Flask-based. To deploy through Streamlit Community Cloud, the Flask interface must first be converted into a Streamlit application.

## 📊 Recommendation Workflow

```text
Movie Dataset
      |
      v
Data Preprocessing
      |
      v
Feature Engineering
      |
      v
TF-IDF Feature Extraction
      |
      v
Cosine Similarity
      |
      v
Movie Recommendations
```

Additional Data Mining Analysis:

```text
Processed Movie Features
        |
        v
  Feature Scaling
        |
        v
   +----+----------------+
   |                     |
   v                     v
  PCA                K-Means
   |                     |
   v                     v
Visualization       Movie Clusters
                         |
                         v
                 Decision Tree
                   /       \
                Gini     Entropy
```

## 🎯 Project Objectives

* Apply data mining techniques to movie datasets.
* Develop a content-based movie recommendation system.
* Extract meaningful features using TF-IDF.
* Measure movie similarity using Cosine Similarity.
* Explore dimensionality reduction using PCA.
* Perform clustering using K-Means.
* Implement and compare Decision Tree classifiers.
* Integrate data mining algorithms into an interactive web application.

## 👥 Conclusion

This project demonstrates the application of Data Mining and Machine Learning techniques in building a movie recommendation system. By combining content-based filtering, dimensionality reduction, clustering, and classification, the system provides movie recommendations while allowing users to explore the underlying data mining processes.

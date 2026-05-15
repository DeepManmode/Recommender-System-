## Introduction

This project is an end-to-end **Content-Based Movie Recommendation System** that suggests movies similar to a user's selection. By leveraging Natural Language Processing (NLP) techniques and cosine similarity, the system analyzes movie metadata — including genres, cast, crew, keywords, and plot overview — to identify and return the most relevant movie suggestions.

The project is deployed as an interactive **Streamlit web application**, making it accessible and intuitive for any user to explore personalized recommendations without needing any technical background.

---

## Background

In the age of streaming platforms, users face an overwhelming volume of content. Recommendation engines have become a core component of platforms like Netflix, Amazon Prime, and Spotify to combat decision fatigue and improve user engagement.

There are three primary types of recommendation systems:

| Type | Description |
|---|---|
| **Content-Based Filtering** | Recommends items similar to what the user has already liked, based on item features |
| **Collaborative Filtering** | Recommends based on patterns from similar users |
| **Hybrid** | Combines both approaches for improved accuracy |

This project implements **Content-Based Filtering**, which requires no historical user interaction data, making it ideal for cold-start scenarios where user rating history is unavailable.

The dataset used is the **TMDB 5000 Movie Dataset** from Kaggle, which contains metadata for 5,000 movies including budget, genres, cast, crew, production companies, and more.

---

## Project Objectives

1. **Build a content-based recommendation engine** that returns the top 5 similar movies for any given input.
2. **Apply NLP techniques** (text vectorization and stemming) to transform movie metadata into a comparable numerical format.
3. **Measure similarity** between movies using cosine similarity on vectorized feature tags.
4. **Interactive web Interface** using Streamlit users can interact with the model .
5. **Demonstrate a complete ML project lifecycle** — from raw data ingestion to a functional deployed product.

---

## Tools & Technology

| Category | Tools Used |
|---|---|
| **Language** | Python 3.8+ |
| **Data Manipulation** | Pandas, NumPy |
| **NLP & Feature Engineering** | NLTK (PorterStemmer), ast (literal_eval) |
| **Machine Learning** | Scikit-learn (CountVectorizer, cosine_similarity) |
| **Model Serialization** | Pickle |
| **Web Application** | Streamlit |
| **Notebook Environment** | Jupyter Notebook |
| **Dataset** | TMDB 5000 Movie Dataset (Kaggle) |
| **Version Control** | Git, GitHub (with Git LFS for large files) |

---

## Workflow Overview

```
Raw Data (TMDB CSVs)
        │
        ▼
   Data Merging
 (movies + credits)
        │
        ▼
  Feature Extraction
(genres, keywords, cast,
  director, overview)
        │
        ▼
  Text Preprocessing
(lowercasing, stemming,
   space removal)
        │
        ▼
  Tag Construction
(combine all features
  into one text blob)
        │
        ▼
  Vectorization
(CountVectorizer →
 Bag-of-Words matrix)
        │
        ▼
  Cosine Similarity
  Matrix (5000×5000)
        │
        ▼
  Recommendation
  Function (Top 5)
        │
        ▼
  Streamlit App
  (Interactive UI)
```

**Step-by-step breakdown:**

1. **Data Ingestion** — Load `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` and merge on the movie title.
2. **Feature Selection** — Extract relevant columns: `genres`, `keywords`, `cast`, `crew`, `overview`, and `title`.
3. **Data Parsing** — Use `ast.literal_eval` to parse JSON-like string columns into Python lists.
4. **Feature Engineering** — Extract the top 3 cast members, the director from the crew column, and flatten all features into a single `tags` string per movie.
5. **Text Preprocessing** — Convert tags to lowercase, apply stemming with NLTK's `PorterStemmer` to normalize word forms.
6. **Vectorization** — Apply `CountVectorizer` with `max_features=5000` and `stop_words='english'` to convert tags into a Bag-of-Words matrix.
7. **Similarity Computation** — Calculate the cosine similarity matrix across all 5,000 movies.
8. **Recommendation Logic** — Given a movie title, fetch its index, sort by similarity score, and return the top 5 results (excluding itself).
9. **Serialization** — Export the processed movie dictionary and similarity matrix as `.pkl` files for use in the Streamlit app.
10. **Web Deployment** — Build and run an interactive Streamlit frontend that accepts user input and displays recommendations.

---

## Key Findings

- **Genre + Keywords + Cast + Director** combined as a unified "tags" feature produced better recommendations than using any single feature alone.
- **Stemming** (e.g., "running" → "run") significantly reduced vocabulary size and improved similarity matching across semantically related movies.
- **Cosine Similarity** outperforms Euclidean distance for text-based sparse vectors because it is scale-invariant — it measures the angle between vectors, not their magnitude.
- Movies with **richer metadata** (detailed overviews, multiple keywords, well-known cast) received more accurate recommendations than low-metadata entries.
- **CountVectorizer** with `max_features=5000` provided a strong balance between vocabulary coverage and computational efficiency.
- The top 3 cast members and the director were the most discriminative features for identifying genre-style similarities between films.

---

## What I Learned

- **End-to-End ML Project Thinking** — Understood how to take a problem from raw messy data all the way through to a deployed, interactive product.
- **NLP Fundamentals in Practice** — Gained hands-on experience with tokenization, stemming, stop-word removal, and the Bag-of-Words model using CountVectorizer.
- **Feature Engineering is Everything** — Realized that the quality of the combined `tags` feature had a greater impact on recommendation quality than the choice of algorithm.
- **Cosine Similarity Intuition** — Developed a strong understanding of why cosine similarity is preferred for high-dimensional sparse text vectors.
- **Streamlit for Rapid Prototyping** — Learned how to quickly build a functional, shareable ML web app without requiring front-end engineering expertise.
- **Pickle for Model Persistence** — Understood how to serialize Python objects to avoid recomputing expensive operations like the similarity matrix on every app load.
- **Git LFS** — Learned how to handle large binary files (CSVs, `.pkl`) in GitHub using Git Large File Storage.

---

## Future Improvements

| Improvement | Description |
|---|---|
| **Hybrid Filtering** | Combine content-based and collaborative filtering for better personalization |
| **TMDB API Integration** | Fetch live movie posters and trailers to enrich the UI experience |
| **TF-IDF Vectorization** | Replace CountVectorizer with TF-IDF to weight rare but meaningful terms more effectively |
| **Transformer Embeddings** | Use sentence-transformers (BERT-based) for richer semantic understanding of movie plots |
| **User Feedback Loop** | Add a thumbs-up/down system to capture user preference and retrain the model |
| **Scalability** | Migrate similarity computation to an approximate nearest-neighbor library (e.g., FAISS) for handling larger datasets |
| **Cloud Deployment** | Deploy on Streamlit Community Cloud, Heroku, or AWS for public access |
| **Multi-language Support** | Extend the dataset to include non-English films for a global audience |

---

## Project Structure

```
Recommender-System/
│
├── Notebook/
│   └── movie-recommender-system.ipynb   # EDA, feature engineering, model building
│
├── Frontend/
│   └── app.py                           # Streamlit web application
│
├── tmdb_5000_movies.csv                 # TMDB movie metadata dataset
├── tmdb_5000_credits.csv                # Cast and crew dataset
├── .gitattributes                       # Git LFS configuration
└── README.md
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/DeepManmode/Recommender-System-.git
cd Recommender-System-
```

**2. Install required libraries**
```bash
pip install pandas numpy scikit-learn nltk streamlit
```

**3. Run the Jupyter Notebook**

Open `Notebook/movie-recommender-system.ipynb` and run all cells. This will generate:
- `movie_dict.pkl` — the processed movie data
- `similarity.pkl` — the precomputed cosine similarity matrix

**4. Launch the Streamlit app**
```bash
streamlit run Frontend/app.py
```

**5. Open your browser** at `http://localhost:8501` and start exploring recommendations!

---

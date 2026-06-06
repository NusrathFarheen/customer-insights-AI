import os
import json
import pandas as pd
import numpy as np
import joblib

# Paths
MODELS_DIR = "models"
SENTIMENT_PATH = os.path.join(MODELS_DIR, "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.joblib")
SAMPLE_CSV_PATH = os.path.join(MODELS_DIR, "sample_reviews.csv")

OUTPUT_WEIGHTS = os.path.join("..", "frontend", "src", "model_weights.json")
OUTPUT_REVIEWS = os.path.join("..", "frontend", "src", "sample_reviews.json")

def export():
    if not (os.path.exists(SENTIMENT_PATH) and os.path.exists(VECTORIZER_PATH) and 
            os.path.exists(KMEANS_PATH) and os.path.exists(METADATA_PATH)):
        print("Models not found. Please train models first.")
        return

    # Load joblib files
    sentiment_model = joblib.load(SENTIMENT_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    kmeans_model = joblib.load(KMEANS_PATH)
    model_metadata = joblib.load(METADATA_PATH)

    # 1. EXPORT MODEL WEIGHTS
    weights = {
        # Vectorizer
        "vocabulary": {word: int(idx) for word, idx in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.tolist(),
        "feature_names": vectorizer.feature_names_,
        
        # Naive Bayes
        "classes": sentiment_model.classes.tolist(),
        "class_log_prior": sentiment_model.class_log_prior_.tolist(),
        "feature_log_prob": sentiment_model.feature_log_prob_.tolist(),
        
        # K-Means
        "cluster_centers": kmeans_model.cluster_centers_.tolist(),
        "cluster_topic_map": {str(k): str(v) for k, v in model_metadata["cluster_topic_map"].items()},
        "cluster_top_words": {str(k): list(v) for k, v in model_metadata["cluster_top_words"].items()},
        "accuracy": float(model_metadata.get("accuracy", 0.98))
    }

    with open(OUTPUT_WEIGHTS, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    print(f"Model weights successfully exported to {OUTPUT_WEIGHTS}")

    # 2. PRE-PROCESS AND EXPORT SAMPLE REVIEWS
    # This prevents the frontend from needing to run predictions on 1000 items at startup
    df = pd.read_csv(SAMPLE_CSV_PATH)
    
    # Run predictions locally to create complete items
    df["review_text"] = df["review_text"].fillna("").astype(str)
    
    # We preprocess and vectorize to generate predictions
    # Standard clean function
    def preprocess_text(text: str) -> str:
        text = text.lower()
        import re
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
        
    clean_texts = df["review_text"].apply(preprocess_text).tolist()
    X_vec = vectorizer.transform(clean_texts)
    
    # Run predictions
    sentiments = sentiment_model.predict(X_vec)
    clusters = kmeans_model.predict(X_vec)
    
    # Terms helper
    terms = vectorizer.get_feature_names_out()
    
    reviews_list = []
    for idx, row in df.iterrows():
        # Extract top 5 tfidf words
        vec_1d = X_vec[idx]
        non_zero_indices = np.nonzero(vec_1d)[0]
        tuples = [(i, vec_1d[i]) for i in non_zero_indices]
        sorted_items = sorted(tuples, key=lambda x: x[1], reverse=True)
        key_terms = [terms[i] for i, score in sorted_items[:5]]
        
        cluster_idx = clusters[idx]
        topic = model_metadata["cluster_topic_map"].get(cluster_idx, "General")
        
        reviews_list.append({
            "review_id": str(row["review_id"]),
            "category": str(row["category"]),
            "rating": int(row["rating"]),
            "review_text": str(row["review_text"]),
            "sentiment": str(sentiments[idx]),
            "topic": str(topic),
            "helpful_votes": int(row["helpful_votes"]),
            "date": str(row["date"]),
            "key_terms": key_terms
        })

    with open(OUTPUT_REVIEWS, "w", encoding="utf-8") as f:
        json.dump(reviews_list, f, indent=2)
    print(f"Pre-calculated reviews successfully exported to {OUTPUT_REVIEWS}")

if __name__ == "__main__":
    export()

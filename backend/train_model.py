import os
import random
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

# Import custom ML components
from pure_ml import PureTfidfVectorizer, PureNaiveBayes, PureKMeans

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Create models directory
os.makedirs("models", exist_ok=True)

# 1. GENERATE A REALISTIC E-COMMERCE REVIEW DATASET
print("Generating realistic customer feedback dataset...")

categories = ["Electronics", "Clothing & Fashion", "Home & Kitchen", "Books & Media"]

review_templates = {
    "positive": [
        "Absolutely love this product! The quality is outstanding and exceeded my expectations.",
        "Best purchase I have made in a long time. Highly recommend to everyone.",
        "Excellent product. It works perfectly and the design is very sleek and modern.",
        "Very satisfied with this purchase. High quality material and works like a charm.",
        "Great value for money. It does exactly what it says and feels premium.",
        "Super fast shipping! The item arrived in perfect condition and is top notch.",
        "Five stars! The customer service was also very helpful and responsive.",
        "Very easy to set up and use. The build quality feels sturdy and durable.",
        "I was skeptical at first, but this is brilliant. Definitely worth the price.",
        "Perfect fit and works beautifully. Will definitely buy from this brand again."
    ],
    "neutral": [
        "It is decent for the price. Not amazing, but does the job fine.",
        "Average product. It works as described, but the build quality could be better.",
        "It is okay. Shipping was a bit slow, but the product is acceptable.",
        "Neutral opinion. It has some good features but also a few design flaws.",
        "The product works, but the customer service was not very helpful.",
        "Okay product, but I feel like it is slightly overpriced for what it offers.",
        "Not bad, not great. Standard product that meets basic expectations.",
        "It performs okay, but there are better alternatives on the market.",
        "The size is smaller than expected, but the quality is fine otherwise.",
        "Delivery was quick, but the product packaging was a bit damaged."
    ],
    "negative": [
        "Terrible quality. It broke within the first day of use. Do not buy!",
        "Very disappointed. The product looks cheap and does not work properly.",
        "Waste of money. Extremely slow shipping and the item is defective.",
        "Worst experience ever. The product didn't match the description at all.",
        "Poor build quality. The plastic feels cheap and it makes a weird noise.",
        "The customer service was awful, and the product arrived damaged.",
        "I would not recommend this. It stopped working after a week of light use.",
        "Very overpriced for such poor quality. I am returning it immediately.",
        "Frustrating to use. The instructions are unclear and it is very fragile.",
        "Horrible purchase. Avoid this seller and product. Save your money."
    ]
}

topic_keywords = {
    "shipping": ["shipping", "delivery", "arrived", "packaging", "shipped", "carrier", "late", "fast"],
    "quality": ["quality", "build", "material", "durable", "sturdy", "cheap", "plastic", "broke", "defective"],
    "price": ["price", "value", "money", "overpriced", "cost", "cheap", "expensive", "deal", "worth"],
    "service": ["service", "customer", "support", "return", "refund", "seller", "contact", "helpful"]
}

def generate_reviews(num_records=1000):
    data = []
    base_date = datetime(2026, 1, 1)
    
    for i in range(num_records):
        # Pick a sentiment
        sentiment = random.choices(["positive", "neutral", "negative"], weights=[0.5, 0.2, 0.3])[0]
        
        # Select base text template
        base_text = random.choice(review_templates[sentiment])
        
        # Inject topic keywords to give KMeans something to cluster
        topic_target = random.choice(["shipping", "quality", "price", "service"])
        keyword = random.choice(topic_keywords[topic_target])
        
        # Contextual phrases to inject keywords naturally
        filler_phrases = [
            f" Regarding the {keyword}, I must say it was notable.",
            f" The {keyword} aspect could be improved.",
            f" I was particularly focused on the {keyword}.",
            f" Standard {keyword} experience overall.",
            f" Especially satisfied with the {keyword}."
        ]
        review_text = base_text + random.choice(filler_phrases)
        
        # Map rating based on sentiment
        if sentiment == "positive":
            rating = random.choices([5, 4], weights=[0.8, 0.2])[0]
        elif sentiment == "neutral":
            rating = random.choices([3, 4, 2], weights=[0.7, 0.15, 0.15])[0]
        else:
            rating = random.choices([1, 2], weights=[0.7, 0.3])[0]
            
        category = random.choice(categories)
        date = base_date + timedelta(days=random.randint(0, 150))
        helpful_votes = random.randint(0, 50)
        
        data.append({
            "review_id": f"REV_{i+1:04d}",
            "category": category,
            "rating": rating,
            "review_text": review_text,
            "sentiment": sentiment,
            "helpful_votes": helpful_votes,
            "date": date.strftime("%Y-%m-%d")
        })
        
    return pd.DataFrame(data)

df = generate_reviews(1000)
df.to_csv("models/sample_reviews.csv", index=False)
print(f"Dataset generated and saved to models/sample_reviews.csv. Columns: {list(df.columns)}")

# 2. PREPROCESS TEXT
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Preprocessing review text...")
df["clean_text"] = df["review_text"].apply(preprocess_text)

# 3. TRAIN SUPERVISED SENTIMENT CLASSIFIER
print("Training Custom Sentiment Classifier...")
X = df["clean_text"]
y = df["sentiment"]

# Pure Python train-test split
def pure_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1.0 - test_size))
    train_idx = shuffled_indices[:split_idx]
    test_idx = shuffled_indices[split_idx:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

X_train, X_test, y_train, y_test = pure_train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = PureTfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train.tolist())
X_test_vec = vectorizer.transform(X_test.tolist())

sentiment_model = PureNaiveBayes()
sentiment_model.fit(X_train_vec, y_train.to_numpy())

# Evaluate
y_pred = sentiment_model.predict(X_test_vec)
accuracy = np.mean(y_pred == y_test.to_numpy())
print(f"Sentiment Classifier Accuracy: {accuracy:.4f}")

# Calculate simple confusion metrics and prints
classes = np.unique(y)
for c in classes:
    true_positive = np.sum((y_test == c) & (y_pred == c))
    false_positive = np.sum((y_test != c) & (y_pred == c))
    false_negative = np.sum((y_test == c) & (y_pred != c))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Class: {c:10s} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# 4. TRAIN UNSUPERVISED TOPIC CLUSTERING (K-Means)
print("Training Unsupervised Topic Clustering (K-Means)...")
X_all_vec = vectorizer.transform(df["clean_text"].tolist())

kmeans = PureKMeans(n_clusters=4, random_state=42)
kmeans.fit(X_all_vec)

# Let's inspect cluster centers and identify key terms to assign topic names
terms = vectorizer.get_feature_names_out()

cluster_topic_map = {}
cluster_top_words = {}

print("Top terms per cluster:")
for i in range(4):
    centroid = kmeans.cluster_centers_[i]
    order_centroids = centroid.argsort()[::-1]
    top_words = [terms[ind] for ind in order_centroids[:15]]
    cluster_top_words[i] = top_words[:5]
    print(f"Cluster {i}: {', '.join(top_words)}")
    
    # Map clusters to logical labels based on top words
    if any(w in top_words for w in ["shipping", "delivery", "arrived", "packaging", "shipped"]):
        cluster_topic_map[i] = "Shipping & Delivery"
    elif any(w in top_words for w in ["quality", "build", "material", "durable", "sturdy", "plastic", "broke"]):
        cluster_topic_map[i] = "Product Quality"
    elif any(w in top_words for w in ["price", "value", "money", "overpriced", "cost", "worth"]):
        cluster_topic_map[i] = "Value & Pricing"
    elif any(w in top_words for w in ["service", "customer", "support", "return", "seller", "refund"]):
        cluster_topic_map[i] = "Customer Service"
    else:
        # Check topic_keywords manually if centroid top_words are sparse
        assigned = False
        for topic_key, keywords in topic_keywords.items():
            if any(k in top_words for k in keywords):
                cluster_topic_map[i] = {
                    "shipping": "Shipping & Delivery",
                    "quality": "Product Quality",
                    "price": "Value & Pricing",
                    "service": "Customer Service"
                }[topic_key]
                assigned = True
                break
        if not assigned:
            cluster_topic_map[i] = f"General Topic {i+1}"

print("Mapped Clusters to Topics:")
for i, topic in cluster_topic_map.items():
    print(f"Cluster {i} -> {topic}")

# 5. SAVE MODELS AND VECTORIZER
print("Saving models to disk...")
joblib.dump(sentiment_model, "models/sentiment_model.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")
joblib.dump(kmeans, "models/kmeans_model.joblib")

# Save metadata (like cluster mapping and vocabulary details)
metadata = {
    "cluster_topic_map": cluster_topic_map,
    "cluster_top_words": cluster_top_words,
    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "accuracy": float(accuracy)
}
joblib.dump(metadata, "models/model_metadata.joblib")

print("All models successfully trained and saved!")

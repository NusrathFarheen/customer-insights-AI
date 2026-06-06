import os
import re
import io
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from pure_ml import PureTfidfVectorizer, PureNaiveBayes, PureKMeans


app = FastAPI(
    title="AI Feedback Insights API",
    description="Endpoints for text sentiment classification and unsupervised topic clustering.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
sentiment_model = None
vectorizer = None
kmeans_model = None
model_metadata = None

# Paths
MODELS_DIR = "models"
SENTIMENT_PATH = os.path.join(MODELS_DIR, "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.joblib")

def load_models():
    global sentiment_model, vectorizer, kmeans_model, model_metadata
    
    # Check if models exist, if not, train them
    if not (os.path.exists(SENTIMENT_PATH) and os.path.exists(VECTORIZER_PATH) and 
            os.path.exists(KMEANS_PATH) and os.path.exists(METADATA_PATH)):
        import sys
        import subprocess
        result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("Failed to auto-train models.")
            
    # Load from files
    print("Loading ML models...")
    sentiment_model = joblib.load(SENTIMENT_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    kmeans_model = joblib.load(KMEANS_PATH)
    model_metadata = joblib.load(METADATA_PATH)
    print("Models loaded successfully!")

@app.on_event("startup")
def startup_event():
    load_models()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_key_terms(text: str, top_n: int = 5) -> List[str]:
    """Extract top TF-IDF words from a single document as its key terms."""
    clean_text = preprocess_text(text)
    if not clean_text:
        return []
    
    # Transform text to tf-idf vector
    vec = vectorizer.transform([clean_text])
    feature_names = vectorizer.get_feature_names_out()
    
    # Get vector coefficients from dense numpy array
    vec_1d = vec[0]
    non_zero_indices = np.nonzero(vec_1d)[0]
    tuples = [(idx, vec_1d[idx]) for idx in non_zero_indices]
    sorted_items = sorted(tuples, key=lambda x: x[1], reverse=True)
    
    # Return top words
    return [feature_names[idx] for idx, score in sorted_items[:top_n]]

# Pydantic models for API request/response
class SingleTextRequest(BaseModel):
    text: str

class ReviewItem(BaseModel):
    review_id: str
    category: str
    rating: int
    review_text: str
    sentiment: str
    topic: str
    helpful_votes: int
    date: str
    key_terms: List[str]

class SingleAnalysisResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    topic: str
    key_terms: List[str]

class BulkAnalysisResponse(BaseModel):
    total_reviews: int
    sentiment_distribution: Dict[str, int]
    topic_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    rating_distribution: Dict[str, int]
    average_rating: float
    reviews: List[ReviewItem]

@app.get("/")
def read_root():
    if sentiment_model is None:
        return {"status": "loading", "message": "Models are loading or training."}
    return {
        "status": "ready",
        "model_details": {
            "sentiment_accuracy": model_metadata.get("accuracy", 0.0),
            "trained_at": model_metadata.get("trained_at", "unknown"),
            "topics": list(model_metadata.get("cluster_topic_map", {}).values())
        }
    }

@app.post("/api/analyze/single", response_model=SingleAnalysisResponse)
def analyze_single(request: SingleTextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    # Preprocess
    cleaned = preprocess_text(request.text)
    if not cleaned:
        return SingleAnalysisResponse(
            text=request.text,
            sentiment="neutral",
            confidence=0.5,
            topic="General",
            key_terms=[]
        )
        
    # Vectorize
    vec = vectorizer.transform([cleaned])
    
    # Sentiment prediction & confidence
    sentiment = sentiment_model.predict(vec)[0]
    probs = sentiment_model.predict_proba(vec)[0]
    classes = sentiment_model.classes
    # Find confidence for predicted class
    class_idx = list(classes).index(sentiment)
    confidence = float(probs[class_idx])
    
    # Topic clustering
    cluster_idx = int(kmeans_model.predict(vec)[0])
    topic = model_metadata["cluster_topic_map"].get(str(cluster_idx), model_metadata["cluster_topic_map"].get(cluster_idx, "General"))
    
    # Key terms
    key_terms = extract_key_terms(request.text, top_n=5)
    
    return SingleAnalysisResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=confidence,
        topic=topic,
        key_terms=key_terms
    )

@app.post("/api/analyze/bulk", response_model=BulkAnalysisResponse)
async def analyze_bulk(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Verify required columns, make assumptions if columns are missing
        text_col = None
        for col in ["review_text", "text", "body", "content", "Review"]:
            if col in df.columns:
                text_col = col
                break
                
        if not text_col:
            # Fallback: find the first text/object column
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                text_col = text_cols[0]
            else:
                raise HTTPException(status_code=400, detail="No suitable text column found in CSV.")
                
        # Fill optional columns if missing
        if "category" not in df.columns:
            df["category"] = "General"
        if "rating" not in df.columns:
            df["rating"] = 3
        if "helpful_votes" not in df.columns:
            df["helpful_votes"] = 0
        if "date" not in df.columns:
            df["date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
        if "review_id" not in df.columns:
            df["review_id"] = [f"UPLOAD_{i+1:04d}" for i in range(len(df))]
            
        # Clean dates
        df["date"] = df["date"].fillna(pd.Timestamp.now().strftime("%Y-%m-%d"))
        df["review_text"] = df[text_col].fillna("").astype(str)
        df["clean_text"] = df["review_text"].apply(preprocess_text)
        
        # Vectorize
        X_vec = vectorizer.transform(df["clean_text"])
        
        # Predictions
        df["sentiment"] = sentiment_model.predict(X_vec)
        df["cluster"] = kmeans_model.predict(X_vec)
        df["topic"] = df["cluster"].apply(
            lambda c: model_metadata["cluster_topic_map"].get(str(c), model_metadata["cluster_topic_map"].get(c, "General"))
        )
        
        # Extract terms for all rows
        key_terms_list = []
        for text in df["review_text"]:
            key_terms_list.append(extract_key_terms(text, top_n=5))
        df["key_terms"] = key_terms_list
        
        # Calculate Aggregated Metrics
        total_reviews = len(df)
        
        # Handle nan/empty counts
        df["sentiment"] = df["sentiment"].astype(str)
        df["topic"] = df["topic"].astype(str)
        df["category"] = df["category"].fillna("General").astype(str)
        df["rating"] = df["rating"].fillna(3).astype(int)
        
        sentiment_dist = df["sentiment"].value_counts().to_dict()
        topic_dist = df["topic"].value_counts().to_dict()
        cat_dist = df["category"].value_counts().to_dict()
        rating_dist = df["rating"].value_counts().to_dict()
        # Convert keys in rating_dist to string for JSON compliance
        rating_dist_str = {str(k): int(v) for k, v in rating_dist.items()}
        
        avg_rating = float(df["rating"].mean()) if total_reviews > 0 else 0.0
        
        # Prepare individual items list (limit to 100 for JSON size performance)
        reviews_list = []
        for _, row in df.iterrows():
            reviews_list.append(ReviewItem(
                review_id=str(row["review_id"]),
                category=str(row["category"]),
                rating=int(row["rating"]),
                review_text=str(row["review_text"]),
                sentiment=str(row["sentiment"]),
                topic=str(row["topic"]),
                helpful_votes=int(row["helpful_votes"]),
                date=str(row["date"]),
                key_terms=row["key_terms"]
            ))
            
        return BulkAnalysisResponse(
            total_reviews=total_reviews,
            sentiment_distribution=sentiment_dist,
            topic_distribution=topic_dist,
            category_distribution=cat_dist,
            rating_distribution=rating_dist_str,
            average_rating=avg_rating,
            reviews=reviews_list
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.get("/api/analyze/sample", response_model=BulkAnalysisResponse)
def analyze_sample():
    sample_path = os.path.join(MODELS_DIR, "sample_reviews.csv")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample dataset not found. Please train models first.")
    try:
        df = pd.read_csv(sample_path)
        
        df["review_text"] = df["review_text"].fillna("").astype(str)
        df["clean_text"] = df["review_text"].apply(preprocess_text)
        
        # Vectorize and predict
        X_vec = vectorizer.transform(df["clean_text"])
        df["sentiment"] = sentiment_model.predict(X_vec)
        df["cluster"] = kmeans_model.predict(X_vec)
        df["topic"] = df["cluster"].apply(
            lambda c: model_metadata["cluster_topic_map"].get(str(c), model_metadata["cluster_topic_map"].get(c, "General"))
        )
        
        # Extract terms
        key_terms_list = []
        for text in df["review_text"]:
            key_terms_list.append(extract_key_terms(text, top_n=5))
        df["key_terms"] = key_terms_list
        
        total_reviews = len(df)
        sentiment_dist = df["sentiment"].value_counts().to_dict()
        topic_dist = df["topic"].value_counts().to_dict()
        cat_dist = df["category"].value_counts().to_dict()
        rating_dist = df["rating"].value_counts().to_dict()
        rating_dist_str = {str(k): int(v) for k, v in rating_dist.items()}
        avg_rating = float(df["rating"].mean()) if total_reviews > 0 else 0.0
        
        reviews_list = []
        for _, row in df.iterrows():
            reviews_list.append(ReviewItem(
                review_id=str(row["review_id"]),
                category=str(row["category"]),
                rating=int(row["rating"]),
                review_text=str(row["review_text"]),
                sentiment=str(row["sentiment"]),
                topic=str(row["topic"]),
                helpful_votes=int(row["helpful_votes"]),
                date=str(row["date"]),
                key_terms=row["key_terms"]
            ))
            
        return BulkAnalysisResponse(
            total_reviews=total_reviews,
            sentiment_distribution=sentiment_dist,
            topic_distribution=topic_dist,
            category_distribution=cat_dist,
            rating_distribution=rating_dist_str,
            average_rating=avg_rating,
            reviews=reviews_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading sample dataset: {str(e)}")

# AI Resume Screening Schemas and Endpoint
from pypdf import PdfReader

class CandidateScreenResponse(BaseModel):
    name: str = ""
    filename: str
    match_score: float
    matching_skills: List[str]
    missing_skills: List[str]

class ScreeningResponse(BaseModel):
    candidates: List[CandidateScreenResponse]

SKILLS_LIST = [
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust", "ruby", "php", "sql", "html", "css",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch", "keras", 
    "scikit-learn", "numpy", "pandas", "scipy", "transformers", "huggingface", "llm", "rag", "embeddings",
    "react", "angular", "vue", "next.js", "vite", "nodejs", "express", "fastapi", "flask", "django", 
    "docker", "kubernetes", "aws", "azure", "gcp", "firebase", "mongodb", "postgresql", "mysql", "redis",
    "git", "github", "agile", "jira", "scrum", "jenkins", "ci/cd"
]

def parse_pdf_text(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        return ""

@app.post("/api/screen", response_model=ScreeningResponse)
async def screen_resumes(
    job_description: str = Form(...),
    files: List[UploadFile] = File(...)
):
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job Description cannot be empty.")
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one resume.")
        
    resume_texts = []
    candidates_info = []
    
    for file in files:
        contents = await file.read()
        text = ""
        if file.filename.endswith('.pdf'):
            text = parse_pdf_text(contents)
        else:
            text = contents.decode("utf-8", errors="ignore")
            
        if not text.strip():
            text = "Empty Resume"
            
        resume_texts.append(text)
        
        # Candidate name from filename
        name = file.filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
        for suffix in [" Resume", " CV", "Resume", "CV"]:
            if name.endswith(suffix):
                name = name[:len(name)-len(suffix)].strip()
        candidates_info.append({"name": name, "filename": file.filename, "text": text})
        
    try:
        # Fit vectorizer temp on Job Description + Resumes
        vectorizer_temp = PureTfidfVectorizer(max_features=1000)
        all_docs = [job_description] + resume_texts
        vectors = vectorizer_temp.fit_transform(all_docs)
        
        jd_vector = vectors[0]
        resume_vectors = vectors[1:]
        
        results = []
        for i, info in enumerate(candidates_info):
            res_vector = resume_vectors[i]
            # L2 normalized vectors dot product = cosine similarity
            score = float(np.dot(jd_vector, res_vector))
            
            jd_lower = job_description.lower()
            res_lower = info["text"].lower()
            
            matching_skills = []
            missing_skills = []
            
            for skill in SKILLS_LIST:
                if skill == "c++":
                    pattern = r'\bc\+\+(?=[^a-zA-Z0-9]|$)'
                elif skill == "c#":
                    pattern = r'\bc#(?=[^a-zA-Z0-9]|$)'
                else:
                    pattern = r'\b' + re.escape(skill) + r'\b'
                    
                if re.search(pattern, jd_lower):
                    if re.search(pattern, res_lower):
                        matching_skills.append(skill)
                    else:
                        missing_skills.append(skill)
                        
            results.append(CandidateScreenResponse(
                name=info["name"],
                filename=info["filename"],
                match_score=score,
                matching_skills=matching_skills,
                missing_skills=missing_skills
            ))
            
        # Sort candidates by score descending
        results.sort(key=lambda x: x.match_score, reverse=True)
        return ScreeningResponse(candidates=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in screening pipeline: {str(e)}")



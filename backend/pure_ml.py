import re
import numpy as np

# Simple list of English stop words to exclude from TF-IDF
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 
    'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 
    'weren', 'won', 'wouldn'
}

def clean_and_tokenize(text):
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    # Filter stop words and short tokens
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

class PureTfidfVectorizer:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocabulary_ = {}
        self.feature_names_ = []
        self.idf_ = []
        
    def fit(self, raw_documents):
        # Count document frequency for each word
        df = {}
        n_samples = len(raw_documents)
        
        for doc in raw_documents:
            tokens = set(clean_and_tokenize(doc))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
                
        # Sort words by document frequency and select top max_features
        sorted_words = sorted(df.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:self.max_features]
        
        # Build vocabulary
        self.vocabulary_ = {word[0]: idx for idx, word in enumerate(top_words)}
        self.feature_names_ = [word[0] for word in top_words]
        
        # Compute IDF with smooth formula: log((1 + n_samples) / (1 + df)) + 1
        self.idf_ = []
        for word in self.feature_names_:
            word_df = df[word]
            idf = np.log((1 + n_samples) / (1 + word_df)) + 1.0
            self.idf_.append(idf)
        self.idf_ = np.array(self.idf_)
        return self
        
    def transform(self, raw_documents):
        n_samples = len(raw_documents)
        n_features = len(self.feature_names_)
        X = np.zeros((n_samples, n_features))
        
        for i, doc in enumerate(raw_documents):
            tokens = clean_and_tokenize(doc)
            if not tokens:
                continue
            
            # Count term frequencies (TF) in document
            tf = {}
            for t in tokens:
                if t in self.vocabulary_:
                    tf[t] = tf.get(t, 0) + 1
                    
            # Populate TF-IDF values
            for word, freq in tf.items():
                idx = self.vocabulary_[word]
                # Log normalization TF
                tf_val = 1 + np.log(freq) if freq > 0 else 0
                X[i, idx] = tf_val * self.idf_[idx]
                
            # L2 normalization of vectors
            norm = np.linalg.norm(X[i])
            if norm > 0:
                X[i] = X[i] / norm
                
        return X

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)
        
    def get_feature_names_out(self):
        return np.array(self.feature_names_)


class PureNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.classes = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        class_count = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))
        
        # Aggregate counts
        for c_idx, c in enumerate(self.classes):
            # Mask for samples in this class
            mask = (y == c)
            class_count[c_idx] = np.sum(mask)
            feature_count[c_idx] = np.sum(X[mask], axis=0)
            
        # Log priors: log(class_count / n_samples)
        self.class_log_prior_ = np.log(class_count / n_samples)
        
        # Log likelihoods: log((count + alpha) / (total_class_count + alpha * n_features))
        smoothed_feature_count = feature_count + self.alpha
        smoothed_class_feature_sum = np.sum(smoothed_feature_count, axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_feature_count / smoothed_class_feature_sum)
        
        return self
        
    def predict_proba(self, X):
        # Calculate joint log probability: X * log_prob.T + log_prior
        # X is shape (n_samples, n_features), feature_log_prob_ is shape (n_classes, n_features)
        jlog = X @ self.feature_log_prob_.T + self.class_log_prior_
        
        # Softmax over classes to get normalized probabilities
        # Subtract max for numerical stability
        jlog_max = np.max(jlog, axis=1, keepdims=True)
        exp_jlog = np.exp(jlog - jlog_max)
        probs = exp_jlog / np.sum(exp_jlog, axis=1, keepdims=True)
        return probs
        
    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes[indices]


class PureKMeans:
    def __init__(self, n_clusters=4, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        
    def fit(self, X):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        
        # Initialize centroids randomly from data points
        indices = rng.permutation(n_samples)[:self.n_clusters]
        self.cluster_centers_ = X[indices].copy()
        
        for _ in range(self.max_iter):
            # Calculate Euclidean distances from all samples to all centroids
            # dist shape: (n_samples, n_clusters)
            distances = np.zeros((n_samples, self.n_clusters))
            for c_idx in range(self.n_clusters):
                distances[:, c_idx] = np.linalg.norm(X - self.cluster_centers_[c_idx], axis=1)
                
            # Assign labels
            labels = np.argmin(distances, axis=1)
            
            # Recalculate centroids
            new_centers = np.zeros_like(self.cluster_centers_)
            for c_idx in range(self.n_clusters):
                # Mask for samples in this cluster
                mask = (labels == c_idx)
                if np.sum(mask) > 0:
                    new_centers[c_idx] = np.mean(X[mask], axis=0)
                else:
                    # Keep old center or pick a random point if cluster is empty
                    new_centers[c_idx] = self.cluster_centers_[c_idx]
                    
            # Check for convergence
            if np.allclose(self.cluster_centers_, new_centers, atol=1e-6):
                break
                
            self.cluster_centers_ = new_centers
            
        return self
        
    def predict(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        for c_idx in range(self.n_clusters):
            distances[:, c_idx] = np.linalg.norm(X - self.cluster_centers_[c_idx], axis=1)
        return np.argmin(distances, axis=1)

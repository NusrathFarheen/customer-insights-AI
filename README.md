# Customer Insights AI - Sentiment & Topic Explorer Dashboard

An end-to-end customer feedback analytics platform featuring a **custom Machine Learning pipeline** built entirely from scratch in pure Python & NumPy, paired with a modern React + Vite + TypeScript dashboard.

Designed to demonstrate end-to-end software development and ML engineering capability for the **Comp India / Tech.us** AI/ML Engineer recruitment drive.

---

## 🚀 Key Features

* **Supervised Sentiment Classifier**: Multinomial Naive Bayes model classifying customer reviews into *Positive*, *Neutral*, and *Negative* sentiments with confidence scoring.
* **Unsupervised Topic Clustering**: K-Means clustering algorithm partitioning reviews into core business topics: *Product Quality*, *Shipping & Delivery*, *Value & Pricing*, and *Customer Service*.
* **Interactive SaaS Dashboard**: Responsive UI showing key metrics (KPIs), sentiment donut charts, ratings distribution, timeline trend charts, and topic distributions using Recharts.
* **Live Sandbox & Explainer**: A text sandbox where users can type custom feedback, get real-time classifications, and view highlighted TF-IDF keywords that influenced the model's prediction.
* **Feedback Database Explorer**: A searchable and filterable data table supporting query searches and multi-value category, sentiment, and topic filters.
* **Bulk Upload**: A drag-and-drop CSV parser that processes custom review datasets on the fly and updates all dashboard analytics instantly.

---

## 🛠️ The Pure Python/NumPy ML Stack

To ensure environment portability and demonstrate foundational machine learning algorithms without relying on heavy C-compiled dependencies (which are frequently blocked in enterprise Windows AppLocker environments), the entire ML pipeline was written from scratch in [pure_ml.py](backend/pure_ml.py) using only **standard Python and NumPy**:

1. **`PureTfidfVectorizer`**:
   * Tokenizes text and filters English stop words.
   * Computes Term Frequencies (TF) and Inverse Document Frequencies (IDF).
   * Generates document-feature matrices and performs L2 vector normalization.
2. **`PureNaiveBayes` (Multinomial)**:
   * Calculates log priors of target classes.
   * Calculates Laplace-smoothed log likelihoods of word tokens per class.
   * Returns normalized classification probabilities via Softmax function.
3. **`PureKMeans`**:
   * Initializes centroids using random data points.
   * Computes Euclidean distances and groups documents iteratively.
   * Assigns cluster labels and aggregates top TF-IDF centroids.

---

## 📦 Project Architecture

```
newMLproject/
├── backend/
│   ├── models/                # Trained joblib binaries & sample CSV data
│   ├── main.py                # FastAPI web server
│   ├── pure_ml.py             # Custom ML stack (TF-IDF, Naive Bayes, KMeans)
│   ├── train_model.py         # Mock data generator and model training script
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Main React Dashboard UI
│   │   ├── index.css          # Glassmorphic layout stylesheet
│   │   └── main.tsx           # React mounting point
│   ├── index.html             # HTML shell
│   └── package.json           # Node configuration & chart libraries
└── README.md                  # Project documentation (this file)
```

---

## 💻 How to Get Started

### 1. Backend Setup (FastAPI)
Navigate to the `backend` folder, set up your virtual environment, install dependencies, and start the API:
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate      # On Windows
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
*Note: The API automatically verifies if models exist. If not, it trains them on startup using 1,000 generated feedback items and saves the weights in the `models/` directory in 1.5 seconds.*

### 2. Frontend Setup (React + Vite)
In a separate terminal, navigate to the `frontend` folder, install npm packages, and run the development server:
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173/](http://localhost:5173/) in your browser to access the dashboard.

---

## 📐 Algorithmic Implementation Details

### Multinomial Naive Bayes Classification
The Naive Bayes classifier is built on Bayes' Theorem, representing the probability of a class $C_k$ given text tokens $\mathbf{x}$:

$$P(C_k | \mathbf{x}) \propto P(C_k) \prod_{i=1}^{n} P(x_i | C_k)$$

To avoid underflow issues with very small decimal numbers, we implement the math in **log-space**:

$$\log P(C_k | \mathbf{x}) \propto \log P(C_k) + \sum_{i=1}^{n} \log P(x_i | C_k)$$

Laplace smoothing is applied to handle vocabulary terms that do not appear in a class during training:

$$\theta_{i, c} = \frac{N_{c, i} + \alpha}{N_c + \alpha \cdot N_{features}}$$

### K-Means Clustering Centroids
The K-Means algorithm partitions reviews into $K$ clusters by minimizing the sum of squared distances between data vectors and their assigned cluster centroid $\mu_j$:

$$J = \sum_{i=1}^{M} \sum_{j=1}^{K} r_{ij} ||x_i - \mu_j||^2$$

Where $r_{ij} = 1$ if document $x_i$ is assigned to cluster $j$, and $0$ otherwise. Centroids are recomputed as the mean of all assigned vectors:

$$\mu_j = \frac{\sum_{i=1}^{M} r_{ij} x_i}{\sum_{i=1}^{M} r_{ij}}$$

import React, { useState, useEffect, useRef } from 'react';
import { 
  BarChart3, 
  UploadCloud, 
  Terminal, 
  Table, 
  BrainCircuit, 
  Sparkles, 
  Smile, 
  Meh, 
  Frown, 
  AlertTriangle, 
  Search, 
  Database,
  Cpu,
  RefreshCw,
  UserCheck
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  PieChart, 
  Pie, 
  Cell, 
  BarChart as ReBarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Legend, 
  AreaChart, 
  Area,
  CartesianGrid
} from 'recharts';

// Import exported weights, precalculated reviews, and ResumeScreener sub-app
import modelWeights from './model_weights.json';
import sampleReviews from './sample_reviews.json';
import { ResumeScreener } from './ResumeScreener';

const BACKEND_URL = 'http://127.0.0.1:8000';

interface ReviewItem {
  review_id: string;
  category: string;
  rating: number;
  review_text: string;
  sentiment: string;
  topic: string;
  helpful_votes: number;
  date: string;
  key_terms: string[];
}

interface BulkData {
  total_reviews: number;
  sentiment_distribution: Record<string, number>;
  topic_distribution: Record<string, number>;
  category_distribution: Record<string, number>;
  rating_distribution: Record<string, number>;
  average_rating: number;
  reviews: ReviewItem[];
}

interface BackendStatus {
  status: string;
  model_details?: {
    sentiment_accuracy: number;
    trained_at: string;
    topics: string[];
  };
}

interface SandboxResult {
  text: string;
  sentiment: string;
  confidence: number;
  topic: string;
  key_terms: string[];
}

// English Stop Words List for in-browser TF-IDF
const STOP_WORDS = new Set([
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
]);

function App() {
  // Sidebar Navigation State: 'feedback' | 'screener'
  const [activeSidebarTab, setActiveSidebarTab] = useState<'feedback' | 'screener'>('feedback');
  
  // Feedback App Navigation Tab: 'dashboard' | 'sandbox' | 'raw_data' | 'upload'
  const [activeTab, setActiveTab] = useState<'dashboard' | 'sandbox' | 'raw_data' | 'upload'>('dashboard');
  
  // Data State
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [bulkData, setBulkData] = useState<BulkData | null>(null);
  
  // Sandbox State
  const [sandboxInput, setSandboxInput] = useState<string>('');
  const [sandboxResult, setSandboxResult] = useState<SandboxResult | null>(null);
  const [sandboxLoading, setSandboxLoading] = useState<boolean>(false);
  
  // Upload State
  const [dragActive, setDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Table Filters State
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [sentimentFilter, setSentimentFilter] = useState<string>('all');
  const [topicFilter, setTopicFilter] = useState<string>('all');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');

  // Check backend health & Load initial data
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/`);
      if (response.ok) {
        const data = await response.json();
        setBackendStatus(data);
        setApiOnline(true);
        loadSampleDataset(true);
      } else {
        setApiOnline(false);
        loadSampleDataset(false);
      }
    } catch (err) {
      setApiOnline(false);
      loadSampleDataset(false);
    }
  };

  const loadSampleDataset = async (useAPI: boolean) => {
    setLoading(true);
    setError(null);
    
    if (useAPI) {
      try {
        const response = await fetch(`${BACKEND_URL}/api/analyze/sample`);
        if (response.ok) {
          const data = await response.json();
          setBulkData(data);
          setLoading(false);
          return;
        }
      } catch (err) {
        // Fallback silently
      }
    }
    
    // Fallback: load static JSON dataset directly in the browser
    setTimeout(() => {
      try {
        const aggregates = computeAggregates(sampleReviews as ReviewItem[]);
        setBulkData(aggregates);
      } catch (err) {
        setError('Failed to load local sample dataset.');
      } finally {
        setLoading(false);
      }
    }, 300);
  };

  // CLIENT-SIDE SENTIMENT INFERENCE ENGINE
  const runLocalInference = (text: string): { sentiment: string; confidence: number; topic: string; key_terms: string[] } => {
    const cleaned = text.toLowerCase().replace(/[^a-zA-Z\s]/g, '').replace(/\s+/g, ' ').trim();
    const tokens = cleaned.split(' ').filter(t => !STOP_WORDS.has(t) && t.length > 1);

    const vocab = modelWeights.vocabulary as Record<string, number>;
    const idf = modelWeights.idf as number[];
    const nFeatures = idf.length;
    
    const vector = new Array(nFeatures).fill(0);
    if (tokens.length > 0) {
      const tf: Record<string, number> = {};
      for (const t of tokens) {
        if (t in vocab) {
          tf[t] = (tf[t] || 0) + 1;
        }
      }
      for (const [word, freq] of Object.entries(tf)) {
        const idx = vocab[word];
        const tfVal = 1 + Math.log(freq);
        vector[idx] = tfVal * idf[idx];
      }
      
      let sumSq = 0;
      for (let i = 0; i < nFeatures; i++) sumSq += vector[i] * vector[i];
      const norm = Math.sqrt(sumSq);
      if (norm > 0) {
        for (let i = 0; i < nFeatures; i++) vector[i] /= norm;
      }
    }

    const classes = modelWeights.classes as string[];
    const classPrior = modelWeights.class_log_prior as number[];
    const featureLogProb = modelWeights.feature_log_prob as number[][];
    
    const logProbs = [...classPrior];
    for (let c = 0; c < classes.length; c++) {
      for (let f = 0; f < nFeatures; f++) {
        logProbs[c] += vector[f] * featureLogProb[c][f];
      }
    }
    
    const maxLog = Math.max(...logProbs);
    const exps = logProbs.map(v => Math.exp(v - maxLog));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probabilities = exps.map(v => v / sumExps);
    
    const predClassIdx = probabilities.indexOf(Math.max(...probabilities));
    const sentiment = classes[predClassIdx];
    const confidence = probabilities[predClassIdx];

    const centroids = modelWeights.cluster_centers as number[][];
    let minClusterIdx = 0;
    let minDist = Infinity;
    
    for (let c = 0; c < centroids.length; c++) {
      let sumSqDiff = 0;
      for (let f = 0; f < nFeatures; f++) {
        const diff = vector[f] - centroids[c][f];
        sumSqDiff += diff * diff;
      }
      const dist = Math.sqrt(sumSqDiff);
      if (dist < minDist) {
        minDist = dist;
        minClusterIdx = c;
      }
    }
    const topic = (modelWeights.cluster_topic_map as Record<string, string>)[String(minClusterIdx)] || 'General';

    const featureNames = modelWeights.feature_names as string[];
    const tuples: { word: string; val: number }[] = [];
    for (let f = 0; f < nFeatures; f++) {
      if (vector[f] > 0) {
        tuples.push({ word: featureNames[f], val: vector[f] });
      }
    }
    const keyTerms = tuples
      .sort((a, b) => b.val - a.val)
      .slice(0, 5)
      .map(item => item.word);

    return { sentiment, confidence, topic, key_terms: keyTerms };
  };

  const computeAggregates = (reviews: ReviewItem[]): BulkData => {
    const total = reviews.length;
    if (total === 0) {
      return {
        total_reviews: 0,
        sentiment_distribution: {},
        topic_distribution: {},
        category_distribution: {},
        rating_distribution: {},
        average_rating: 0,
        reviews: []
      };
    }

    const sentDist: Record<string, number> = { positive: 0, neutral: 0, negative: 0 };
    const topicDist: Record<string, number> = {};
    const catDist: Record<string, number> = {};
    const ratingDist: Record<string, number> = { '1': 0, '2': 0, '3': 0, '4': 0, '5': 0 };
    let ratingSum = 0;

    reviews.forEach(review => {
      sentDist[review.sentiment] = (sentDist[review.sentiment] || 0) + 1;
      topicDist[review.topic] = (topicDist[review.topic] || 0) + 1;
      catDist[review.category] = (catDist[review.category] || 0) + 1;
      ratingDist[String(review.rating)] = (ratingDist[String(review.rating)] || 0) + 1;
      ratingSum += review.rating;
    });

    return {
      total_reviews: total,
      sentiment_distribution: sentDist,
      topic_distribution: topicDist,
      category_distribution: catDist,
      rating_distribution: ratingDist,
      average_rating: ratingSum / total,
      reviews
    };
  };

  const handleSandboxAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sandboxInput.trim()) return;
    
    setSandboxLoading(true);
    setError(null);

    if (apiOnline) {
      try {
        const response = await fetch(`${BACKEND_URL}/api/analyze/single`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: sandboxInput })
        });
        if (response.ok) {
          const data = await response.json();
          setSandboxResult(data);
          setSandboxLoading(false);
          return;
        }
      } catch (err) {
        // Fallback silently
      }
    }

    // Local Fallback
    setTimeout(() => {
      try {
        const result = runLocalInference(sandboxInput);
        setSandboxResult({
          text: sandboxInput,
          ...result
        });
      } catch (err) {
        setError('Error running client-side inference.');
      } finally {
        setSandboxLoading(false);
      }
    }, 100);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = async (file: File) => {
    if (!file.name.endsWith('.csv')) {
      setError('Please upload a valid CSV file.');
      return;
    }

    setLoading(true);
    setError(null);

    if (apiOnline) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await fetch(`${BACKEND_URL}/api/analyze/bulk`, {
          method: 'POST',
          body: formData
        });
        if (response.ok) {
          const data = await response.json();
          setBulkData(data);
          setActiveTab('dashboard');
          setLoading(false);
          return;
        }
      } catch (err) {
        // Fallback silently
      }
    }

    // Local Fallback: Parse CSV
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        if (!text) throw new Error('Empty file content.');
        
        const lines = text.split(/\r?\n/);
        if (lines.length <= 1) throw new Error('CSV must contain a header row.');
        
        const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
        let textColIdx = headers.findIndex(h => /text|review|body|content/i.test(h));
        if (textColIdx === -1) textColIdx = 0;
        
        const catColIdx = headers.findIndex(h => /category/i.test(h));
        const ratingColIdx = headers.findIndex(h => /rating/i.test(h));
        const dateColIdx = headers.findIndex(h => /date/i.test(h));
        
        const parsedReviews: ReviewItem[] = [];
        for (let i = 1; i < lines.length; i++) {
          const line = lines[i];
          if (!line.trim()) continue;
          
          const cols: string[] = [];
          let insideQuote = false;
          let currentVal = '';
          for (let charIdx = 0; charIdx < line.length; charIdx++) {
            const char = line[charIdx];
            if (char === '"') {
              insideQuote = !insideQuote;
            } else if (char === ',' && !insideQuote) {
              cols.push(currentVal.trim().replace(/^"|"$/g, ''));
              currentVal = '';
            } else {
              currentVal += char;
            }
          }
          cols.push(currentVal.trim().replace(/^"|"$/g, ''));
          
          const reviewText = cols[textColIdx] || '';
          if (!reviewText.trim()) continue;
          
          const rating = ratingColIdx !== -1 && cols[ratingColIdx] ? parseInt(cols[ratingColIdx]) || 3 : 3;
          const category = catColIdx !== -1 && cols[catColIdx] ? cols[catColIdx] : 'General';
          const date = dateColIdx !== -1 && cols[dateColIdx] ? cols[dateColIdx] : new Date().toISOString().split('T')[0];
          
          const inf = runLocalInference(reviewText);
          
          parsedReviews.push({
            review_id: `LOCAL_${i}`,
            category,
            rating,
            review_text: reviewText,
            sentiment: inf.sentiment,
            topic: inf.topic,
            helpful_votes: 0,
            date,
            key_terms: inf.key_terms
          });
        }
        
        const aggregates = computeAggregates(parsedReviews);
        setBulkData(aggregates);
        setActiveTab('dashboard');
      } catch (err: any) {
        setError(err.message || 'Error parsing CSV file locally.');
      } finally {
        setLoading(false);
      }
    };
    reader.readAsText(file);
  };

  const getSentimentChartData = () => {
    if (!bulkData) return [];
    const dist = bulkData.sentiment_distribution;
    return [
      { name: 'Positive', value: dist.positive || 0, color: '#10b981' },
      { name: 'Neutral', value: dist.neutral || 0, color: '#f59e0b' },
      { name: 'Negative', value: dist.negative || 0, color: '#ef4444' }
    ].filter(item => item.value > 0);
  };

  const getTopicChartData = () => {
    if (!bulkData) return [];
    return Object.entries(bulkData.topic_distribution).map(([topic, count]) => ({
      name: topic,
      count: count
    }));
  };

  const getRatingChartData = () => {
    if (!bulkData) return [];
    const dist = bulkData.rating_distribution;
    return [1, 2, 3, 4, 5].map(rating => ({
      rating: `${rating} ★`,
      count: dist[String(rating)] || dist[rating] || 0
    }));
  };

  const getTimelineChartData = () => {
    if (!bulkData) return [];
    const dateCounts: Record<string, { positive: number, neutral: number, negative: number }> = {};
    bulkData.reviews.forEach(review => {
      const dateStr = review.date;
      if (!dateCounts[dateStr]) dateCounts[dateStr] = { positive: 0, neutral: 0, negative: 0 };
      if (review.sentiment === 'positive') dateCounts[dateStr].positive++;
      else if (review.sentiment === 'neutral') dateCounts[dateStr].neutral++;
      else if (review.sentiment === 'negative') dateCounts[dateStr].negative++;
    });
    return Object.entries(dateCounts)
      .map(([date, counts]) => ({
        date: date,
        Positive: counts.positive,
        Neutral: counts.neutral,
        Negative: counts.negative,
        Total: counts.positive + counts.neutral + counts.negative
      }))
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  };

  const getFilteredReviews = () => {
    if (!bulkData) return [];
    return bulkData.reviews.filter(review => {
      const matchesSearch = 
        review.review_text.toLowerCase().includes(searchQuery.toLowerCase()) ||
        review.review_id.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesSentiment = sentimentFilter === 'all' || review.sentiment === sentimentFilter;
      const matchesTopic = topicFilter === 'all' || review.topic === topicFilter;
      const matchesCategory = categoryFilter === 'all' || review.category === categoryFilter;
      return matchesSearch && matchesSentiment && matchesTopic && matchesCategory;
    });
  };

  const getUniqueCategories = () => {
    if (!bulkData) return [];
    return Array.from(new Set(bulkData.reviews.map(r => r.category)));
  };

  const getUniqueTopics = () => {
    if (!bulkData) return [];
    return Array.from(new Set(bulkData.reviews.map(r => r.topic)));
  };

  const getSentimentStats = () => {
    if (!bulkData) return { posPercent: 0, neuPercent: 0, negPercent: 0 };
    const total = bulkData.total_reviews;
    if (total === 0) return { posPercent: 0, neuPercent: 0, negPercent: 0 };
    return {
      posPercent: Math.round(((bulkData.sentiment_distribution.positive || 0) / total) * 100),
      neuPercent: Math.round(((bulkData.sentiment_distribution.neutral || 0) / total) * 100),
      negPercent: Math.round(((bulkData.sentiment_distribution.negative || 0) / total) * 100)
    };
  };

  const stats = getSentimentStats();
  const filteredReviews = getFilteredReviews();

  return (
    <div className="app-layout">
      
      {/* Sleek Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <Cpu size={22} style={{ color: 'var(--primary)' }} />
          <span className="sidebar-brand-title">AI Portfolio</span>
        </div>
        
        <nav className="sidebar-menu">
          <div 
            className={`sidebar-item ${activeSidebarTab === 'feedback' ? 'active' : ''}`}
            onClick={() => {
              setActiveSidebarTab('feedback');
              setError(null);
            }}
          >
            <BrainCircuit size={18} />
            <span className="sidebar-item-text">Feedback Analytics</span>
          </div>
          
          <div 
            className={`sidebar-item ${activeSidebarTab === 'screener' ? 'active' : ''}`}
            onClick={() => {
              setActiveSidebarTab('screener');
              setError(null);
            }}
          >
            <UserCheck size={18} />
            <span className="sidebar-item-text">Resume Screener</span>
          </div>
        </nav>
        
        <div className="sidebar-footer">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            <Cpu size={14} />
            <span className="sidebar-footer-text">Local fallback active</span>
          </div>
        </div>
      </aside>

      {/* Main Workspace Frame */}
      <div className="main-content-wrapper">
        
        {/* Render View 1: Customer Feedback Insights */}
        {activeSidebarTab === 'feedback' && (
          <div className="dashboard-container" style={{ padding: 0 }}>
            {/* Top Dashboard Header */}
            <header className="header">
              <div className="brand-section">
                <h1>Customer Insights AI</h1>
                <p>Supervised Sentiment Classifier & Unsupervised Topic Clustered Dashboard</p>
              </div>
              
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <button className="upload-btn" style={{ marginTop: 0, padding: '0.4rem 0.8rem', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--glass-border)' }} onClick={() => loadSampleDataset(apiOnline)} disabled={loading}>
                  <RefreshCw size={14} className={loading ? 'loading-pulse' : ''} style={{ marginRight: '0.25rem' }} />
                  Reload Dataset
                </button>
                
                <div className="model-status-badge">
                  <span className="status-dot" style={{ backgroundColor: apiOnline ? 'var(--sentiment-pos)' : 'var(--accent)', boxShadow: apiOnline ? '0 0 8px var(--sentiment-pos)' : '0 0 8px var(--accent)' }}></span>
                  <span>{apiOnline ? 'Cloud Backend Online' : 'In-Browser ML Engine'}</span>
                </div>
              </div>
            </header>

            {/* Global Error Banner */}
            {error && (
              <div className="alert">
                <AlertTriangle size={18} />
                <div>{error}</div>
              </div>
            )}

            {loading ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px', gap: '1rem' }}>
                <div className="loading-spinner"></div>
                <p className="loading-pulse" style={{ color: 'var(--text-secondary)' }}>Processing NLP features & clustering centroids...</p>
              </div>
            ) : (
              <main className="main-content" style={{ padding: 0 }}>
                {/* Inner Tab Bar for Feedback Sub-sections */}
                <nav className="tabs-navigation" style={{ marginBottom: '1.5rem' }}>
                  <button className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
                    <BarChart3 size={16} />
                    Overview
                  </button>
                  <button className={`tab-btn ${activeTab === 'sandbox' ? 'active' : ''}`} onClick={() => setActiveTab('sandbox')}>
                    <Terminal size={16} />
                    Live Sandbox
                  </button>
                  <button className={`tab-btn ${activeTab === 'raw_data' ? 'active' : ''}`} onClick={() => setActiveTab('raw_data')}>
                    <Table size={16} />
                    Feedback Explorer
                  </button>
                  <button className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>
                    <UploadCloud size={16} />
                    Bulk Upload
                  </button>
                </nav>

                {/* 1.1 Overview */}
                {activeTab === 'dashboard' && bulkData && (
                  <>
                    <div className="stats-grid">
                      <div className="glass-card stat-card">
                        <span className="stat-title">Feedback Volume</span>
                        <span className="stat-value">{bulkData.total_reviews.toLocaleString()}</span>
                        <div className="stat-footer">
                          <Database size={12} />
                          <span>Reviews Analyzed</span>
                        </div>
                      </div>
                      <div className="glass-card stat-card">
                        <span className="stat-title">Customer Satisfaction</span>
                        <span className="stat-value">{bulkData.average_rating.toFixed(2)} ★</span>
                        <div className="stat-footer">
                          <Sparkles size={12} style={{ color: '#fbbf24' }} />
                          <span>Average Rating</span>
                        </div>
                      </div>
                      <div className="glass-card stat-card">
                        <span className="stat-title">Positive Ratio</span>
                        <span className="stat-value" style={{ color: 'var(--sentiment-pos)' }}>{stats.posPercent}%</span>
                        <div className="stat-footer">
                          <Smile size={12} style={{ color: 'var(--sentiment-pos)' }} />
                          <span>Positive Sentiments</span>
                        </div>
                      </div>
                      <div className="glass-card stat-card">
                        <span className="stat-title">Negative Ratio</span>
                        <span className="stat-value" style={{ color: 'var(--sentiment-neg)' }}>{stats.negPercent}%</span>
                        <div className="stat-footer">
                          <Frown size={12} style={{ color: 'var(--sentiment-neg)' }} />
                          <span>Negative Sentiments</span>
                        </div>
                      </div>
                    </div>

                    <div className="dashboard-grid">
                      <div className="glass-card chart-card">
                        <h3 className="section-title"><Smile size={18} style={{ color: 'var(--primary)' }} /> Sentiment Distribution</h3>
                        <div className="chart-container" style={{ display: 'flex', alignItems: 'center' }}>
                          <div style={{ width: '60%', height: '100%' }}>
                            <ResponsiveContainer width="100%" height="100%">
                              <PieChart>
                                <Pie
                                  data={getSentimentChartData()}
                                  cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value"
                                >
                                  {getSentimentChartData().map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                  ))}
                                </Pie>
                                <Tooltip contentStyle={{ background: '#0d0c22', border: '1px solid var(--glass-border)', borderRadius: '8px', color: 'var(--text-primary)' }} />
                              </PieChart>
                            </ResponsiveContainer>
                          </div>
                          <div style={{ width: '40%', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                            {getSentimentChartData().map((entry, idx) => (
                              <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                                <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: entry.color }}></span>
                                <span>{entry.name}: <b>{entry.value}</b></span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div className="glass-card chart-card">
                        <h3 className="section-title"><Sparkles size={18} style={{ color: '#fbbf24' }} /> Star Rating Breakdown</h3>
                        <div className="chart-container">
                          <ResponsiveContainer width="100%" height="100%">
                            <ReBarChart data={getRatingChartData()}>
                              <XAxis dataKey="rating" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                              <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} />
                              <Tooltip cursor={{ fill: 'rgba(255,255,255,0.01)' }} contentStyle={{ background: '#0d0c22', border: '1px solid var(--glass-border)', borderRadius: '8px', color: 'var(--text-primary)' }} />
                              <Bar dataKey="count" fill="url(#colorRating)" radius={[4, 4, 0, 0]}>
                                <defs>
                                  <linearGradient id="colorRating" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.8}/>
                                    <stop offset="95%" stopColor="#d97706" stopOpacity={0.2}/>
                                  </linearGradient>
                                </defs>
                              </Bar>
                            </ReBarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>

                    <div className="dashboard-grid" style={{ gridTemplateColumns: '1.2fr 0.8fr' }}>
                      <div className="glass-card chart-card">
                        <h3 className="section-title"><BarChart3 size={18} style={{ color: 'var(--accent)' }} /> Timeline Volume Trends</h3>
                        <div className="chart-container">
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={getTimelineChartData()}>
                              <defs>
                                <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.3}/>
                                  <stop offset="95%" stopColor="var(--primary)" stopOpacity={0}/>
                                </linearGradient>
                              </defs>
                              <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={10} tickLine={false} />
                              <YAxis stroke="var(--text-muted)" fontSize={10} tickLine={false} axisLine={false} />
                              <CartesianGrid stroke="rgba(255,255,255,0.02)" vertical={false} />
                              <Tooltip contentStyle={{ background: '#0d0c22', border: '1px solid var(--glass-border)', borderRadius: '8px', color: 'var(--text-primary)' }} />
                              <Legend wrapperStyle={{ fontSize: 11 }} />
                              <Area type="monotone" dataKey="Total" stroke="var(--primary)" fill="url(#colorTotal)" />
                              <Area type="monotone" dataKey="Positive" stroke="var(--sentiment-pos)" fillOpacity={0} />
                              <Area type="monotone" dataKey="Negative" stroke="var(--sentiment-neg)" fillOpacity={0} />
                            </AreaChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      <div className="glass-card chart-card">
                        <h3 className="section-title"><BrainCircuit size={18} style={{ color: 'var(--secondary)' }} /> Topic Clusters</h3>
                        <div className="chart-container">
                          <ResponsiveContainer width="100%" height="100%">
                            <ReBarChart data={getTopicChartData()} layout="vertical">
                              <XAxis type="number" stroke="var(--text-muted)" fontSize={10} hide />
                              <YAxis dataKey="name" type="category" stroke="var(--text-primary)" fontSize={10} width={90} tickLine={false} axisLine={false} />
                              <Tooltip cursor={{ fill: 'rgba(255,255,255,0.01)' }} contentStyle={{ background: '#0d0c22', border: '1px solid var(--glass-border)', borderRadius: '8px', color: 'var(--text-primary)' }} />
                              <Bar dataKey="count" fill="url(#colorTopic)" radius={[0, 4, 4, 0]}>
                                <defs>
                                  <linearGradient id="colorTopic" x1="0" y1="0" x2="1" y2="0">
                                    <stop offset="5%" stopColor="var(--secondary)" stopOpacity={0.8}/>
                                    <stop offset="95%" stopColor="var(--primary)" stopOpacity={0.2}/>
                                  </linearGradient>
                                </defs>
                              </Bar>
                            </ReBarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>

                    <div className="glass-card" style={{ padding: '1rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      <span>Model: <b>Multinomial Naive Bayes & K-Means</b> (Acc: <b>{((backendStatus?.model_details?.sentiment_accuracy || modelWeights.accuracy) * 100).toFixed(1)}%</b>)</span>
                      <span>Mode: <b>{apiOnline ? 'FastAPI Server' : 'In-Browser JS Engine'}</b></span>
                    </div>
                  </>
                )}

                {/* 1.2 Sandbox */}
                {activeTab === 'sandbox' && (
                  <div className="glass-card sandbox-card">
                    <h2 className="section-title"><Terminal size={18} style={{ color: 'var(--primary)' }} /> Real-time Sandbox</h2>
                    <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem', fontSize: '0.9rem' }}>Type customer reviews below. The pipeline will vectorize terms and compute sentiment & cluster topics instantly.</p>
                    <div className="sandbox-layout">
                      <form onSubmit={handleSandboxAnalyze}>
                        <textarea
                          className="sandbox-textarea"
                          placeholder="e.g. Absolutely outstanding build quality! Extremely sturdy design..."
                          value={sandboxInput}
                          onChange={(e) => setSandboxInput(e.target.value)}
                        />
                        <div className="analyze-btn-row">
                          <button type="submit" className="upload-btn" style={{ margin: 0 }} disabled={sandboxLoading || !sandboxInput.trim()}>
                            {sandboxLoading ? 'Analyzing...' : 'Run Analysis'}
                          </button>
                        </div>
                      </form>
                      <div className="sandbox-results">
                        {sandboxResult ? (
                          <div>
                            <div className="result-label">Predicted Sentiment</div>
                            <div className="result-value-big">
                              {sandboxResult.sentiment === 'positive' && <span className="sentiment-badge positive"><Smile size={14} style={{ marginRight: '0.25rem' }} /> Positive</span>}
                              {sandboxResult.sentiment === 'neutral' && <span className="sentiment-badge neutral"><Meh size={14} style={{ marginRight: '0.25rem' }} /> Neutral</span>}
                              {sandboxResult.sentiment === 'negative' && <span className="sentiment-badge negative"><Frown size={14} style={{ marginRight: '0.25rem' }} /> Negative</span>}
                            </div>
                            <div className="result-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span>Confidence Score</span>
                              <span>{Math.round(sandboxResult.confidence * 100)}%</span>
                            </div>
                            <div className="confidence-bar-wrapper">
                              <div className="confidence-bar" style={{ width: `${sandboxResult.confidence * 100}%`, backgroundColor: sandboxResult.sentiment === 'positive' ? 'var(--sentiment-pos)' : sandboxResult.sentiment === 'neutral' ? 'var(--sentiment-neu)' : 'var(--sentiment-neg)' }}></div>
                            </div>
                            <div className="result-label">Topic Cluster</div>
                            <div style={{ fontWeight: 600, marginBottom: '1rem' }}><span className="topic-chip">{sandboxResult.topic}</span></div>
                            <div className="result-label">Explainer Key Terms</div>
                            <div className="word-highlights">
                              {sandboxResult.key_terms.map((w, i) => <span key={i} className="word-highlight-tag">{w}</span>)}
                            </div>
                          </div>
                        ) : (
                          <div className="result-placeholder">Submit text to view in-browser prediction</div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* 1.3 Explorer Table */}
                {activeTab === 'raw_data' && bulkData && (
                  <div className="glass-card table-card">
                    <h2 className="section-title"><Table size={18} /> Feedback Explorer</h2>
                    <div className="table-header-actions">
                      <div style={{ position: 'relative' }}>
                        <Search size={14} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                        <input
                          type="text" className="search-input" style={{ paddingLeft: '2.25rem' }}
                          placeholder="Search text..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
                        />
                      </div>
                      <div className="filters-row">
                        <select className="filter-select" value={sentimentFilter} onChange={(e) => setSentimentFilter(e.target.value)}>
                          <option value="all">All Sentiments</option>
                          <option value="positive">Positive</option>
                          <option value="neutral">Neutral</option>
                          <option value="negative">Negative</option>
                        </select>
                        <select className="filter-select" value={topicFilter} onChange={(e) => setTopicFilter(e.target.value)}>
                          <option value="all">All Topics</option>
                          {getUniqueTopics().map((t, idx) => <option key={idx} value={t}>{t}</option>)}
                        </select>
                        <select className="filter-select" value={categoryFilter} onChange={(e) => setCategoryFilter(e.target.value)}>
                          <option value="all">All Categories</option>
                          {getUniqueCategories().map((c, idx) => <option key={idx} value={c}>{c}</option>)}
                        </select>
                      </div>
                    </div>

                    <div className="table-wrapper">
                      <table className="custom-table">
                        <thead>
                          <tr>
                            <th>ID</th>
                            <th>Category</th>
                            <th>Rating</th>
                            <th>Review Text</th>
                            <th>Sentiment</th>
                            <th>Topic</th>
                            <th>Key Terms</th>
                          </tr>
                        </thead>
                        <tbody>
                          {filteredReviews.slice(0, 100).map((row, idx) => (
                            <tr key={idx}>
                              <td style={{ fontFamily: 'monospace', color: 'var(--primary)', fontSize: '0.8rem' }}>{row.review_id}</td>
                              <td>{row.category}</td>
                              <td>
                                <div className="rating-stars">
                                  {Array.from({ length: row.rating }).map((_, i) => <span key={i}>★</span>)}
                                </div>
                              </td>
                              <td style={{ maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={row.review_text}>{row.review_text}</td>
                              <td><span className={`sentiment-badge ${row.sentiment}`}>{row.sentiment}</span></td>
                              <td><span className="topic-chip">{row.topic}</span></td>
                              <td>
                                <div className="terms-list">
                                  {row.key_terms.slice(0, 3).map((w, i) => <span key={i} className="term-tag">{w}</span>)}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* 1.4 Bulk Upload */}
                {activeTab === 'upload' && (
                  <div className="glass-card sandbox-card" style={{ padding: '3rem 2rem' }}>
                    <h2 className="section-title"><UploadCloud size={18} /> Batch CSV Upload</h2>
                    <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>Upload a CSV file of reviews. The engine will parse and score each item, dynamically updating all graphs.</p>
                    <div 
                      className={`upload-container ${dragActive ? 'drag-active' : ''}`}
                      onDragEnter={handleDrag} onDragOver={handleDrag} onDragLeave={handleDrag} onDrop={handleDrop}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <input type="file" ref={fileInputRef} style={{ display: 'none' }} accept=".csv" onChange={handleFileChange} />
                      <UploadCloud className="upload-icon" />
                      <div className="upload-text">Drag and drop reviews CSV file here</div>
                      <div className="upload-subtext">Click to choose file</div>
                    </div>
                  </div>
                )}

              </main>
            )}
          </div>
        )}

        {/* Render View 2: Resume Screening System */}
        {activeSidebarTab === 'screener' && (
          <div className="dashboard-container" style={{ padding: 0 }}>
            {/* Top Dashboard Header */}
            <header className="header">
              <div className="brand-section">
                <h1>Candidate Screening System</h1>
                <p>AI Resume Parser, TF-IDF Job Description Similarity Matcher, and Candidate Ranker</p>
              </div>
              
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div className="model-status-badge">
                  <span className="status-dot" style={{ backgroundColor: apiOnline ? 'var(--sentiment-pos)' : 'var(--accent)', boxShadow: apiOnline ? '0 0 8px var(--sentiment-pos)' : '0 0 8px var(--accent)' }}></span>
                  <span>{apiOnline ? 'Cloud Backend Online' : 'In-Browser ML Engine'}</span>
                </div>
              </div>
            </header>

            <ResumeScreener apiOnline={apiOnline} />
          </div>
        )}

      </div>
    </div>
  );
}

export default App;

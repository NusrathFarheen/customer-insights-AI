import React, { useState, useRef, useEffect } from 'react';
import { 
  UploadCloud, 
  Sparkles, 
  CheckCircle, 
  XCircle, 
  User, 
  Briefcase, 
  Trash2, 
  FileText,
  FileCode,
  ShieldAlert
} from 'lucide-react';

const BACKEND_URL = 'http://127.0.0.1:8000';

interface Candidate {
  review_id?: string; // fallback matching ReviewItem interface if needed
  name: string;
  filename: string;
  match_score: number;
  matching_skills: string[];
  missing_skills: string[];
  text?: string;
}

interface ResumeScreenerProps {
  apiOnline: boolean;
}

// Common tech skills dictionary for matching
const SKILLS_LIST = [
  "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust", "ruby", "php", "sql", "html", "css",
  "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch", "keras", 
  "scikit-learn", "numpy", "pandas", "scipy", "transformers", "huggingface", "llm", "rag", "embeddings",
  "react", "angular", "vue", "next.js", "vite", "nodejs", "express", "fastapi", "flask", "django", 
  "docker", "kubernetes", "aws", "azure", "gcp", "firebase", "mongodb", "postgresql", "mysql", "redis",
  "git", "github", "agile", "jira", "scrum", "jenkins", "ci/cd"
];

// Stop words for pure TS TF-IDF Vectorizer
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

// Helper to dynamically load PDF.js from CDN for client-side PDF parsing
const loadPDFJS = (): Promise<any> => {
  return new Promise((resolve, reject) => {
    if ((window as any).pdfjsLib) {
      resolve((window as any).pdfjsLib);
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.min.js';
    script.onload = () => {
      const pdfjsLib = (window as any).pdfjsLib;
      pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js';
      resolve(pdfjsLib);
    };
    script.onerror = () => reject(new Error('Failed to load PDF.js CDN.'));
    document.head.appendChild(script);
  });
};

// Pure TS TF-IDF Vectorizer
class TSTfidfVectorizer {
  vocabulary: Record<string, number> = {};
  featureNames: string[] = [];
  idf: number[] = [];
  maxFeatures = 1000;

  cleanAndTokenize(text: string): string[] {
    const cleaned = text.toLowerCase().replace(/[^a-zA-Z\s]/g, '').replace(/\s+/g, ' ').trim();
    return cleaned.split(' ').filter(t => !STOP_WORDS.has(t) && t.length > 1);
  }

  fitTransform(docs: string[]): number[][] {
    const df: Record<string, number> = {};
    const nSamples = docs.length;

    // 1. Compute Document Frequencies
    for (const doc of docs) {
      const tokens = new Set(this.cleanAndTokenize(doc));
      for (const token of tokens) {
        df[token] = (df[token] || 0) + 1;
      }
    }

    // 2. Select Top Features
    const sorted = Object.entries(df).sort((a, b) => b[1] - a[1]);
    const topWords = sorted.slice(0, this.maxFeatures);
    
    this.vocabulary = {};
    this.featureNames = [];
    topWords.forEach(([word], idx) => {
      this.vocabulary[word] = idx;
      this.featureNames.push(word);
    });

    // 3. Compute IDF
    this.idf = [];
    this.featureNames.forEach(word => {
      const wordDf = df[word];
      const idfVal = Math.log((1 + nSamples) / (1 + wordDf)) + 1.0;
      this.idf.push(idfVal);
    });

    // 4. Transform all docs into L2 normalized vectors
    return docs.map(doc => {
      const tokens = this.cleanAndTokenize(doc);
      const vector = new Array(this.featureNames.length).fill(0);
      if (tokens.length === 0) return vector;

      const tf: Record<string, number> = {};
      for (const token of tokens) {
        if (token in this.vocabulary) {
          tf[token] = (tf[token] || 0) + 1;
        }
      }

      for (const [word, freq] of Object.entries(tf)) {
        const idx = this.vocabulary[word];
        const tfVal = 1 + Math.log(freq);
        vector[idx] = tfVal * this.idf[idx];
      }

      // L2 Normalization
      let sumSq = 0;
      for (let i = 0; i < vector.length; i++) sumSq += vector[i] * vector[i];
      const norm = Math.sqrt(sumSq);
      if (norm > 0) {
        for (let i = 0; i < vector.length; i++) vector[i] /= norm;
      }
      return vector;
    });
  }
}

export const ResumeScreener: React.FC<ResumeScreenerProps> = ({ apiOnline }) => {
  const [jobDescription, setJobDescription] = useState<string>(
    `We are seeking a talented AI/Machine Learning Engineer to join our team. 
Requirements:
- Strong programming experience in Python.
- Completed coursework or projects involving Machine Learning and NLP.
- Hands-on experience with Scikit-Learn, NumPy, and Pandas.
- Familiarity with deep learning frameworks like PyTorch or TensorFlow.
- Experience with FastAPI, Git, Github, and Agile methodologies is preferred.`
  );
  
  const [resumes, setResumes] = useState<{ filename: string; file: File | null; text: string; candidateName: string }[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);
  
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load standard pre-filled candidates to show immediate value
  useEffect(() => {
    loadPreFilledCandidates();
  }, []);

  const loadPreFilledCandidates = () => {
    const preFilled = [
      {
        name: "Nusrath Farheen (Recommended)",
        filename: "Nusrath_Farheen_ML_Resume.pdf",
        text: "Talented B.Tech graduate skilled in AI, Machine Learning, and Python. Completed academic projects involving NLP sentiment analysis and unsupervised clustering. Hands-on experience with Scikit-Learn, NumPy, Pandas, and Git/Github. Familiar with PyTorch, FastAPI, Docker, and Agile technologies. Participation in Kaggle competitions.",
        rating: 5
      },
      {
        name: "Vijay Kumar",
        filename: "Vijay_Kumar_Resume.pdf",
        text: "Software Developer specializing in Java, Spring Boot, MySQL, and HTML/CSS. Strong background in object-oriented programming. Familiar with Git, Javascript, and AWS. Looking for full-stack opportunities.",
        rating: 3
      },
      {
        name: "Sarah Jenkins",
        filename: "Sarah_Jenkins_CV.txt",
        text: "Data Analyst with 2 years of experience. Expert in Python, SQL, Tableau, and Excel. Strong skills in Pandas, NumPy, and basic data visualization. Completed data analytics coursework.",
        rating: 4
      }
    ];

    // Local evaluation on startup to compute scores against the pre-filled JD
    const mockResumes = preFilled.map(c => ({
      filename: c.filename,
      file: null,
      text: c.text,
      candidateName: c.name
    }));
    
    setResumes(mockResumes);
    runLocalScreening(jobDescription, mockResumes);
  };

  // Run local TS/JS TF-IDF Cosine Similarity Matching
  const runLocalScreening = (jd: string, list: typeof resumes) => {
    if (!jd.trim() || list.length === 0) {
      setCandidates([]);
      return;
    }

    try {
      const vectorizer = new TSTfidfVectorizer();
      const allDocs = [jd, ...list.map(r => r.text)];
      const vectors = vectorizer.fitTransform(allDocs);
      
      const jdVector = vectors[0];
      const results: Candidate[] = [];

      for (let i = 0; i < list.length; i++) {
        const resVector = vectors[i + 1];
        
        // Cosine similarity = dot product of L2 normalized vectors
        let score = 0;
        for (let j = 0; j < jdVector.length; j++) {
          score += jdVector[j] * resVector[j];
        }

        // Skills matching
        const jdLower = jd.toLowerCase();
        const resLower = list[i].text.toLowerCase();
        const matchingSkills: string[] = [];
        const missingSkills: string[] = [];

        const escapeRegExp = (str: string) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        SKILLS_LIST.forEach(skill => {
          let pattern: RegExp;
          if (skill === 'c++') {
            pattern = /\bc\+\+(?=[^a-zA-Z0-9]|$)/i;
          } else if (skill === 'c#') {
            pattern = /\bc#(?=[^a-zA-Z0-9]|$)/i;
          } else {
            pattern = new RegExp(`\\b${escapeRegExp(skill)}\\b`, 'i');
          }
          
          if (pattern.test(jdLower)) {
            if (pattern.test(resLower)) {
              matchingSkills.push(skill);
            } else {
              missingSkills.push(skill);
            }
          }
        });

        results.push({
          name: list[i].candidateName,
          filename: list[i].filename,
          match_score: score,
          matching_skills: matchingSkills,
          missing_skills: missingSkills,
          text: list[i].text
        });
      }

      // Sort by score descending
      results.sort((a, b) => b.match_score - a.match_score);
      setCandidates(results);
    } catch (err: any) {
      console.error(err);
      setError(`Failed to run local ML similarity matcher: ${err.message || err}`);
    }
  };

  // Run screening (API or Client-side depending on online status)
  const handleScreening = async () => {
    if (resumes.length === 0) {
      setError("Please upload or add at least one candidate resume.");
      return;
    }
    
    setLoading(true);
    setError(null);
    setSelectedCandidate(null);

    // If API is online, use FastAPI multipart upload
    if (apiOnline) {
      try {
        const formData = new FormData();
        formData.append("job_description", jobDescription);
        
        // Loop over resumes. If they uploaded files, append them. Otherwise, append mock files
        resumes.forEach((r) => {
          if (r.file) {
            formData.append("files", r.file);
          } else {
            // Create a fake text blob for pre-filled data
            const blob = new Blob([r.text], { type: 'text/plain' });
            formData.append("files", blob, r.filename);
          }
        });

        const response = await fetch(`${BACKEND_URL}/api/screen`, {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          const data = await response.json();
          // Merge text back into results for the details modal
          const formatted: Candidate[] = data.candidates.map((c: any) => {
            const match = resumes.find(r => r.filename === c.filename);
            return {
              ...c,
              text: match ? match.text : ""
            };
          });
          setCandidates(formatted);
          setLoading(false);
          return;
        }
      } catch (err) {
        // Fallback silently
      }
    }

    // Fallback: Run local TS/JS screening (immediate and robust!)
    setTimeout(() => {
      runLocalScreening(jobDescription, resumes);
      setLoading(false);
    }, 400);
  };

  // Parse TXT / PDF files
  const processResumeFile = async (file: File) => {
    setError(null);
    const filename = file.name;

    // Generate Candidate Name
    const lastDotIdx = filename.lastIndexOf('.');
    let candidateName = lastDotIdx !== -1 ? filename.slice(0, lastDotIdx) : filename;
    candidateName = candidateName.replace(/_|--|-/g, ' ').trim();
    // Capitalize words
    candidateName = candidateName.split(' ').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    // Clean suffix
    for (const suffix of [" Resume", " CV", "Resume", "CV"]) {
      if (candidateName.endsWith(suffix)) {
        candidateName = candidateName.slice(0, -suffix.length).trim();
      }
    }

    try {
      let text = "";
      if (filename.endsWith('.pdf')) {
        text = await extractTextFromPDF(file);
      } else {
        // Read text files
        text = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target?.result as string || "");
          reader.onerror = () => reject(new Error('Failed to read text file.'));
          reader.readAsText(file);
        });
      }

      if (!text.trim()) {
        throw new Error("Could not extract readable text from this file.");
      }

      const newResume = { filename, file, text, candidateName };
      setResumes(prev => {
        const updated = [...prev.filter(r => r.filename !== filename), newResume];
        // Auto-run screening on file changes
        runLocalScreening(jobDescription, updated);
        return updated;
      });

    } catch (err: any) {
      setError(`Error reading file ${filename}: ${err.message || err}`);
    }
  };

  const extractTextFromPDF = async (file: File): Promise<string> => {
    try {
      const pdfjsLib = await loadPDFJS();
      const arrayBuffer = await file.arrayBuffer();
      const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
      let text = '';
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const tokenizedText = await page.getTextContent();
        const pageText = tokenizedText.items.map((token: any) => token.str).join(' ');
        text += pageText + '\n';
      }
      return text;
    } catch (e) {
      throw new Error("Failed to load PDF.js CDN parser. Verify your internet connection or use a TXT file.");
    }
  };

  // Drag and Drop
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
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      for (let i = 0; i < e.dataTransfer.files.length; i++) {
        await processResumeFile(e.dataTransfer.files[i]);
      }
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      for (let i = 0; i < e.target.files.length; i++) {
        await processResumeFile(e.target.files[i]);
      }
    }
  };

  const removeResume = (filename: string) => {
    const updated = resumes.filter(r => r.filename !== filename);
    setResumes(updated);
    if (selectedCandidate?.filename === filename) {
      setSelectedCandidate(null);
    }
    runLocalScreening(jobDescription, updated);
  };

  const clearAllResumes = () => {
    setResumes([]);
    setCandidates([]);
    setSelectedCandidate(null);
  };

  return (
    <div className="glass-card sandbox-card" style={{ padding: '2rem' }}>
      <h2 className="section-title">
        <Sparkles size={20} style={{ color: 'var(--primary)' }} />
        AI Resume Screening System
      </h2>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem', fontSize: '0.95rem' }}>
        Compare candidate resumes against job description requirements. Our engine tokenizes files, computes **TF-IDF vectors**, calculates **Cosine Similarity**, and identifies matching & missing technical skills.
      </p>

      {error && (
        <div className="alert" style={{ marginBottom: '1.5rem' }}>
          <ShieldAlert size={18} />
          <div>{error}</div>
        </div>
      )}

      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px', gap: '1rem' }}>
          <div className="loading-spinner"></div>
          <p className="loading-pulse" style={{ color: 'var(--text-secondary)' }}>Extracting PDF text and compiling TF-IDF vector similarity metrics...</p>
        </div>
      ) : (
        <div className="sandbox-layout" style={{ gridTemplateColumns: '1fr 1.5fr', gap: '2rem' }}>
          
          {/* Left Panel: Job Description & Resumes List */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* Job Description Textarea */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <Briefcase size={16} style={{ color: 'var(--primary)' }} />
                <label className="result-label" style={{ margin: 0, fontSize: '0.85rem' }}>Job Description (JD)</label>
              </div>
              <textarea
                className="sandbox-textarea"
                style={{ height: '220px', fontSize: '0.9rem' }}
                placeholder="Paste Job Description here..."
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
              />
            </div>

            {/* Resume Upload Box */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <UploadCloud size={16} style={{ color: 'var(--secondary)' }} />
                <label className="result-label" style={{ margin: 0, fontSize: '0.85rem' }}>Upload Resumes (PDF / TXT)</label>
              </div>
              
              <div 
                className={`upload-container ${dragActive ? 'drag-active' : ''}`}
                style={{ padding: '1.5rem 1rem', borderRadius: '12px' }}
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept=".pdf,.txt"
                  multiple
                  onChange={handleFileChange}
                />
                <UploadCloud size={28} style={{ color: 'var(--primary)', margin: '0 auto 0.5rem' }} />
                <div style={{ fontSize: '0.8rem', fontWeight: 600 }}>Drag Resumes Here</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Supports PDFs & TXT files</div>
              </div>
            </div>

            {/* Uploaded Candidates List */}
            {resumes.length > 0 && (
              <div className="glass-card" style={{ padding: '1rem', background: 'rgba(0,0,0,0.15)', borderRadius: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    Uploaded Resumes ({resumes.length})
                  </span>
                  <button 
                    onClick={clearAllResumes}
                    style={{ background: 'transparent', border: 'none', color: 'var(--sentiment-neg)', cursor: 'pointer', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                  >
                    <Trash2 size={12} /> Clear All
                  </button>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '150px', overflowY: 'auto', paddingRight: '0.25rem' }}>
                  {resumes.map((resume, idx) => (
                    <div 
                      key={idx} 
                      style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.02)', padding: '0.5rem', borderRadius: '6px', border: '1px solid var(--glass-border)' }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', overflow: 'hidden' }}>
                        {resume.filename.endsWith('.pdf') ? (
                          <FileText size={14} style={{ color: 'var(--sentiment-neg)' }} />
                        ) : (
                          <FileCode size={14} style={{ color: 'var(--accent)' }} />
                        )}
                        <span style={{ fontSize: '0.8rem', fontWeight: 500, textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }} title={resume.filename}>
                          {resume.candidateName}
                        </span>
                      </div>
                      <button 
                        onClick={() => removeResume(resume.filename)}
                        style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
                      >
                        <Trash2 size={12} className="tab-btn-hover" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            <button className="upload-btn" style={{ margin: 0, width: '100%' }} onClick={handleScreening}>
              Analyze & Rank Candidates
            </button>
          </div>
          
          {/* Right Panel: Ranked Candidates Table & Details Panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* Rankings Table */}
            <div className="glass-card" style={{ padding: '1.25rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <User size={16} style={{ color: 'var(--accent)' }} />
                <h3 style={{ fontSize: '0.95rem', fontWeight: 600, margin: 0 }}>Candidate Match Ranks</h3>
              </div>

              {candidates.length > 0 ? (
                <div style={{ flex: 1, overflowY: 'auto' }}>
                  <table className="custom-table" style={{ fontSize: '0.85rem' }}>
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Name</th>
                        <th>Match Score</th>
                        <th>Matching Skills</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {candidates.map((c, idx) => (
                        <tr 
                          key={idx} 
                          style={{ cursor: 'pointer', background: selectedCandidate?.filename === c.filename ? 'rgba(139, 92, 246, 0.05)' : '' }}
                          onClick={() => setSelectedCandidate(c)}
                        >
                          <td style={{ fontWeight: 700, color: idx === 0 ? 'var(--sentiment-pos)' : 'var(--text-secondary)' }}>
                            #{idx + 1}
                          </td>
                          <td style={{ fontWeight: 600 }}>{c.name}</td>
                          <td>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: '100px' }}>
                              <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>
                                {Math.round(c.match_score * 100)}%
                              </span>
                              <div className="confidence-bar-wrapper" style={{ height: '5px', flex: 1, margin: 0, background: 'rgba(255,255,255,0.05)' }}>
                                <div 
                                  className="confidence-bar" 
                                  style={{ 
                                    width: `${c.match_score * 100}%`,
                                    background: idx === 0 ? 'linear-gradient(90deg, var(--sentiment-pos), #34d399)' : 'linear-gradient(90deg, var(--primary), var(--secondary))'
                                  }}
                                ></div>
                              </div>
                            </div>
                          </td>
                          <td>
                            <div className="terms-list">
                              {c.matching_skills.slice(0, 3).map((s, i) => (
                                <span key={i} className="term-tag" style={{ background: 'rgba(16,185,129,0.1)', color: 'var(--sentiment-pos)', border: '1px solid rgba(16,185,129,0.2)', fontSize: '0.65rem' }}>
                                  {s}
                                </span>
                              ))}
                              {c.matching_skills.length > 3 && (
                                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                                  +{c.matching_skills.length - 3}
                                </span>
                              )}
                            </div>
                          </td>
                          <td>
                            <button 
                              className="tab-btn" 
                              style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', border: '1px solid var(--glass-border)' }}
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedCandidate(c);
                              }}
                            >
                              Inspect
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.9rem', padding: '3rem 0' }}>
                  <User size={36} style={{ opacity: 0.3, marginBottom: '0.5rem' }} />
                  <span>No candidates screened yet.</span>
                  <span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Click "Analyze & Rank" to run matching pipeline.</span>
                </div>
              )}
            </div>

            {/* Candidate Details Drawer */}
            {selectedCandidate && (
              <div className="glass-card" style={{ padding: '1.25rem', borderLeft: '3px solid var(--primary)', background: 'rgba(13, 12, 34, 0.5)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <div>
                    <h4 style={{ fontSize: '1.05rem', fontWeight: 700 }}>{selectedCandidate.name}</h4>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{selectedCandidate.filename}</span>
                  </div>
                  <div className="sentiment-badge positive" style={{ fontSize: '0.8rem', padding: '0.3rem 0.6rem' }}>
                    Match: {Math.round(selectedCandidate.match_score * 100)}%
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.85rem' }}>
                  <div>
                    <div className="result-label" style={{ marginBottom: '0.25rem' }}>
                      Matching Skills ({selectedCandidate.matching_skills.length})
                    </div>
                    <div className="word-highlights">
                      {selectedCandidate.matching_skills.length > 0 ? (
                        selectedCandidate.matching_skills.map((skill, idx) => (
                          <span key={idx} className="word-highlight-tag" style={{ background: 'rgba(16,185,129,0.15)', color: '#a7f3d0', border: '1px solid rgba(16,185,129,0.3)' }}>
                            <CheckCircle size={10} style={{ marginRight: '0.25rem', color: 'var(--sentiment-pos)' }} /> {skill}
                          </span>
                        ))
                      ) : (
                        <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>No matching keywords found.</span>
                      )}
                    </div>
                  </div>

                  <div>
                    <div className="result-label" style={{ marginBottom: '0.25rem' }}>
                      Missing Required Skills ({selectedCandidate.missing_skills.length})
                    </div>
                    <div className="word-highlights">
                      {selectedCandidate.missing_skills.length > 0 ? (
                        selectedCandidate.missing_skills.map((skill, idx) => (
                          <span key={idx} className="word-highlight-tag" style={{ background: 'rgba(239,68,68,0.15)', color: '#fca5a5', border: '1px solid rgba(239,68,68,0.3)' }}>
                            <XCircle size={10} style={{ marginRight: '0.25rem', color: 'var(--sentiment-neg)' }} /> {skill}
                          </span>
                        ))
                      ) : (
                        <span style={{ color: 'var(--sentiment-pos)', fontStyle: 'italic', fontWeight: 500 }}>
                          <CheckCircle size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.25rem' }} /> All required skills found!
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

          </div>

        </div>
      )}
    </div>
  );
};

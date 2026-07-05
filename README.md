# 🚀 AI Research Paper Intelligence System

An end-to-end AI Research Paper Intelligence System assistant that enables semantic paper discovery, automatic summarization, keyword extraction, named entity recognition, topic analysis, deep learning-based classification, and report generation from ArXiv Machine Learning papers—all through an interactive Streamlit web application.

---

# 📖 Project Overview

Reading hundreds or even thousands of research papers is both time-consuming and inefficient. This project automates the complete research paper analysis workflow by combining multiple NLP and Deep Learning techniques into a single pipeline.

The system is capable of:

- 🔍 Performing semantic search across more than **2,000 ArXiv Machine Learning papers** using **FAISS vector search**
- 📝 Producing concise paper summaries with the **T5 Transformer**
- 🏷️ Extracting important keywords and keyphrases using **KeyBERT**
- 👤 Detecting named entities such as people, organizations, and locations through **spaCy NER**
- 📈 Analyzing topic distributions and research trends
- 🧠 Predicting paper categories using an **LSTM-based Deep Learning classifier**
- 📄 Creating comprehensive intelligence reports in **JSON** and **CSV** formats

---

# 🏗️ System Architecture

```text
                     User Query
                         │
                         ▼
        ┌────────────────────────────────┐
        │     FAISS Semantic Search      │
        │ Retrieve relevant research papers
        └───────────────┬────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │      T5 Text Summarization     │
        │ Generate concise paper summaries
        └───────────────┬────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │    KeyBERT Keyword Extraction  │
        │ Extract important keywords
        └───────────────┬────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │    spaCy Named Entity Recognition
        │ Detect PERSON, ORG, GPE & LOC
        └───────────────┬────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │     Topic Trend Analysis       │
        │ Statistical insights & charts
        └───────────────┬────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │     LSTM Topic Classification  │
        │ Predict paper category
        └───────────────┬────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │    Final Intelligence Report   │
        │ Export results as JSON & CSV
        └────────────────────────────────┘
```

---

# 📂 Project Directory

```text
AI Research Paper Intelligence System/
│
├── run_pipeline.py          # Executes the complete processing pipeline
├── app.py                   # Streamlit web application
├── ML-Arxiv-Papers.csv      # Raw dataset (optional local copy)
├── README.md
│
├── pipeline/
│   ├── 01_eda_and_embeddings.py
│   ├── 02_searcher.py
│   ├── 03_summarizer.py
│   ├── 04_keyword_finder.py
│   ├── 05_ner.py
│   ├── 06_topic_trend_analysis.py
│   ├── 07_deep_learning_classifier.py
│   └── 08_genai.py
│
├── screenshots/
│   ├── search_paper.png
│   ├── topic_trend.png
│   ├── classify_paper.png
│   └── final_paper.png
│
└── data/
    ├── cleaned_papers.csv
    ├── arxiv_embeddings.npy
    ├── paper_faiss.index
    ├── search_results.json
    ├── summarized_results.json
    ├── keywords_results.json
    ├── ner_results.json
    ├── trend_report.csv
    ├── lstm_classifier.h5
    ├── tokenizer.pkl
    ├── label_encoder.pkl
    ├── cluster_names.json
    ├── classification_report.json
    ├── final_report.json
    └── final_report.csv
```

---

# 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| Hugging Face Datasets | Source of ArXiv Machine Learning papers |
| Sentence Transformers (`all-MiniLM-L6-v2`) | Sentence embedding generation |
| FAISS | High-speed semantic similarity search |
| T5-small | Transformer-based text summarization |
| KeyBERT | Keyword and keyphrase extraction |
| spaCy (`en_core_web_sm`) | Named Entity Recognition |
| TensorFlow / Keras | Deep Learning LSTM classifier |
| scikit-learn | Data preprocessing, label encoding & train/test split |
| Streamlit | Interactive dashboard |
| Pandas / NumPy | Data manipulation and preprocessing |
| Matplotlib | Data visualization |

---

# 🚀 Getting Started

## Step 1 — Install Dependencies

```bash
pip install datasets sentence-transformers faiss-cpu transformers keybert spacy tensorflow scikit-learn pandas numpy matplotlib streamlit
```

Install the required spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

## Step 2 — Execute the Pipeline

Run the following command only once to build all project artifacts:

```bash
python run_pipeline.py
```

During execution, the pipeline performs the following tasks:

- Downloads approximately **2,000 ArXiv Machine Learning papers**
- Generates sentence embeddings
- Builds the FAISS vector index
- Creates paper summaries
- Extracts keywords and keyphrases
- Performs Named Entity Recognition
- Trains the LSTM topic classifier (3 epochs)
- Saves every generated artifact inside the `data/` directory

> **Estimated first-time runtime:** 10–15 minutes. Previously generated outputs are automatically reused on future runs.

---

## Step 3 — Launch the Web Dashboard

```bash
streamlit run app.py --server.port 8502
```

Once the server starts, open:

```
http://localhost:8502
```

in your web browser.

---

# 📊 Dashboard Modules

| Module | Description |
|---------|-------------|
| 🔍 Search Papers | Search research papers semantically and view summaries, extracted keywords, detected entities, and predicted topics |
| 📊 Topic Trends | Explore topic distributions using interactive bar charts and pie charts |
| 🧠 Classify Paper | Enter any research abstract to predict its topic along with confidence score |
| 📄 Final Report | Browse and download the generated intelligence report in CSV format |

---

# 📈 Model Details

| Component | Specification |
|------------|---------------|
| Embedding Model | `all-MiniLM-L6-v2` (384-dimensional sentence embeddings) |
| Summarization Model | `t5-small` Transformer (~240 MB, optimized for CPU inference) |
| Classification Model | Single-layer LSTM (32 units), trained for 3 epochs, achieving approximately **70.7% validation accuracy** using TF-IDF + KMeans labels |
| Named Entity Recognition | spaCy `en_core_web_sm` supporting PERSON, ORG, GPE and LOC entities |

---

# ⚡ System Requirements

| Requirement | Minimum Specification |
|--------------|----------------------|
| RAM | 6 GB |
| Storage | Approximately 2 GB |
| Python Version | Python 3.9 or above |
| GPU | Not required (Runs efficiently on CPU) |

---

**Dataset:** [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)

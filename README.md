# Social Media Sentiment Analysis Dashboard

An enterprise-grade, NLP-powered dashboard for real-time and batch sentiment analysis of social media text. Built with Python 3.10+, PyTorch, Hugging Face Transformers, and Streamlit.

## 🚀 Overview

This project provides a complete pipeline for decoding emotions in social media content. It combines the power of a fine-tuned **RoBERTa** model with a modern, interactive web interface.

### Key Features
- **Live Analysis**: Instantly analyze a single comment or tweet with detailed confidence scoring and visualization.
- **Batch Processing**: Upload CSV datasets to process thousands of records at once, complete with distribution analytics and exportable results.
- **Robust Preprocessing**: Custom cleaning engine designed specifically for social media (mentions, hashtags, emojis, and URLs).
- **Production Ready**: Fully containerized with Docker and optimized with Pydantic-based configuration and rotating logs.

---

## 🏗️ Project Structure

```text
.
├── config/
│   ├── config.yaml          # Static configuration (paths, model names, hyperparams)
│   └── settings.py          # Pydantic Settings & directory auto-initialization
├── src/
│   ├── app/
│   │   └── app.py           # Streamlit Dashboard UI
│   ├── data/
│   │   └── preprocess.py    # Social media cleaning & vectorised processing
│   └── models/
│       ├── train.py         # Model fine-tuning & evaluation scripts
│       └── predict.py       # GPU-accelerated inference pipeline
├── utils/
│   └── logger.py            # Professional rotating logging utility
├── data/                    # (Auto-generated) Dataset storage
├── logs/                    # (Auto-generated) Application logs
├── saved_model/             # (Auto-generated) Model weights storage
├── Dockerfile               # Production container definition
├── docker-compose.yml       # Service orchestration & volume mapping
├── Makefile                 # Simplified CLI commands
└── requirements.txt         # Project dependencies
```

---

## 🛠️ Installation & Setup

### Option 1: Using Docker (Recommended)
The project is containerized to ensure consistency and handle heavy dependencies like PyTorch automatically.

1. **Build and Run**:
   ```bash
   make build
   make up
   ```
2. **Access the App**: Open [http://localhost:8501](http://localhost:8501) in your browser.

### Option 2: Local Installation
If you prefer to run locally without Docker:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch Streamlit**:
   ```bash
   python -m streamlit run src/app/app.py
   ```

---

## ⚙️ Configuration

Static settings are managed in `config/config.yaml`. You can modify:
- **Model Name**: Change the base Hugging Face model.
- **Paths**: Customize where data, logs, and models are stored.
- **Inference Params**: Adjust `max_length` and `batch_size` for performance tuning.

---

## 📊 Usage Guide

### 1. Live Analysis
- Navigate to the **Live Analysis** tab in the sidebar.
- Paste any social media comment.
- Click **Analyze** to view the sentiment class (Positive, Negative, Neutral) and the confidence bar chart.

### 2. Batch Processing
- Navigate to the **Batch Processing** tab.
- Upload a CSV file (ensure it contains a text column).
- Select your column and click **Run Batch Analysis**.
- View the overall sentiment distribution (Pie Chart) and download the annotated CSV.

---

## 🛡️ Best Practices Implemented
- **Vectorization**: Uses list comprehensions for high-performance string manipulation in Dataframes.
- **Caching**: Employs `@st.cache_resource` to keep the Transformer model in memory, preventing lag.
- **Error Handling**: Robust try/except blocks in the inference and batch pipelines.
- **Logging**: Rotating file handlers to ensure logs don't consume excessive disk space.

---

## 📜 License
Internal Project - Senior Machine Learning Engineering Team.

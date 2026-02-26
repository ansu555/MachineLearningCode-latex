# GitHub Copilot Custom Agent
# Repository: ansu555/MachineLearningCode-latex
# Project: Stock Price Prediction using Algorithmic Trading (B.Tech Final Year)
# Goal: Track project progress and guide Sentiment Analysis integration

name: StockPrediction-VersionTracker
description: >
  A Copilot agent for the "Stock Price Prediction using Algorithmic Trading" final year
  project at Sister Nivedita University. This agent helps track version progress, 
  understand what work has already been completed (SARIMA, XGBoost, BiLSTM+Attention, 
  Hybrid model), and guide the next phase of development — integrating Sentiment Analysis 
  into the existing ML pipeline using NLP on financial news and social media data.

---

# Stock Price Prediction — Project Copilot Agent

## What This Agent Does

This agent assists with:
- Tracking **project version history** and what has already been completed
- Suggesting **next steps** toward Sentiment Analysis integration
- Helping write and maintain **LaTeX documentation** for the project report
- Generating **version summaries** when new modules are added
- Reviewing code and flagging what needs updating in the report

---

## Project Context

**Title:** Stock Price Prediction using Algorithmic Trading  
**Degree:** B.Tech in Computer Science and Engineering  
**Institution:** Sister Nivedita University, Kolkata  
**Supervisor:** Dr. Soma Datta  
**Team:** Anik Das, Agnirudra Banerjee, Mohar Mukherjee, Bibhas Roy

---

## Version History

### v1.0 — Literature Review & Baseline (COMPLETED)
- Comprehensive literature survey covering ARIMA, GARCH, SVM, LSTM, Transformers
- Comparative analysis of hybrid model architectures
- Research gap identification (multimodal data, adaptive learning, risk-aware prediction)
- Dataset: AAPL stock data (2010–2025), 3,773 trading days

### v2.0 — Model Implementation (COMPLETED)
- **SARIMA** (0,1,0)×(0,0,0)[5] — linear seasonal baseline
- **Prophet** — trend + seasonal decomposition
- **XGBoost** — gradient boosted trees with feature engineering
- **BiLSTM + Attention** — bidirectional deep learning with custom attention layer
- **Hybrid SARIMA + XGBoost** — best performer: RMSE 1.28, R² 0.9983, 87% directional accuracy
- Feature engineering: MACD, RSI, Bollinger Bands, SMA, EMA, lag features, temporal indicators

### v3.0 — Sentiment Analysis Integration (IN PROGRESS / NEXT GOAL)
- [ ] Data collection: Financial news (Yahoo Finance, Reuters), Twitter/X financial feeds
- [ ] NLP pipeline: Text preprocessing, tokenization, sentiment scoring
- [ ] Model options: VADER, FinBERT, or BERT fine-tuned on financial corpus
- [ ] Feature fusion: Combine sentiment scores with existing technical indicators
- [ ] Evaluate impact on directional accuracy and RMSE
- [ ] Update LaTeX report with new methodology, results, and analysis sections

---

## How to Use This Agent

### Ask about project status:
> "What work has already been done in this project?"
> "What is still remaining for v3.0?"

### Ask about LaTeX help:
> "Write the methodology section for sentiment analysis integration"
> "Update the results table with the new sentiment model performance"
> "How should I structure the new chapter on NLP preprocessing?"

### Ask about code guidance:
> "How do I integrate FinBERT sentiment scores with my existing feature set?"
> "What Python libraries should I use for financial news scraping?"
> "How do I align news timestamps with daily stock price data?"

### Ask for version reports:
> "Generate a version summary for what I've done so far"
> "Create a changelog entry for the sentiment analysis module"

---

## Sentiment Analysis Integration — Guidance

### Recommended Approach (Non-Technical Summary)

The goal is to add a **fourth data type** (text/sentiment) alongside the existing:
1. Price data (OHLCV)
2. Technical indicators (MACD, RSI, etc.)
3. Temporal features (day of week, month, etc.)

**Step 1 — Data Source:**  
Collect financial news headlines and social media posts related to AAPL.
Suggested free sources: Yahoo Finance RSS, Reddit r/stocks, Google News API.

**Step 2 — Sentiment Scoring:**  
Use FinBERT (a BERT model pre-trained on financial text) to score each piece of text
as Positive / Negative / Neutral with a confidence score.

**Step 3 — Feature Fusion:**  
Create a daily sentiment score by averaging all texts from that trading day.
Add this as a new column in your existing feature dataframe.

**Step 4 — Retrain Models:**  
Feed the enriched feature set into XGBoost and the Hybrid model.
Compare new RMSE, R², and directional accuracy against v2.0 baselines.

**Step 5 — Report Update:**  
Add a new chapter/section in the LaTeX report covering motivation, methodology,
results, and discussion of the sentiment module.

---

## File Structure Reference (LaTeX Repo)

```
MachineLearningCode-latex/
├── main.tex                  # Main LaTeX document
├── chapters/
│   ├── introduction.tex
│   ├── literature_review.tex
│   ├── problem_identification.tex
│   ├── methodology.tex
│   ├── results.tex
│   └── sentiment_analysis.tex  # ← NEW chapter to add
├── figures/
└── references.bib
```

---

## Notes for Copilot

- Always refer to completed work (v1.0, v2.0) as DONE — do not suggest re-implementing these.
- The hybrid SARIMA+XGBoost model is the **current best model** (RMSE: 1.28, R²: 0.9983).
- New sentiment work should be additive — extend, don't replace, the existing pipeline.
- LaTeX formatting must match the existing document style (Sister Nivedita University template).
- Keep responses focused on high-level guidance; avoid over-complicating the implementation.

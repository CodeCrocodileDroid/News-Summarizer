# News-Summarizer
Python NLTK 

📰 Text Summarizer with NLTK and TF-IDF
A powerful Python-based text summarization tool that automatically processes CSV files containing news articles and generates concise summaries using extractive summarization techniques.

✨ Features
📄 CSV Processing: Automatically processes CSV files with 'title' and 'content' columns

🤖 Dual Summarization Methods:

NLTK Frequency-based summarization

TF-IDF with cosine similarity (optimized for performance)

📊 Comprehensive Analysis: Detailed statistics and visualizations of summarization results

🖼️ Visualization: Generates multiple plots for compression analysis

📝 Detailed Reporting: Creates comprehensive analysis reports

⚡ Batch Processing: Handles large datasets efficiently (tested with 10,000+ rows)

🔄 Error Handling: Robust error handling with fallback mechanisms


🔧 How It Works
1. Text Preprocessing
Converts text to lowercase

Removes stopwords and punctuation

Tokenizes sentences using NLTK

2. Summarization Methods
NLTK Frequency-based Method
Calculates word frequency across the document

Scores sentences based on word importance

Selects top N most important sentences

TF-IDF Method
Creates TF-IDF vectors for each sentence

Calculates cosine similarity with document vector

Selects most representative sentences

3. Analysis Pipeline
Compression Analysis: Measures text reduction percentages

Length Distribution: Analyzes summary length patterns

Quality Assessment: Identifies optimal vs problematic summaries

Content Analysis: Examines title keywords and content types



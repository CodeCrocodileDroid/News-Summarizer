import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Download required NLTK data
print("Initializing NLTK...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("✓ NLTK data loaded")
except:
    print("⚠ NLTK download may have issues, but continuing...")


class TextSummarizer:
    def __init__(self, language='english'):
        """Initialize the summarizer with language settings"""
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.punctuation = set(string.punctuation)

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def extractive_summarize_nltk(self, text, num_sentences=3):
        """
        Extractive summarization using NLTK - Frequency-based approach
        """
        if not text or not isinstance(text, str) or len(text.strip()) < 10:
            return ""

        try:
            # Tokenize sentences
            sentences = sent_tokenize(text)

            if len(sentences) <= num_sentences:
                return text

            # Preprocess and tokenize words
            words = word_tokenize(text.lower())

            # Remove stopwords and punctuation
            words = [word for word in words
                     if word not in self.stop_words
                     and word not in self.punctuation]

            if not words:
                return sentences[0] if sentences else ""

            # Calculate word frequency
            freq_dist = FreqDist(words)

            # Score sentences based on word frequency
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                sentence_words = word_tokenize(sentence.lower())
                score = 0
                word_count = 0

                for word in sentence_words:
                    if word not in self.stop_words and word not in self.punctuation:
                        score += freq_dist[word]
                        word_count += 1

                if word_count > 0:
                    sentence_scores[i] = score / word_count

            # Select top N sentences
            if sentence_scores:
                ranked_sentences = sorted(sentence_scores.items(),
                                          key=lambda x: x[1], reverse=True)[:num_sentences]

                # Sort selected sentences by original order
                ranked_sentences = sorted(ranked_sentences, key=lambda x: x[0])

                # Create summary
                summary = ' '.join([sentences[idx] for idx, _ in ranked_sentences])
                return summary
            else:
                # Return first few sentences as fallback
                return ' '.join(sentences[:min(num_sentences, len(sentences))])

        except Exception as e:
            # Return first 200 characters as fallback
            return text[:200] + ("..." if len(text) > 200 else "")

    def extractive_summarize_tfidf_simple(self, text, num_sentences=3):
        """
        Simple TF-IDF summarization without cosine similarity issues
        """
        if not text or not isinstance(text, str) or len(text.strip()) < 10:
            return ""

        try:
            # Tokenize sentences
            sentences = sent_tokenize(text)

            if len(sentences) <= num_sentences:
                return text

            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Calculate sentence scores based on average TF-IDF score
            sentence_scores = []

            # Convert to dense array for calculations
            dense_matrix = tfidf_matrix.toarray()

            for i in range(len(sentences)):
                # Get non-zero TF-IDF scores for this sentence
                sentence_scores_i = dense_matrix[i]
                # Calculate average TF-IDF score for the sentence
                non_zero_scores = sentence_scores_i[sentence_scores_i > 0]
                avg_score = np.mean(non_zero_scores) if len(non_zero_scores) > 0 else 0
                sentence_scores.append((i, avg_score))

            # Select top N sentences
            if any(score > 0 for _, score in sentence_scores):
                ranked_sentences = sorted(sentence_scores,
                                          key=lambda x: x[1], reverse=True)[:num_sentences]

                # Sort selected sentences by original order
                ranked_sentences = sorted(ranked_sentences, key=lambda x: x[0])

                # Create summary
                summary = ' '.join([sentences[idx] for idx, _ in ranked_sentences])
                return summary
            else:
                # Fall back to NLTK method if no scores
                return self.extractive_summarize_nltk(text, num_sentences)

        except Exception as e:
            # Fall back to NLTK method if TF-IDF fails
            return self.extractive_summarize_nltk(text, num_sentences)

    def summarize_content(self, content, method='tfidf_simple', num_sentences=3):
        """
        Summarize content using specified method
        """
        if method == 'nltk':
            return self.extractive_summarize_nltk(content, num_sentences)
        elif method == 'tfidf_simple':
            return self.extractive_summarize_tfidf_simple(content, num_sentences)
        else:
            raise ValueError("Method must be 'nltk' or 'tfidf_simple'")


def summarize_csv_file():
    """
    Hardcoded function to process text_summarizer_data.csv
    Automatically saves output to summarized_text_summarizer_data.csv
    """
    # Hardcoded file paths
    input_file = "text_summarizer_data.csv"
    output_file = "history/summarized_text_summarizer_data.csv"

    # Hardcoded settings
    method = 'tfidf_simple'
    num_sentences = 3

    print("=" * 70)
    print("TEXT SUMMARIZER - AUTOMATIC PROCESSING")
    print("=" * 70)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Method:      {method}")
    print(f"Sentences:   {num_sentences}")
    print("=" * 70)

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"\n❌ ERROR: File '{input_file}' not found!")
        print("\nLooking for CSV files in current directory...")
        current_dir = os.getcwd()
        csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]

        if csv_files:
            print(f"Found CSV files: {csv_files}")
            print(f"Current directory: {current_dir}")
        else:
            print("No CSV files found in current directory.")

        print("\nPlease make sure 'text_summarizer_data.csv' is in the same folder")
        print("as this Python script, or modify the script with the correct path.")
        input("\nPress Enter to exit...")
        return None

    # Load CSV file
    try:
        print(f"\n📂 Loading '{input_file}'...")
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df):,} rows")
        print(f"✓ Columns: {list(df.columns)}")
    except Exception as e:
        print(f"\n❌ ERROR reading CSV file: {e}")
        input("\nPress Enter to exit...")
        return None

    # Check if required columns exist
    required_columns = ['title', 'content']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"\n❌ ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        print("\nThe CSV file must have 'title' and 'content' columns.")
        input("\nPress Enter to exit...")
        return None

    # Initialize summarizer
    print(f"\n🔧 Initializing summarizer ({method} method)...")
    summarizer = TextSummarizer()

    # Add summaries to dataframe
    print(f"\n⚡ Starting summarization...")
    print("-" * 50)

    summaries = []
    error_count = 0
    success_count = 0
    empty_content_count = 0

    for idx, row in df.iterrows():
        # Get content (handle NaN values)
        content = ""
        if 'content' in row and pd.notna(row['content']):
            content = str(row['content'])

        # Check if content is empty
        if not content or len(content.strip()) < 10:
            summaries.append("")
            empty_content_count += 1
            success_count += 1
            continue

        # Generate summary
        try:
            summary = summarizer.summarize_content(
                content,
                method=method,
                num_sentences=num_sentences
            )
            summaries.append(summary)
            success_count += 1
        except Exception as e:
            summaries.append("")
            error_count += 1

            # Only print first few errors
            if error_count <= 5:
                print(f"  Error in row {idx}: {str(e)[:80]}")

        # Print progress every 1000 rows
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,}/{len(df):,} rows...")

    # Add summary column to dataframe
    df['summary'] = summaries

    print("-" * 50)
    print(f"\n✅ PROCESSING COMPLETE!")
    print(f"   Total rows:           {len(df):,}")
    print(f"   Successfully processed: {success_count:,}")
    print(f"   Empty content:        {empty_content_count:,}")
    print(f"   Errors:               {error_count:,}")

    # Calculate statistics
    non_empty_summaries = df[df['summary'].str.len() > 0]
    avg_summary_len = non_empty_summaries['summary'].str.len().mean() if len(non_empty_summaries) > 0 else 0
    avg_content_len = df['content'].astype(str).str.len().mean() if len(df) > 0 else 0

    if avg_content_len > 0 and avg_summary_len > 0:
        compression = (1 - avg_summary_len / avg_content_len) * 100
        print(f"   Avg. compression:      {compression:.1f}%")

    # Display sample results
    print("\n" + "=" * 70)
    print("SAMPLE SUMMARIES (first 5 rows with summaries):")
    print("=" * 70)

    # Find rows with non-empty summaries
    non_empty_indices = df[df['summary'].str.len() > 0].index.tolist()

    if non_empty_indices:
        samples_to_show = min(5, len(non_empty_indices))
        for i in range(samples_to_show):
            idx = non_empty_indices[i]
            print(f"\n📝 Sample {i + 1} (Row {idx}):")
            print("-" * 40)

            # Get title
            title = ""
            if 'title' in df.columns and pd.notna(df.iloc[idx]['title']):
                title = str(df.iloc[idx]['title'])
                if len(title) > 80:
                    title = title[:80] + "..."

            # Get content preview
            content_preview = ""
            if 'content' in df.columns and pd.notna(df.iloc[idx]['content']):
                content = str(df.iloc[idx]['content'])
                content_preview = content[:100] + ("..." if len(content) > 100 else "")

            # Get summary
            summary = df.iloc[idx]['summary']
            if len(summary) > 200:
                summary_display = summary[:200] + "..."
            else:
                summary_display = summary

            print(f"Title:    {title}")
            print(f"Original: {content_preview}")
            print(f"Summary:  {summary_display}")
            print(f"Length:   {len(content_preview)} chars → {len(summary)} chars")

    else:
        print("\n⚠ No summaries generated. Check your input data.")

    # Save to output file
    print("\n" + "=" * 70)
    print("SAVING RESULTS...")
    print("=" * 70)

    try:
        # Save the dataframe
        df.to_csv(output_file, index=False, encoding='utf-8')

        # Get file size
        file_size = os.path.getsize(output_file) / 1024  # KB

        print(f"\n✅ SUCCESS! File saved as: '{output_file}'")
        print(f"   File size: {file_size:.2f} KB")
        print(f"   Total rows saved: {len(df):,}")
        print(f"   Columns in output: {list(df.columns)}")

        # Show first few rows of output
        print(f"\n📄 First few rows of output file:")
        print("-" * 40)
        print(df[['title', 'summary']].head(3).to_string())

    except Exception as e:
        print(f"\n❌ ERROR saving file: {e}")
        # Try alternative save location
        try:
            alt_output = "summarized_output.csv"
            df.to_csv(alt_output, index=False, encoding='utf-8')
            print(f"✅ Saved to alternative location: '{alt_output}'")
        except:
            print("❌ Could not save file anywhere.")
            return None

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)

    # Return the dataframe
    return df


def main():
    """
    Main function - runs the summarizer automatically
    """
    print("\n" + "=" * 70)
    print("AUTOMATIC CSV TEXT SUMMARIZER")
    print("=" * 70)
    print("This script will automatically:")
    print("1. Process 'text_summarizer_data.csv'")
    print("2. Generate summaries using TF-IDF method")
    print("3. Save results to 'summarized_text_summarizer_data.csv'")
    print("=" * 70)

    # Ask for confirmation
    print("\n⚠ WARNING: This will process the entire CSV file.")
    print("   Processing may take several minutes for large files.")

    response = input("\nDo you want to continue? (yes/no): ").strip().lower()

    if response in ['yes', 'y', '']:
        # Run the summarizer
        result_df = summarize_csv_file()

        if result_df is not None:
            # Show completion message
            print("\n🎉 SUMMARY PROCESSING COMPLETE!")
            print("\nNext steps:")
            print("1. Check 'summarized_text_summarizer_data.csv' for results")
            print("2. The new file has 'title', 'content', and 'summary' columns")
            print("3. You can open it in Excel or any spreadsheet program")

        # Wait before exiting
        input("\nPress Enter to exit...")

    else:
        print("\nOperation cancelled.")
        input("Press Enter to exit...")


# Direct execution - no need to call anything
if __name__ == "__main__":
    # Run the summarizer immediately when script starts
    main()
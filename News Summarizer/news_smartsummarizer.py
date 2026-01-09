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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import textstat

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


class SummaryAnalyzer:
    """Class to analyze summarization results"""

    @staticmethod
    def calculate_text_metrics(text):
        """Calculate various text metrics"""
        if not text or not isinstance(text, str):
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0
            }

        # Basic metrics
        char_count = len(text)
        words = word_tokenize(text)
        word_count = len(words)
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)

        # Average metrics
        avg_word_length = np.mean([len(word) for word in words]) if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Readability scores (if textstat is available)
        readability = 0
        try:
            readability = textstat.flesch_reading_ease(text)
        except:
            pass

        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'readability': readability
        }

    @staticmethod
    def calculate_compression_ratio(original, summary):
        """Calculate compression ratio between original and summary"""
        if not original or len(original) == 0:
            return 0

        original_len = len(original)
        summary_len = len(summary) if summary else 0

        if original_len > 0:
            compression = (1 - summary_len / original_len) * 100
            return compression
        return 0

    @staticmethod
    def analyze_dataframe(df):
        """Perform comprehensive analysis on the summarized dataframe"""

        print("\n" + "=" * 80)
        print("SUMMARY ANALYSIS REPORT")
        print("=" * 80)

        # Basic statistics
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   Total rows: {len(df):,}")

        # Count non-empty summaries
        non_empty_summaries = df[df['summary'].str.len() > 0]
        print(f"   Rows with summaries: {len(non_empty_summaries):,} ({len(non_empty_summaries) / len(df) * 100:.1f}%)")

        # Count empty content
        empty_content = df[df['content'].astype(str).str.len() < 10]
        print(f"   Rows with empty/short content: {len(empty_content):,} ({len(empty_content) / len(df) * 100:.1f}%)")

        # Calculate metrics for all rows
        print(f"\n📈 TEXT METRICS ANALYSIS:")

        # For original content
        content_metrics = []
        for content in df['content'].dropna():
            metrics = SummaryAnalyzer.calculate_text_metrics(str(content))
            content_metrics.append(metrics)

        if content_metrics:
            content_df = pd.DataFrame(content_metrics)
            print(f"   Original Content:")
            print(f"     Avg. characters: {content_df['char_count'].mean():.0f}")
            print(f"     Avg. words: {content_df['word_count'].mean():.0f}")
            print(f"     Avg. sentences: {content_df['sentence_count'].mean():.1f}")
            print(f"     Avg. word length: {content_df['avg_word_length'].mean():.1f} chars")
            print(f"     Avg. sentence length: {content_df['avg_sentence_length'].mean():.1f} words")

        # For summaries
        summary_metrics = []
        for summary in non_empty_summaries['summary']:
            metrics = SummaryAnalyzer.calculate_text_metrics(str(summary))
            summary_metrics.append(metrics)

        if summary_metrics:
            summary_df = pd.DataFrame(summary_metrics)
            print(f"\n   Generated Summaries:")
            print(f"     Avg. characters: {summary_df['char_count'].mean():.0f}")
            print(f"     Avg. words: {summary_df['word_count'].mean():.0f}")
            print(f"     Avg. sentences: {summary_df['sentence_count'].mean():.1f}")
            print(f"     Avg. word length: {summary_df['avg_word_length'].mean():.1f} chars")
            print(f"     Avg. sentence length: {summary_df['avg_sentence_length'].mean():.1f} words")

        # Compression analysis
        print(f"\n🎯 COMPRESSION ANALYSIS:")
        compression_ratios = []
        for idx, row in non_empty_summaries.iterrows():
            original = str(row['content']) if pd.notna(row['content']) else ""
            summary = str(row['summary']) if pd.notna(row['summary']) else ""
            compression = SummaryAnalyzer.calculate_compression_ratio(original, summary)
            compression_ratios.append(compression)

        if compression_ratios:
            compression_series = pd.Series(compression_ratios)
            print(f"   Avg. compression: {compression_series.mean():.1f}%")
            print(f"   Min compression: {compression_series.min():.1f}%")
            print(f"   Max compression: {compression_series.max():.1f}%")
            print(f"   Std. deviation: {compression_series.std():.1f}%")

            # Distribution of compression ratios
            print(f"\n   Compression Distribution:")
            bins = [0, 25, 50, 75, 90, 95, 100]
            for i in range(len(bins) - 1):
                count = ((compression_series >= bins[i]) & (compression_series < bins[i + 1])).sum()
                percentage = count / len(compression_series) * 100
                print(f"     {bins[i]:3d}-{bins[i + 1]:3d}%: {count:5,d} rows ({percentage:.1f}%)")

        # Summary length analysis
        print(f"\n📏 SUMMARY LENGTH DISTRIBUTION:")
        summary_lengths = non_empty_summaries['summary'].str.len()

        if len(summary_lengths) > 0:
            bins = [0, 50, 100, 200, 300, 500, 1000, float('inf')]
            bin_labels = ['<50', '50-100', '100-200', '200-300', '300-500', '500-1000', '>1000']

            for i in range(len(bins) - 1):
                if i == len(bins) - 2:  # Last bin
                    count = (summary_lengths >= bins[i]).sum()
                else:
                    count = ((summary_lengths >= bins[i]) & (summary_lengths < bins[i + 1])).sum()

                if count > 0:
                    percentage = count / len(summary_lengths) * 100
                    print(f"     {bin_labels[i]:8s}: {count:5,d} rows ({percentage:.1f}%)")

        # Sample analysis of best and worst summaries
        print(f"\n⭐ SAMPLE ANALYSIS - BEST AND WORST SUMMARIES:")

        if len(non_empty_summaries) > 0:
            # Find summaries with optimal compression (50-80%)
            optimal_mask = (pd.Series(compression_ratios) >= 50) & (pd.Series(compression_ratios) <= 80)
            optimal_summaries = non_empty_summaries[optimal_mask]

            if len(optimal_summaries) > 0:
                print(f"\n   Optimal Summaries (50-80% compression):")
                for idx in optimal_summaries.index[:2]:  # Show 2 examples
                    title = str(optimal_summaries.loc[idx, 'title'])[:60] + "..." if len(
                        str(optimal_summaries.loc[idx, 'title'])) > 60 else str(optimal_summaries.loc[idx, 'title'])
                    compression = compression_ratios[list(non_empty_summaries.index).index(idx)]
                    print(f"     - '{title}'")
                    print(f"       Compression: {compression:.1f}%")

            # Find very short summaries (potential issues)
            short_summaries = non_empty_summaries[non_empty_summaries['summary'].str.len() < 30]
            if len(short_summaries) > 0:
                print(f"\n   ⚠ Very Short Summaries (<30 chars, potential issues): {len(short_summaries):,}")
                for idx in short_summaries.index[:2]:
                    print(f"     - Row {idx}: '{str(short_summaries.loc[idx, 'summary'])}'")

        # Content type analysis based on titles
        print(f"\n🔍 CONTENT TYPE ANALYSIS (from titles):")
        if 'title' in df.columns:
            titles = df['title'].dropna().astype(str)

            # Common keywords in titles
            all_words = []
            for title in titles:
                words = word_tokenize(title.lower())
                words = [w for w in words if w not in stopwords.words('english') and w not in string.punctuation]
                all_words.extend(words)

            if all_words:
                word_freq = Counter(all_words)
                common_words = word_freq.most_common(10)
                print(f"   Most common title keywords:")
                for word, count in common_words:
                    print(f"     '{word}': {count:,}")

        return {
            'total_rows': len(df),
            'summarized_rows': len(non_empty_summaries),
            'avg_compression': np.mean(compression_ratios) if compression_ratios else 0,
            'summary_metrics': summary_df.mean().to_dict() if 'summary_df' in locals() else {}
        }

    @staticmethod
    def generate_visualizations(df, output_dir="analysis_plots"):
        """Generate visualization plots for the analysis"""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Filter non-empty summaries
        non_empty_summaries = df[df['summary'].str.len() > 0]

        if len(non_empty_summaries) == 0:
            print("⚠ No summaries to visualize")
            return

        # 1. Compression Ratio Distribution
        plt.figure(figsize=(12, 10))

        # Calculate compression ratios
        compression_ratios = []
        for idx, row in non_empty_summaries.iterrows():
            original = str(row['content']) if pd.notna(row['content']) else ""
            summary = str(row['summary']) if pd.notna(row['summary']) else ""
            compression = SummaryAnalyzer.calculate_compression_ratio(original, summary)
            compression_ratios.append(compression)

        plt.subplot(2, 2, 1)
        plt.hist(compression_ratios, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Compression Ratio (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Compression Ratios')
        plt.grid(True, alpha=0.3)

        # 2. Summary Length Distribution
        plt.subplot(2, 2, 2)
        summary_lengths = non_empty_summaries['summary'].str.len()
        plt.hist(summary_lengths, bins=30, edgecolor='black', alpha=0.7, color='green')
        plt.xlabel('Summary Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Summary Lengths')
        plt.grid(True, alpha=0.3)

        # 3. Original vs Summary Length Scatter
        plt.subplot(2, 2, 3)
        original_lengths = non_empty_summaries['content'].astype(str).str.len()
        plt.scatter(original_lengths, summary_lengths, alpha=0.5, s=10)
        plt.xlabel('Original Text Length')
        plt.ylabel('Summary Length')
        plt.title('Original vs Summary Length')
        plt.grid(True, alpha=0.3)

        # Add trend line
        if len(original_lengths) > 1 and len(summary_lengths) > 1:
            z = np.polyfit(original_lengths, summary_lengths, 1)
            p = np.poly1d(z)
            plt.plot(original_lengths, p(original_lengths), "r--", alpha=0.8)

        # 4. Compression vs Original Length
        plt.subplot(2, 2, 4)
        plt.scatter(original_lengths, compression_ratios, alpha=0.5, s=10, color='purple')
        plt.xlabel('Original Text Length')
        plt.ylabel('Compression Ratio (%)')
        plt.title('Compression Ratio vs Original Length')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/summary_analysis.png", dpi=150, bbox_inches='tight')
        print(f"✓ Visualizations saved to {output_dir}/summary_analysis.png")
        plt.show()


def summarize_csv_file():
    """
    Hardcoded function to process text_summarizer_data.csv
    Automatically saves output to summarized_text_summarizer_data.csv
    """
    # Hardcoded file paths
    input_file = "text_summarizer_data.csv"
    output_file = "summarized_text_summarizer_data.csv"

    # Hardcoded settings
    method = 'tfidf_simple'
    num_sentences = 3

    print("=" * 80)
    print("TEXT SUMMARIZER - AUTOMATIC PROCESSING WITH ANALYSIS")
    print("=" * 80)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Method:      {method}")
    print(f"Sentences:   {num_sentences}")
    print("=" * 80)

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
    print(f"\n✅ SUMMARIZATION COMPLETE!")
    print(f"   Total rows:           {len(df):,}")
    print(f"   Successfully processed: {success_count:,}")
    print(f"   Empty content:        {empty_content_count:,}")
    print(f"   Errors:               {error_count:,}")

    # Run comprehensive analysis
    print("\n" + "=" * 80)
    print("RUNNING COMPREHENSIVE ANALYSIS...")
    print("=" * 80)

    analyzer = SummaryAnalyzer()
    analysis_results = analyzer.analyze_dataframe(df)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)

    try:
        analyzer.generate_visualizations(df, output_dir="summary_analysis")
    except Exception as e:
        print(f"⚠ Could not generate visualizations: {e}")
        print("   (This might be due to missing matplotlib or display issues)")

    # Save to output file
    print("\n" + "=" * 80)
    print("SAVING RESULTS...")
    print("=" * 80)

    try:
        # Save the dataframe
        df.to_csv(output_file, index=False, encoding='utf-8')

        # Get file size
        file_size = os.path.getsize(output_file) / 1024  # KB

        print(f"\n✅ SUCCESS! File saved as: '{output_file}'")
        print(f"   File size: {file_size:.2f} KB")
        print(f"   Total rows saved: {len(df):,}")
        print(f"   Columns in output: {list(df.columns)}")

        # Save analysis report to text file
        report_file = "summary_analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SUMMARY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Input File: {input_file}\n")
            f.write(f"Output File: {output_file}\n")
            f.write(f"Processing Method: {method}\n")
            f.write(f"Target Sentences: {num_sentences}\n\n")

            f.write(f"Total Rows Processed: {len(df):,}\n")
            f.write(f"Rows with Summaries: {analysis_results['summarized_rows']:,} ")
            f.write(f"({analysis_results['summarized_rows'] / len(df) * 100:.1f}%)\n")
            f.write(f"Average Compression: {analysis_results['avg_compression']:.1f}%\n\n")

            # Add key metrics
            if analysis_results['summary_metrics']:
                f.write("Summary Text Metrics:\n")
                for key, value in analysis_results['summary_metrics'].items():
                    f.write(f"  {key}: {value:.2f}\n")

            # Add sample summaries
            f.write("\n" + "=" * 80 + "\n")
            f.write("SAMPLE SUMMARIES\n")
            f.write("=" * 80 + "\n\n")

            non_empty = df[df['summary'].str.len() > 0]
            for i in range(min(5, len(non_empty))):
                idx = non_empty.index[i]
                f.write(f"Sample {i + 1} (Row {idx}):\n")
                f.write(f"Title: {str(df.iloc[idx]['title'])[:100]}\n")
                f.write(f"Original Length: {len(str(df.iloc[idx]['content']))} chars\n")
                f.write(f"Summary Length: {len(df.iloc[idx]['summary'])} chars\n")
                f.write(f"Summary: {df.iloc[idx]['summary'][:200]}\n")
                f.write("-" * 60 + "\n\n")

        print(f"✓ Analysis report saved to: '{report_file}'")

    except Exception as e:
        print(f"\n❌ ERROR saving files: {e}")
        # Try alternative save location
        try:
            alt_output = "summarized_output.csv"
            df.to_csv(alt_output, index=False, encoding='utf-8')
            print(f"✅ Saved to alternative location: '{alt_output}'")
        except:
            print("❌ Could not save file anywhere.")
            return None

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)
    print("\n📋 OUTPUT FILES CREATED:")
    print(f"   1. {output_file} - Summarized data with new 'summary' column")
    print(f"   2. summary_analysis_report.txt - Detailed analysis report")
    print(f"   3. summary_analysis/summary_analysis.png - Visualization plots")

    # Return the dataframe
    return df, analysis_results


def main():
    """
    Main function - runs the summarizer automatically
    """
    print("\n" + "=" * 80)
    print("AUTOMATIC CSV TEXT SUMMARIZER WITH ANALYSIS")
    print("=" * 80)
    print("This script will automatically:")
    print("1. Process 'text_summarizer_data.csv'")
    print("2. Generate summaries using TF-IDF method")
    print("3. Perform comprehensive analysis of results")
    print("4. Create visualizations and detailed report")
    print("5. Save results to multiple output files")
    print("=" * 80)

    # Ask for confirmation
    print("\n⚠ WARNING: This will process the entire CSV file.")
    print("   Processing may take several minutes for large files.")
    print("   Analysis includes statistics and visualization generation.")

    response = input("\nDo you want to continue? (yes/no): ").strip().lower()

    if response in ['yes', 'y', '']:
        # Run the summarizer with analysis
        result = summarize_csv_file()

        if result is not None:
            df, analysis_results = result

            # Show completion message with insights
            print("\n🎉 PROCESSING AND ANALYSIS COMPLETE!")
            print("\n📊 KEY INSIGHTS:")
            print(
                f"   • {analysis_results['summarized_rows']:,} out of {analysis_results['total_rows']:,} rows were summarized")
            print(f"   • Average compression: {analysis_results['avg_compression']:.1f}%")

            if analysis_results['summary_metrics']:
                avg_words = analysis_results['summary_metrics'].get('word_count', 0)
                avg_sentences = analysis_results['summary_metrics'].get('sentence_count', 0)
                print(f"   • Average summary: {avg_words:.0f} words, {avg_sentences:.1f} sentences")

            print("\n📁 OUTPUT FILES:")
            print("   • summarized_text_summarizer_data.csv - Main output with summaries")
            print("   • summary_analysis_report.txt - Detailed analysis")
            print("   • summary_analysis/summary_analysis.png - Visualizations")

        # Wait before exiting
        input("\nPress Enter to exit...")

    else:
        print("\nOperation cancelled.")
        input("Press Enter to exit...")


# Install required packages if missing
def check_dependencies():
    """Check and install required packages"""
    required_packages = ['matplotlib', 'seaborn', 'textstat']

    print("Checking dependencies...")

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"⚠ {package} not found. Analysis features will be limited.")
            print(f"   Install with: pip install {package}")


# Run dependency check
check_dependencies()

# Direct execution
if __name__ == "__main__":
    # Run the summarizer immediately when script starts
    main()
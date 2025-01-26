import pandas as pd
from transformers import pipeline

# Load the pre-trained RoBERTa model for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiment(review_text):
    """
    Classifies sentiment as Positive, Neutral, or Negative.
    Args:
        review_text (str): Input review.
    Returns:
        str: Sentiment label (Positive, Negative, or Neutral).
    """
    if not isinstance(review_text, str) or not review_text.strip():
        return "No Text"  # Handle empty or invalid text

    # Perform sentiment analysis
    result = sentiment_pipeline(review_text)[0]
    label = result["label"]

    # Map labels to human-readable format
    sentiment_map = {
        "LABEL_0": "Negative 😡",
        "LABEL_1": "Neutral 😐",
        "LABEL_2": "Positive 😊"
    }
    
    return sentiment_map.get(label, "Unknown")

def process_csv(input_csv, output_csv):
    """
    Reads a CSV file with reviews, analyzes sentiment, and writes results to a new CSV.
    Args:
        input_csv (str): Path to input CSV file.
        output_csv (str): Path to output CSV file.
    """
    # Load CSV file
    df = pd.read_csv(input_csv)

    # Ensure the column containing reviews exists (adjust column name as needed)
    review_column = "review"  # Change this if your CSV has a different column name
    if review_column not in df.columns:
        print(f"CSV file must contain a column named '{review_column}'")
        return

    # Process each review and get sentiment analysis results
    df["Sentiment"] = df[review_column].apply(analyze_sentiment)

    # Save the updated CSV file with sentiment results
    df.to_csv(output_csv, index=False)
    
    print(f"Processed {len(df)} reviews. Results saved to '{output_csv}'")

# Example Usage
process_csv("input_reviews.csv", "output_sentiments.csv") 
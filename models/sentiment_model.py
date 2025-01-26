from transformers import pipeline

# Load a more powerful pre-trained sentiment analysis model (roberta-large)
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiment(review_text):
    """
    Classifies sentiment as Positive, Neutral, or Negative.
    Args:
        review_text (str): Input review.
    Returns:
        str: Sentiment category.
    """
    if not review_text.strip():
        return "No review text provided!"

    # Perform sentiment analysis
    result = sentiment_pipeline(review_text)[0]
    label = result["label"]  # Example: "LABEL_2" -> negative, positive, or neutral based on model

    # We use sentiment classification values (you may need to adjust based on specific model)
    if label == 'LABEL_0':  # negative
        sentiment_label = "Negative 😡"
    elif label == 'LABEL_1':  # neutral
        sentiment_label = "Neutral 😐"
    else:  # positive (LABEL_2 or whatever is positive in your specific model)
        sentiment_label = "Positive 😊"

    return sentiment_label

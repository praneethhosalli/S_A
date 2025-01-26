
import pandas as pd
import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

    # Map sentiment based on the model's result
    sentiment_map = {
        "LABEL_0": "Negative 😡",   # Negative sentiment
        "LABEL_1": "Neutral 😐",    # Neutral sentiment
        "LABEL_2": "Positive 😊"    # Positive sentiment
    }

    # Return the mapped sentiment or "Unknown" if the label is unexpected
    label = result["label"]
    return sentiment_map.get(label, "Unknown")

def process_csv(df):
    """
    Analyzes sentiment for each review in the dataframe and returns the sentiment results.
    Args:
        df (pandas.DataFrame): Dataframe containing reviews.
    Returns:
        pandas.DataFrame: Dataframe with an additional 'Sentiment' column.
    """
    # Ensure the column containing reviews exists
    review_column = "review"  # Change this if your CSV has a different column name
    if review_column not in df.columns:
        st.error(f"CSV file must contain a column named '{review_column}'")
        return None

    # Process each review and get sentiment analysis results
    df["Sentiment"] = df[review_column].apply(analyze_sentiment)
    return df

def plot_sentiment_distribution(df):
    """
    Plot a pie chart to show the distribution of sentiments in the reviews.
    Args:
        df (pandas.DataFrame): DataFrame with the sentiment column.
    """
    # Count the occurrences of each sentiment
    sentiment_counts = df["Sentiment"].value_counts()

    # Plot the pie chart aiuchhcvdjcgvdhg
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=["#ff9999","#66b3ff", "#99ff99"])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)  # Display the pie chart in Streamlit

def generate_word_cloud(df, sentiment):
    """
    Generates a word cloud of frequently used words in reviews for a given sentiment.
    Args:
        df (pandas.DataFrame): Dataframe with reviews.
        sentiment (str): Sentiment type ('Positive', 'Negative', 'Neutral').
    """
    # Filter reviews based on sentiment
    filtered_reviews = df[df['Sentiment'] == sentiment]
    all_reviews = " ".join(filtered_reviews["review"].dropna())

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

    # Display the word cloud using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)  # Display the word cloud

# Streamlit app layout
st.title('Review Sentiment Analysis')

# Create tabs for different functionality
tabs = st.selectbox("Choose an option", ["Single Review", "Batch Processing"])

# Single Review Sentiment Analysis
if tabs == "Single Review":
    st.header("Enter your review for sentiment analysis")

    review_input = st.text_area("Enter review", "Type your review here...")
    
    if st.button("Analyze Sentiment"):
        sentiment = analyze_sentiment(review_input)
        st.write(f"Sentiment: {sentiment}")

# Batch Processing for CSV files
elif tabs == "Batch Processing":
    st.header("Upload your CSV file for batch sentiment analysis")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        st.write("Data preview:")
        st.dataframe(df.head())  # Show a preview of the data

        # Process the CSV to analyze sentiment
        processed_df = process_csv(df)

        if processed_df is not None:
            st.write("Processed results:")
            st.dataframe(processed_df)  # Show the processed results with sentiment

            # Show the sentiment distribution as a pie chart
            st.write("Sentiment Distribution")
            plot_sentiment_distribution(processed_df)

            # Separate reviews based on sentiment
            positive_reviews = processed_df[processed_df["Sentiment"] == "Positive 😊"]
            neutral_reviews = processed_df[processed_df["Sentiment"] == "Neutral 😐"]
            negative_reviews = processed_df[processed_df["Sentiment"] == "Negative 😡"]

            # Allow the user to download each set of reviews as a separate CSV file
            st.write("Download the sentiment results as CSV files:")

            # Positive reviews download
            positive_file = "positive_reviews.csv"
            positive_reviews.to_csv(positive_file, index=False)
            with open(positive_file, "rb") as f:
                st.download_button(
                    label="Download Positive Reviews",
                    data=f,
                    file_name=positive_file,
                    mime="text/csv"
                )

            # Neutral reviews download
            neutral_file = "neutral_reviews.csv"
            neutral_reviews.to_csv(neutral_file, index=False)
            with open(neutral_file, "rb") as f:
                st.download_button(
                    label="Download Neutral Reviews",
                    data=f,
                    file_name=neutral_file,
                    mime="text/csv"
                )

            # Negative reviews download
            negative_file = "negative_reviews.csv"
            negative_reviews.to_csv(negative_file, index=False)
            with open(negative_file, "rb") as f:
                st.download_button(
                    label="Download Negative Reviews",
                    data=f,
                    file_name=negative_file,
                    mime="text/csv"
                )

            # Generate and display word clouds for each sentiment type
            st.write("Word Clouds for Each Sentiment")

            # Word cloud for Positive reviews
            st.write("Positive Reviews Word Cloud")
            generate_word_cloud(processed_df, "Positive 😊")

            # Word cloud for Neutral reviews
            st.write("Neutral Reviews Word Cloud")
            generate_word_cloud(processed_df, "Neutral 😐")

            # Word cloud for Negative reviews
            st.write("Negative Reviews Word Cloud")
            generate_word_cloud(processed_df, "Negative 😡")

# sentiment_analysis.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
from transformers import pipeline
import nltk

# Ensure that NLTK resources are downloaded
nltk.download('punkt')

# Function to scrape news articles
def scrape_news(url):
    """Scrape news articles from a given URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    headlines = [headline.get_text() for headline in soup.find_all('h2')]
    return headlines

# Function to preprocess text data
def preprocess_text(text):
    """Preprocess the text data by removing unnecessary characters and lowercasing."""
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text

# Function to perform sentiment analysis using TextBlob
def sentiment_analysis_textblob(texts):
    """Analyze sentiment of texts using TextBlob."""
    results = []
    for text in texts:
        analysis = TextBlob(text)
        results.append({
            'text': text,
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        })
    return results

# Function to perform sentiment analysis using Hugging Face BERT model
def sentiment_analysis_bert(texts):
    """Analyze sentiment of texts using a BERT model."""
    classifier = pipeline('sentiment-analysis')
    results = classifier(texts)
    return results

# Main function
def main():
    """Main function to run sentiment analysis on news data."""
    # URL of news website to scrape
    url = 'https://news.ycombinator.com/'  # Example URL, change as needed
    
    # Scrape news headlines
    headlines = scrape_news(url)
    
    # Preprocess the headlines
    preprocessed_headlines = [preprocess_text(headline) for headline in headlines]
    
    # Perform sentiment analysis using TextBlob
    textblob_results = sentiment_analysis_textblob(preprocessed_headlines)
    
    # Perform sentiment analysis using BERT
    bert_results = sentiment_analysis_bert(preprocessed_headlines)
    
    # Convert results to DataFrame for easy handling
    textblob_df = pd.DataFrame(textblob_results)
    bert_df = pd.DataFrame(bert_results)
    
    # Save results to CSV files
    textblob_df.to_csv('textblob_sentiment_results.csv', index=False)
    bert_df.to_csv('bert_sentiment_results.csv', index=False)
    
    print("Sentiment analysis completed and results saved to CSV files.")

# Entry point
if __name__ == "__main__":
    main()

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import geopandas as gpd  # For geospatial data processing
from textblob import TextBlob  # For basic sentiment analysis (news data)
import datetime as dt

# Step 1: Data Acquisition

# Step 1.1: Load the Rice Weather Data from CSV (Combined Weather and Crop Data)
file_path = '/kaggle/input/crops-recommendations/combined_rice_weather_data.csv'
rice_weather_data = pd.read_csv(file_path)

# Preview the data
print("Rice Weather Data:\n", rice_weather_data.head())

# Step 1.2: Load Geospatial Satellite Data (Example)
# Assuming geospatial data in GeoJSON format (adjust as per your real data)
# Example: regions with geospatial boundaries and relevant crop data
geo_data_path = '/kaggle/input/geospatial-satellite-data/geo_data.json'
geo_df = gpd.read_file(geo_data_path)

# Preview Geospatial Data
print("Geospatial Satellite Data:\n", geo_df.head())

# Step 1.3: Load Import/Export Data (Example CSV)
# Example: Data with production increase/decrease and import/export metrics
trade_data_path = '/kaggle/input/import-export-data/trade_data.csv'
trade_data = pd.read_csv(trade_data_path)

# Preview Import/Export Data
print("Import/Export Data:\n", trade_data.head())

# Step 1.4: Simulate News Analysis Data (Using Sentiment Analysis from TextBlob)
# Simulated headlines related to crop production (could be from a news API or scraping)
news_headlines = [
    "Good monsoon boosts rice production",
    "Government policies raise export tariffs",
    "Heavy rainfall damages crops in northern regions",
    "Rice production expected to increase by 10% this season",
    "Global demand for rice surges"
]

# Perform sentiment analysis on the news headlines
news_sentiment_scores = [TextBlob(headline).sentiment.polarity for headline in news_headlines]

# Step 2: Data Cleaning

# Drop any unnecessary columns or rows with missing data in rice_weather_data
rice_weather_data.dropna(inplace=True)

# Clean Geospatial Data (optional filtering)
geo_df = geo_df[['region', 'crop_yield', 'geometry']]  # Select important columns

# Step 3: Merging Datasets

# Merge rice weather data with import/export production data
merged_data = pd.merge(rice_weather_data, trade_data, how='inner', on='Date')

# Merging with geospatial data (based on region)
merged_data = pd.merge(merged_data, geo_df[['region', 'crop_yield']], how='left', left_on='Region', right_on='region')

# Add the news sentiment analysis data as an additional feature
merged_data['News_Sentiment'] = np.random.choice(news_sentiment_scores, len(merged_data))

# Step 4: Normalizing/Standardizing the Data
# We normalize the numerical features (excluding categorical/geospatial features)

# Selecting numerical columns for normalization
numerical_features = ['Price', 'Temperature', 'Humidity', 'Rainfall', 'Sentiment', 'Import', 'Export', 'crop_yield']

# Normalizing using MinMaxScaler
scaler = MinMaxScaler()
merged_data[numerical_features] = scaler.fit_transform(merged_data[numerical_features])

# Step 5: Splitting the Data

# Let's split into 70% training, 15% validation, and 15% test sets
train_data, test_data = train_test_split(merged_data, test_size=0.3, shuffle=False)
val_data, test_data = train_test_split(test_data, test_size=0.5, shuffle=False)

print("\nData after splitting:")
print("Train set:", len(train_data))
print("Validation set:", len(val_data))
print("Test set:", len(test_data))

# Step 6: Visualization of Preprocessed Data (Optional)
plt.figure(figsize=(12,6))

# Plotting Price Trend
plt.subplot(2, 1, 1)
plt.plot(train_data['Date'], train_data['Price'], color='blue', label='Training Data')
plt.plot(val_data['Date'], val_data['Price'], color='orange', label='Validation Data')
plt.plot(test_data['Date'], test_data['Price'], color='green', label='Test Data')
plt.title("Price Trend Over Time")
plt.ylabel("Normalized Price")
plt.legend()

# Plotting Temperature Trend
plt.subplot(2, 1, 2)
plt.plot(train_data['Date'], train_data['Temperature'], color='red', label='Training Data')
plt.plot(val_data['Date'], val_data['Temperature'], color='orange', label='Validation Data')
plt.plot(test_data['Date'], test_data['Temperature'], color='green', label='Test Data')
plt.title("Temperature Trend Over Time")
plt.ylabel("Normalized Temperature")
plt.legend()

plt.tight_layout()
plt.show()

# Step 7: Save preprocessed data (Optional)
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("\nPreprocessed data saved to CSV files.")

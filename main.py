import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

ticker = 'BTC-USD'
interval = '1h'

start_date = '2023-01-01'
end_date = '2025-11-07'

data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

data = data[['Close']]

print(data.head())
print(data.tail())

print(data.shape)
print(data.index.min(), data.index.max())

print(data.isnull().sum())

data.fillna(method='ffill', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

sequence_length = 1440

X_train = []
y_train = []

for i in range(sequence_length, len(scaled_data)):
    X_train.append(scaled_data[i-sequence_length:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1)

test_data = scaled_data[-sequence_length:]

X_test = []
X_test.append(test_data)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_price = model.predict(X_test)

predicted_price = scaler.inverse_transform(predicted_price)
print("Predicted price for the next hour:", predicted_price[0][0])

import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

reddit = praw.Reddit(client_id='hsEVf1uk1FBif_erPJ_QJQ',
                     client_secret='JCC7jKUgyX5LoQ6SmIHY1__QKm2S_w',
                     user_agent='RevolutionForward300 ', check_for_async=False)

analyzer = SentimentIntensityAnalyzer()

analyzer.lexicon.update({
    "bullish": 2.0,
    "bearish": -2.0,
    "HODL": 1.5,
    "FOMO": -1.5,
    "whale": 0.5,
    "dump": -2.0,
    "moon": 1.8
})

subreddit = reddit.subreddit('cryptocurrency')
for submission in subreddit.top('day', limit=100):
    title_sentiment = analyzer.polarity_scores(submission.title)["compound"]
    body_sentiment = analyzer.polarity_scores(submission.selftext)["compound"]

    submission.comments.replace_more(limit=0)
    comment_sentiments = []
    for comment in submission.comments.list():
        comment_sentiment = analyzer.polarity_scores(comment.body)["compound"]
        comment_sentiments.append(comment_sentiment)

    avg_comment_sentiment = sum(comment_sentiments) / len(comment_sentiments) if comment_sentiments else 0

    overall_sentiment = (title_sentiment * 0.4) + (body_sentiment * 0.4) + (avg_comment_sentiment * 0.2)

    totalsentimentdata = []
    totalsentimentdata.append(float(overall_sentiment))
sen = sum(totalsentimentdata)

if sen < 0:
  print(f"Negative Sentiment: {sen}")
elif sen == 0:
  print(f"Neutral Sentiment: {sen}")
else:
  print(f"Positive Sentiment: {sen}")

predictions = model.predict(X_train)
predictions = scaler.inverse_transform(predictions)

plt.figure(figsize=(14, 5))
plt.plot(data.index[sequence_length:], data['Close'][sequence_length:], label='Actual Price')
plt.plot(data.index[sequence_length:], predictions, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
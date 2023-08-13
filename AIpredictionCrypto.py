import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import plotly.graph_objects as go
import talib
from keras.models import load_model
import os
import telebot

# Configurare bot Telegram (înlocuiește "YOUR_BOT_TOKEN" cu token-ul botului tău)
bot = telebot.TeleBot("6511601658:AAGXAZqXdHbtEy60bQScq0c1oTP-34i_s0s")

def get_bitfinex_data(symbol, interval, limit):
    url = f"https://api-pub.bitfinex.com/v2/candles/trade:{interval}:t{symbol}/hist?limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['time', 'open', 'close', 'high', 'low', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df = df.sort_index()
    return df

def calculate_indicators(df):
    df['RSI'] = talib.RSI(df['close'])
    df['MA'] = talib.SMA(df['close'], timeperiod=60)
    macd, signal, _ = talib.MACD(df['close'])
    df['MACD'] = macd - signal
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'])
    return df

def prepare_data(df):
    data = df.filter(['close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data, scaler, dataset

def create_train_test_data(scaled_data, dataset, num_days_prediction):
    train_data = scaled_data[:-num_days_prediction]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_test = train_data[len(train_data)-60:, :]
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (1, x_test.shape[0], 1))
    return x_train, y_train, x_test

def create_model(x_train):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=32))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train, num_epochs):
    model.fit(x_train, y_train, batch_size=32, epochs=num_epochs, validation_split=0.1)
    return model

def load_model_if_exists(model_path):
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    else:
        return None

def save_model(model, model_path):
    model.save(model_path)

def get_future_predictions(model, scaler, scaled_data, num_days_prediction):
    input_data = scaled_data[len(scaled_data)-60:, 0]
    input_data = np.reshape(input_data, (1, 60, 1))
    future_predictions = []
    for _ in range(num_days_prediction):
        prediction = model.predict(input_data)
        prediction = np.squeeze(prediction)
        future_predictions.append(prediction)
        prediction_3d = np.expand_dims(np.expand_dims([prediction], -1), 0)
        input_data = np.append(input_data[:,1:,:], prediction_3d, axis=1)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

def plot_results(df, future_predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=pd.date_range(start=df.index[-1], periods=8)[1:], y=future_predictions.flatten(), mode='lines', name='Prediction'))
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA'], mode='lines', name='MA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], mode='lines', name='BB_upper'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], mode='lines', name='BB_middle'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], mode='lines', name='BB_lower'))
    fig.update_layout(title='Model', xaxis_title='Date', yaxis_title='Close Price USD ($)')
    fig.write_html("crypto_predictions.html")

def select_crypto_symbol(message):
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True)
    markup.add('BTC', 'ETH', 'LTC', 'ADA', 'MATIC', 'DOT')
    msg = bot.send_message(message.chat.id, "Select the cryptocurrency symbol:", reply_markup=markup)
    bot.register_next_step_handler(msg, process_crypto_choice)

def process_crypto_choice(message):
    crypto_symbol = message.text
    df = get_bitfinex_data(crypto_symbol + 'USD', '1D', 1000)
    df = calculate_indicators(df)
    scaled_data, scaler, dataset = prepare_data(df)
    x_train, y_train, x_test = create_train_test_data(scaled_data, dataset, num_days_prediction=7)
    model_path = 'model.h5'
    model = load_model_if_exists(model_path)
    if model is None:
        model = create_model(x_train)
        model = train_model(model, x_train, y_train, num_epochs=100)
        save_model(model, model_path)
    future_predictions = get_future_predictions(model, scaler, scaled_data, num_days_prediction=7)
    plot_results(df, future_predictions)
    bot.send_document(message.chat.id, open("crypto_predictions.html", "rb"), timeout=30)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    select_crypto_symbol(message)

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, "I'm a cryptocurrency predictions bot. Please use the /start command to begin.")

bot.polling()

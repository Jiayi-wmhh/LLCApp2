from urllib import response
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import requests
import re
from requests.structures import CaseInsensitiveDict
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
import json
from datetime import date,timedelta
from dateutil.relativedelta import *
import time
import yfinance as yf
import math
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import datetime
# Create your views here.

def search(request):
    return render(request,"stock/search.html")

def result(request):
    ticker = request.POST.get('ticker')
    ticker = str(ticker).upper()
    tweet_num = 100
    start_date = '2020-01-01'
    epoch_num = 50
    affect_rate = request.POST.get('SArate')
    affect_rate = float(affect_rate)
    print(affect_rate)
    apikey = "&token=c8kjg9qad3ibbdm3takg"
    res = []

    #company detail
    company_base_url = "https://finnhub.io/api/v1/stock/profile2?symbol="
    company_base_url += ticker
    company_base_url += apikey
    resp1 = requests.get(company_base_url).json()
    resp1 = json.dumps(resp1)
    res.append(resp1)

    #company summary
    summary_base_url = "https://finnhub.io/api/v1/quote?symbol="
    summary_base_url += ticker
    summary_base_url += apikey
    resp2 = requests.get(summary_base_url).json()
    resp2 = json.dumps(resp2)
    res.append(resp2)

    #company news

    news_base_url = "https://finnhub.io/api/v1/company-news?symbol="
    news_base_url += ticker
    current_date = date.today().isoformat() 
    days_before = (date.today()-timedelta(days=3)).isoformat()
    news_base_url += "&from=" + days_before +"&to=" + current_date
    news_base_url += apikey
    resp3 = requests.get(news_base_url).json()
    resp3 = json.dumps(resp3)
    res.append(resp3)

    #company chart

    chart_base_url = "https://finnhub.io/api/v1/stock/candle?symbol="
    chart_base_url += ticker
    chart_base_url += "&resolution=D"
    current_date = date.today()
    days_before = current_date + relativedelta(months=-6,days=-1)
    chart_base_url += "&from=" + str(int(time.mktime(days_before.timetuple()))) +"&to=" + str(int(time.mktime(current_date.timetuple())))
    chart_base_url += apikey
    resp4 = requests.get(chart_base_url).json()
    resp4 = json.dumps(resp4)
    res.append(resp4)
    
    # twitter
    num_twitter = tweet_num
    # change bearer token when the company account is created
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAM48aAEAAAAAi%2FLi%2ByJN40pC0y39uAGACG8joMw%3DWSvLjUQofW6rPSZoNaC9QWdJQfXy8EtTWdaUSIFePc4VWyT4mP"
    url = "https://api.twitter.com/2/tweets/search/recent?query="+ticker+"&max_results="+str(num_twitter)
    # add headers information to get request
    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    headers["Authorization"] = "Bearer " + bearer_token


    resp = requests.get(url, headers=headers).json()
    data = resp["data"]
    tweets = []

    for i in range(len(data)):
        parsed_tweet =[]
        parsed_tweet.append(ticker)
        tweet = data[i]["text"]
        parsed_tweet.append(tweet)
        tweets.append(parsed_tweet)

    columns = ['ticker','text']
    df = pd.DataFrame(tweets,columns=columns)
    vader = SentimentIntensityAnalyzer()
    # Iterate through the headlines and get the polarity scores using vader
    scores = df['text'].apply(vader.polarity_scores).tolist()
    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)
    # Join the DataFrames of the news and the list of dicts
    df = df.join(scores_df, rsuffix='_right')
    result = df.to_json(orient = "records")

    weight_sum = 0
    for i in range(df.shape[0]):
        weight_sum += df["compound"][i] / df.shape[0]
    print(weight_sum)

    result_json = json.dumps(result)
    res.append(result_json)

    # finviz

    finwiz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}
    # add required ticker inside tickers list
    tickers = []
    tickers.append(ticker)

    for Ticker in tickers:
        url = finwiz_url + Ticker
        req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
        response = urlopen(req)    
        # Read the contents of the file into 'html'
        html = BeautifulSoup(response)
        # Find 'news-table' in the Soup and load it into 'news_table'
        news_table = html.find(id='news-table')
        # Add the table to our dictionary
        news_tables[Ticker] = news_table


    parsed_news = []

    # Iterate through the news
    for file_name, news_table in news_tables.items():
        # Iterate through all tr tags in 'news_table'
        for x in news_table.findAll('tr'):
            # read the text from each tr tag into text
            # get text from a only
            text = x.a.get_text() 
            # splite text in the td tag into a list 
            date_scrape = x.td.text.split()
            # if the length of 'date_scrape' is 1, load 'time' as the only element

            if len(date_scrape) == 1:
                news_time = date_scrape[0]
                
            # else load 'date' as the 1st element and 'time' as the second    
            else:
                news_date = date_scrape[0]
                news_time = date_scrape[1]
            # Extract the ticker from the file name, get the string up to the 1st '_'  
            ticker = file_name.split('_')[0]
            
            # Append ticker, date, time and headline as a list to the 'parsed_news' list
            parsed_news.append([ticker, news_date, news_time, text])
            
    vader = SentimentIntensityAnalyzer()

    # Set column names
    columns = ['ticker', 'date', 'time', 'text']

    # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
    parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_and_scored_news['text'].apply(vader.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

    # Convert the date column from string to datetime
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
    parsed_and_scored_news = parsed_and_scored_news.to_json(orient = "records")
    result_json = json.dumps(parsed_and_scored_news)
    res.append(result_json)

    
    # LSTM Model Part

    # Read historic stock data from yfinance
    yf_Res = yf.Ticker(ticker)
    df = yf_Res.history(period="10y",start=start_date, interval='1d')
    df.reset_index(inplace=True)
    data_types_dict = {'Date': str}
    df = df.astype(data_types_dict)

    print('Number of rows and columns:', df.shape)


    data_len = df.shape[0]
    split_point = round(data_len * 0.9)
    print('Number of split point',split_point)

    training_set = df.iloc[:data_len, 1:2].values
    # in performance measurement part use code below
    # training_set = df.iloc[:split_point, 1:2].values

    test_set = df.iloc[split_point:, 1:2].values


    # Feature Scaling
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(60, split_point):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    print('Shape of Training set',X_train.shape)



    model = Sequential()
    #Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units = 1))

    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs = epoch_num, batch_size = 32)


    # test the model with data after split point

    dataset_train = df.iloc[:split_point, 1:2]
    dataset_test = df.iloc[split_point:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    '''
    temp = dataset_total.shape[0] - dataset_test.shape[0] - 60
    inputs = dataset_total[temp:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, data_len-split_point+60):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print('Shape of Testing set',X_test.shape)


    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)


    pred_res = []
    for i in range(predicted_stock_price.shape[0]):
        pred_res.append(predicted_stock_price[i][0])
    '''

    #  Perfomance measurement part
    '''
    TPos = 0
    FNeg = 0
    FPos = 0
    TNeg = 0

    for i in range(len(pred_res) - 1):
        act_diff = dataset_test.values[i+1][0] - dataset_test.values[i][0]
        pred_diff = pred_res[i+1] - pred_res[i]
        if act_diff >= 0 and pred_diff >= 0:
            TPos = TPos + 1
        elif act_diff <= 0 and pred_diff >= 0:
            FNeg = FNeg + 1
        elif act_diff >= 0 and pred_diff <= 0:
            FPos = FPos + 1
        else:
            TNeg = TNeg + 1

    precision = TPos / (TPos + FPos)
    recall = TPos / (TPos + FNeg)
    accuracy = (TPos + TNeg)/(TPos + TNeg + FNeg + FPos)
    f_score = 2 * precision * recall / (precision + recall)

    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Accuracy: " + str(accuracy))
    print("F Score: " + str(f_score))


    plt.plot(df.loc[split_point:, 'Date'],dataset_test.values, color = 'red', label = 'Real TESLA Stock Price')
    plt.plot(df.loc[split_point:, 'Date'],pred_res, color = 'blue', label = 'Predicted TESLA Stock Price')
    plt.xticks(np.arange(0,X_test.shape[0],300))
    plt.title('TESLA Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('TESLA Stock Price')
    plt.legend()
    plt.show()

    inputs = dataset_total[dataset_total.shape[0] - 60:].values
    print(inputs)
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    input_pred = []
    input_pred.append(inputs[:60, 0])
    input_pred = np.array(input_pred)
    input_pred = np.reshape(input_pred, (input_pred.shape[0], input_pred.shape[1], 1))
    predicted_stock_price = model.predict(input_pred)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    print(str(ticker) + ":Prediction of Stock Price in future one day is: " + str(predicted_stock_price[0][0]))
    '''
    temp_list = []
    result = []

    inputs = dataset_total[dataset_total.shape[0] - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)


    input_pred = []
    input_pred.append(inputs[:60, 0])
    input_pred = np.array(input_pred)
    function = 1 + (weight_sum * affect_rate)
    input_pred[0][59] = input_pred[0][59] * function
    input_pred = np.reshape(input_pred, (input_pred.shape[0], input_pred.shape[1], 1))
    predicted_stock_price = model.predict(input_pred)
    temp_list.append(predicted_stock_price[0][0])
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    result.append(predicted_stock_price[0][0])


    for i in range(1,7):
        input_pred = []
        input_pred.append(inputs[i:60, 0])
        input_pred[0] = np.append(input_pred,temp_list)
        input_pred = np.array(input_pred)
        input_pred = np.reshape(input_pred, (input_pred.shape[0], input_pred.shape[1], 1))

        predicted_stock_price = model.predict(input_pred)
        temp_list.append(predicted_stock_price[0][0])
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        result.append(predicted_stock_price[0][0])

    print(result)
    res.append(result)

    return render(request,"stock/result.html",{ "Result":res }) 

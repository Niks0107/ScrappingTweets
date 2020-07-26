import GetOldTweets3 as got
import pandas as pd
import numpy as np
import os
import nltk
import  re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

'''
Getting Tweets of Work From Home using GetOldTweets3
'''

def GetData(QueryWord,TweetsCount):  
    data = pd.DataFrame()

    text_query = QueryWord
    count = TweetsCount
    # Creation of query object
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query)\
                                            .setMaxTweets(count)
    # tweetCriteria = got.manager.TweetCriteria().setQuerySearch('Nikhil')\
                                            # .setMaxTweets(10)
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    # Creating list of chosen tweet data
    # user_tweets = [[tweet.date, tweet.text] for tweet in tweets]
    
    for tweet in tweets:
        # user = tweet.username
        text = tweet.text
        TweetDf = pd.DataFrame({'tweet':[text]})
        data = pd.concat([data,TweetDf],ignore_index=True)
            
    return data

data = GetData("india china standoff",10)

def clean_tweet(tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) \
                                    |(\w+:\/\/\S+)", " ", tweet).split()) 

'''
Sentment analysis on tweets using SentimentIntensityAnalyzer from nltk
'''


def GetSentiments(data):    
    data["label"] = 0   
    vader = SentimentIntensityAnalyzer()
    for i in range(data.shape[0]):        
        
        tweet = data['tweet'][i]
        
        polarity = vader.polarity_scores(clean_tweet(tweet))['compound']
        
        if polarity>0.1:
            label = "Positive"
        elif polarity<-0.1:
            label = "Negative"
        else:
            label = "Neutral"
    
        data["label"].iloc[i] = label
    return data

FinalData = GetSentiments(data.copy())

'''
Final Data is the Data with Tweets and its label
'''


   
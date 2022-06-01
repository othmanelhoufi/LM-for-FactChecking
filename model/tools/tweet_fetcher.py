#!/usr/bin/env python3

#This code creates the dataset from Corpus.csv which is downloadable from the
#internet well known dataset which is labeled manually by hand. But for the text
#of tweets you need to fetch them with their IDs.
import tweepy
from twitter_conf import *
import pandas as pd


def fetch_tweets(df, client):
    tweet_ids = df['id'].tolist()
    df_tmp = pd.DataFrame(columns=['id', 'text'])

    batch_size = 100
    for i in range(0, len(tweet_ids), batch_size):
        tweets = client.get_tweets(ids=tweet_ids[i:i+batch_size])
        for tweet in tweets.data:
            text = tweet['text'].split()
            text = " ".join(text)
            dict = {'id': int(tweet['id']), 'text': text}
            df_tmp = df_tmp.append(dict, ignore_index=True)

    result = pd.merge(df, df_tmp, on='id')
    return result

def main():
    #Put your Bearer Token in the parenthesis below
    client = tweepy.Client(bearer_token=bearer_token)

    #ANTiVax Dataset
    df = pd.read_csv('../../dataset/ANTiVax/data_raw.csv')
    res = fetch_tweets(df, client)
    res.to_csv('../../dataset/ANTiVax/data_with_text.csv', index=False)


    # Splitting Dataset to Train & Dev
    
    # df_with_text = pd.read_csv('../../dataset/ANTiVax/data_with_text.csv')
    # print(df['is_misinfo'].value_counts())
    # print(df_with_text['is_misinfo'].value_counts())
    #
    # print('splitting')
    # df_train = df_with_text.sample(frac=0.7, random_state=1)
    # df_dev = df_with_text.drop(df_train.index)
    #
    # print(df_train['is_misinfo'].value_counts())
    # print(df_dev['is_misinfo'].value_counts())
    # df_train.to_csv('../../dataset/ANTiVax/train_raw.csv', index=False)
    # df_dev.to_csv('../../dataset/ANTiVax/dev_raw.csv', index=False)

if __name__ == '__main__':
    main()

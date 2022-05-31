#This code creates the dataset from Corpus.csv which is downloadable from the
#internet well known dataset which is labeled manually by hand. But for the text
#of tweets you need to fetch them with their IDs.
import tweepy

# Twitter Developer keys here
# It is CENSORED


#Put your Bearer Token in the parenthesis below
client = tweepy.Client(bearer_token=bearer_token)

# client = tweepy.Client(
#     consumer_key=api_key,
#     consumer_secret=api_secret,
#     access_token=access_token,
#     access_token_secret=secret_token
# )

print(tweet.text)

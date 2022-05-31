#This code creates the dataset from Corpus.csv which is downloadable from the
#internet well known dataset which is labeled manually by hand. But for the text
#of tweets you need to fetch them with their IDs.
import tweepy

# Twitter Developer keys here
# It is CENSORED
api_key = '6BxZqg16tz0ftGwLXc5Cye2Ti'
api_secret = '1GsAd70iFyHMNnocddXigKeXBsNT2S3yRYgkXTATgmxJ5zTMiZ'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAMecdAEAAAAACUiJT32prMz%2F4aP64sUSyUbFeqE%3DLNTI1BTwLjTpTWIPS1gHECFDCxhsxeIBnVZ5os1JOtugLPAIM6'
access_token = '2211362768-RSmDyTfHpdifbXq4mSboo1N6LgMngjBqT9FWKw8'
secret_token = 'BKuE2b0awHUbNZI6y0Pr0aVfHTn0emlOlLcL7KFTkebID'

#Put your Bearer Token in the parenthesis below
client = tweepy.Client(bearer_token=bearer_token)

# client = tweepy.Client(
#     consumer_key=api_key,
#     consumer_secret=api_secret,
#     access_token=access_token,
#     access_token_secret=secret_token
# )

print(tweet.text)

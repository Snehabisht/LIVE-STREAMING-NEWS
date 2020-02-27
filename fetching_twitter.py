from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s


#consumer key, consumer secret, access token, access secret.
ckey="cOdCCd9pQs4s0vLsr8EWf3t1h"
csecret="RU3aPtmgKNasJR8Fg2H237ecVCD4aC3Ons5HTIpF6NfhEc8nSp"
atoken="1232773024451104768-L1Ntko6aPlQutIaJXpFAhDsftWi1Yb"
asecret="zXYD6opeihNDrJKlbjJAe8Y76myXoTHE3U4MERA7CI6On"

#from twitterapistuff import *

class listener(StreamListener):

    def on_data(self, data):
        all_data=json.loads(data)
        tweet = all_data["text"]
        print(tweet)
        sentiment_value,confidence = s.sentiment(tweet)
        print(tweet, sentiment_value,confidence)
        if confidence*100 >= 80:
            output = open("twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()
        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)  
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])



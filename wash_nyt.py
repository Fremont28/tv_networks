#6/20/18 -comparing the washington post and new york times tweets using 
# term-frequency-inverse document frequency (tfidf) and cosine similarity 
import tweepy 

#washinton post and new york times tweets 
consumer_key="xxxxxxxxxx"
consumer_secret="xxxxxxxxxxxxxxxxx" 
access_token="xxxxxxxxxxxxxxxxxxxx"
access_token_secret="xxxxxxxxxxxxxxxxxxxxxxx" 

auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api=tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)

#tweets (for the washington post/new york times)
news2=[]
for status in tweepy.Cursor(api.user_timeline,screen_name="washingtonpost").items():
    stream=(status._json['text'])
    news2.append(stream)

news2_list=news2.split(' ') 
news_5_list=news5.split(' ') 

#Tfidf vectorizer 
import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer 

vectorizer=TfidfVectorizer()
news2_list=news2.split(' ') #string to list 
matrix=vectorizer.fit_transform(news2_list) 
matrix 

for i, feature in enumerate(vectorizer.get_feature_names()):
    print(i,feature) 

tfidf=TfidfVectorizer(preprocessor= lambda x: x, tokenizer= lambda x:x)
tfidf_matrix=tfidf.fit_transform(news2_list)

#convert list to np array 
news2x=np.asarray(news2)
news5x=np.asarray(news_5_list)
news2x.shape,news5x.shape 
news5x1=news5x[0:3217] 
news5x1.shape 

#convert array to list
news2x1=news2x.tolist() 
news5x11=news5x1.tolist() 

#tfidf-vectorizer 
tfidf1=TfidfVectorizer().fit_transform(news2x1)
tfidf2=TfidfVectorizer().fit_transform(news5x11)

#cosine similarites (after tfidf)
from sklearn.metrics.pairwise import linear_kernel
cos_sim1=linear_kernel(tfidf1,tfidf2).flatten() 
cos_sim1 #cosine similarity score between the new york times and the washington post 



#Cosine Similarity 
#6/12/18 
import re 
import string 

cnn_scroll1=cnn_scroll 
cnn_scroll1=' '.join(cnn_scroll1) #list to string 

#split words by whitespace
cnn_scroll1=cnn_scroll1.split() 

#select words (select strings con alphanumeric characters?) -lists 
cnn_scroll1=re.split(r'\W+',cnn_scroll1) 

#split by whitespace and remove punctuation from each word 
table=str.maketrans('','',string.punctuation)
stripped=[w.translate(table) for w in cnn_scroll1]

#convert words to lowercase 
cnn_scroll1=[word.lower() for word in cnn_scroll1] #list 

#tokenization and clean with nltk 
import nltk 
from nltk import sent_tokenize 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

#filter out punctuation (is.alpha()-removes all tokens that are all non-alphabetic) 
cnn_scroll1=word_tokenize(cnn_scroll1)
#remove all tokens that are not alphabetic 
cnn_scroll1=[w for w in cnn_scroll1 if w.isalpha()]
cnn_scroll1
#filter out stops words 
stop_words=set(stopwords.words('english'))
cnn_scroll1=[w for w in cnn_scroll1 if not w in stop_words]
#stem words 
porter=PorterStemmer()
cnn_scroll1=[porter.stem(w) for w in cnn_scroll1]

#2. apply to the FOX news stream 
fox_scroll1=fox_scroll 
fox_scroll1=' '.join(fox_scroll1)
fox_scroll1=word_tokenize(fox_scroll1)
fox_scroll1=[w for w in fox_scroll1 if w.isalpha() ]
fox_scroll1=[w for w in fox_scroll1 if not w in stop_words]
porter=PorterStemmer() 
fox_scroll1=[porter.stem(w) for w in fox_scroll1]

#vectorize documents for cosine similarity 
vectorizer=CountVectorizer(analyzer='word',min_df=1,
stop_words='english',lowercase=True,token_pattern='[a-zA-Z0-9]{3,}')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

cnn_vectorized=vectorizer.fit_transform(cnn_scroll1)
fox_vectorized=vectorizer.fit_transform(fox_scroll1) 

#cosine simialairty score
cosine_sim_score=cosine_similarity(fox_vectorized1,fox_vectorized1) 
cosine_sim_score.mean() #0.102

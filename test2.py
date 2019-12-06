# Import required libraries
import pandas as pd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

# Import Data
df=pd.read_csv('./small_list.csv')

# Check for null values
df[df.isnull().any(axis=1)]

# Drop rows with null Values
df.drop(df[df.isnull().any(axis=1)].index,inplace=True)

print(df)

# remove_stopwords = True

# # Remove stop words
# if remove_stopwords:
#   stops = set(stopwords.words("english"))
#   words = [w for w in text.lower().split() if not w in stops]
    
# final_text = " ".join(words)

# # Special Characters
# review_text = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", final_text )
# review_text = re.sub(r"\'s", " 's ", final_text )
# review_text = re.sub(r"\'ve", " 've ", final_text )
# review_text = re.sub(r"n\'t", " 't ", final_text )
# review_text = re.sub(r"\'re", " 're ", final_text )
# review_text = re.sub(r"\'d", " 'd ", final_text )
# review_text = re.sub(r"\'ll", " 'll ", final_text )
# review_text = re.sub(r",", " ", final_text )
# review_text = re.sub(r"\.", " ", final_text )
# review_text = re.sub(r"!", " ", final_text )
# review_text = re.sub(r"\(", " ( ", final_text )
# review_text = re.sub(r"\)", " ) ", final_text )
# review_text = re.sub(r"\?", " ", final_text )
# review_text = re.sub(r"\s{2,}", " ", final_text )

# labeled_questions=[]
# labeled_questions.append(TaggedDocument('questions1'[i].split(), df[df.index == i].qid1))
# labeled_questions.append(TaggedDocument('questions2'[i].split(), df[df.index == i].qid2))

# model = Doc2Vec(dm = 1, min_count=1, window=10, size=150, sample=1e-4, negative=10)
# model.build_vocab(labeled_questions)

# # Train the model with 20 epochs 

# for epoch in range(20):
#     model.train(labeled_questions,epochs=model.iter,total_examples=model.corpus_count)
#     print("Epoch #{} is complete.".format(epoch+1))

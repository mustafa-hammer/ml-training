#Sample compare of two statements where they are converted to vectors and then a cosine similarity is generated. 
# From reading, for a model to be remotely accurate it needs at least 50,000 items. So keep that in mind as you see the output.  
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize

model= Doc2Vec.load("compare_d2v.model")

#to find the vector of a document which is not in training data
test_data1 = word_tokenize("How can I increase the speed of my internet connection while using a VPN?".lower())

test_data2 = word_tokenize("How can i increase speed internet connection?".lower())

test_data3 = word_tokenize("Where is London?".lower())

#Compare the same statement to itself. This statement is also in the model. 
print(model.wv.n_similarity(test_data1,test_data1))

#Compare two statements that are slightly different
print(model.wv.n_similarity(test_data1,test_data2))

#Compare two very different statements
print(model.wv.n_similarity(test_data1,test_data3))

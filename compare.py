#Sample compare of two questions where they are converted to vectors and then a cosine similarity is generated. 
# From reading, for a model to be remotely accurate it needs at least 50,000 items. So keep that in mind as you see the output.  
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize

model= Doc2Vec.load("d2v.model")

#to find the vector of a document which is not in training data
test_data = word_tokenize("How can I increase the speed of my internet connection while using a VPN?".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

test_data2 = word_tokenize("increase speed internet connection?".lower())
v2 = model.infer_vector(test_data2)
print("V2_infer", v2)

print(type(v2))

print(model.wv.n_similarity(test_data,test_data2))

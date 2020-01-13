#This script is responsible for generating a model based on a list of questions as input. 

#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd

# Import Data
# you can switch this to "statements_long.csv" but you will need to perform some basic ETL on it as the model will fail to create. 
data=pd.read_csv('./statements_small.csv')

q1_list = data['question1'].values.tolist()

#TODO: cleanup the data by removing nulls, extra characters etcs. The data needs to go through a basic ETL process which im skipping at the moment. 

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(q1_list)]

#print(tagged_data)

max_epochs = 100
vec_size = 20
alpha = 0.025

#TODO: also understand these numbers
model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("compare_d2v.model")
print("Model Saved")

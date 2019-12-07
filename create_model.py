#This script is responsible for generating a model based on a list of questions as input. 

#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd


# Import Data
# you can switch this to "questions.csv" and then rerun to generate a new model. small_list is a subset to speed up development. 
data=pd.read_csv('./small_list.csv')

q1_list = data['question1'].values.tolist()

#TODO: cleanup the data by removing nulls, extra characters etcs. The data needs to go through an ETL process which im skipping at the moment. 

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(q1_list)]

print(tagged_data)

#TODO understand these numbers
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

model.save("d2v.model")
print("Model Saved")

#helpfull links:
# https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
# https://kanoki.org/2019/03/07/sentence-similarity-in-python-using-doc2vec/
# https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4
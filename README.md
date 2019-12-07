# ml-training
Trying stuff out with ML for fun

Main tools: doc2vec (algorithm) and gensim. 

This only runs on python 3 and above. 

Stuff you need to install:
```
pip3.7 install -U gensim
pip3.7 install tokenizer
pip3.7 install nltk
pip3.7 install pandas
pip3.7 install scikit-learn
```

#To run:

## step 1: create model, depending on which csv you use (and specs) this might take 5 min to 1 hour. 
```
python3.7 create_model.py
```

## step 2: use the model to compare two questions
```
python3.7 compare.py
```


Helpful links: 
Gensim: https://radimrehurek.com/gensim/models/doc2vec.html

Tutorial (doesnt work but good pointer): https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5

Another Tutorial: https://kanoki.org/2019/03/07/sentence-similarity-in-python-using-doc2vec/

Data source used for this: https://www.kaggle.com/quora/question-pairs-dataset/data

More examples: https://radimrehurek.com/gensim/auto_examples/ (i need to go through them)
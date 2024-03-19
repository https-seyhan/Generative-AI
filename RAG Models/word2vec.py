from gensim.models import Word2Vec
sentences = [["this", "is", "an", "example", "sentence"], ["another", "example", "sentence"]]
# train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
# access word vectors
vector = model.wv['example']
print(vector)


from gensim.models import Word2Vec
sentences = [["About", "75", "percent", "of", "the", "value", "that", "generative", 
"AI", "use", "cases", "could", "deliver", "falls", "across", "four", "areas:", "Customer", "operations", "marketing", "and", "sales", "software", "engineering", "and", "R&D"], 
["The", "era", "of", "generative", "AI", "is", "beginning"], ["Excitement" ,"technology", "is", "palpable", "pilots", "are", "compelling"]]
# train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
# access word vectors
vector = model.wv['AI']
print(vector)
print(len(vector))


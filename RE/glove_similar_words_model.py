from gensim.models import KeyedVectors
import os

def load_glove_model(file_path):
    """Load a pre-trained GloVe model."""
    return KeyedVectors.load_word2vec_format(file_path, binary=False)

def find_similar_words(model, word, top_n=5):
    """Find similar words to a given word in the GloVe model."""
    try:
        similar_words = model.most_similar(word, topn=top_n)
        return similar_words
    except KeyError:
        return f"'{word}' not found in the vocabulary."

if __name__ == "__main__":
	os.chdir('/home/saul/Desktop/generative-AI/RE')
    # Specify the path to the GloVe file (you can download pre-trained GloVe models from the official website)
    
    #glove_file_path = 'path/to/glove.6B.50d.txt'  # Adjust the path and dimensionality accordingly
	glove_file_path = 'glove.6B.50d.txt'  # Adjust the path and dimensionality accordingly
    # Load the GloVe model
    glove_model = load_glove_model(glove_file_path)

    # Specify the word for which you want to find similar words
    target_word = 'example'

    # Find and print similar words
    similar_words = find_similar_words(glove_model, target_word)
    print(f"Words similar to '{target_word}': {similar_words}")

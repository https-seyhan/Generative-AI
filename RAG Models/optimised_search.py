# Using a set for fast token membership checking
token_set = {"token1", "token2", "token3"}

def is_token_in_set(token):
    return token in token_set

# Example usage
print(is_token_in_set("token1"))  # Output: True


import numpy as np

tokens = np.array(["token1", "token2", "token3"])
search_token = "token2"

def search_vectorized(tokens, search_token):
    return np.any(tokens == search_token)

# Example usage
print(search_vectorized(tokens, search_token))  # Output: True


from ahocorasick import Automaton

def build_automaton(tokens):
    automaton = Automaton()
    for idx, token in enumerate(tokens):
        automaton.add_word(token, (idx, token))
    automaton.make_automaton()
    return automaton

def search_automaton(automaton, text):
    for end_idx, (idx, token) in automaton.iter(text):
        print(f'Found "{token}" at position {end_idx - len(token) + 1}')

tokens = ["he", "she", "his", "hers"]
text = "ahishers"

automaton = build_automaton(tokens)
search_automaton(automaton, text)

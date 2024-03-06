import spacy
import random
from spacy.util import minibatch, compounding
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

from pathlib import Path


class NER_Analysis():
	nlp = spacy.load('en_core_web_md')
	text = """
    "Net income was $9.4 million compared to the prior year of $2.7 million.",
    "Revenue exceeded twelve billion dollars, with a loss of $1b. run",
"""
	def __init__(self):
		print('Init')
		self.get_ner()
		
		
	def get_ner(self):
		print('get Ner')
		ner = self.nlp.create_pipe("ner")
		print(ner)

		doc = self.nlp(self.text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

def set_model():
	print('Model Called!')

if __name__ == '__main__':
	set_model()
	re_ner = NER_Analysis()

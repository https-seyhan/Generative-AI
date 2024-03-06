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
	text = """
	Setting a new benchmark in apartment living, The Rockford presents sophisticated residences in a desirable village setting renowned for its cosmopolitan retail, dining and recreational amenity. With an elevated north-facing position The Rockford offers sun-drenched interiors, stylish open plan living areas, seamless indoor-outdoor flow and large balconies designed for entertaining.
	At the heart of your home, the kitchens in The Rockford are as stylish as they are practical. Luxe stone benchtops, modern SMEG appliances and inventive design come together to create the perfect space for family dinners and celebrations with friends. The Rockford was made for you, for you to make some memories.
	Surrounded by a strong network of public transport infrastructure and flush with an array of schools at all levels of learning, Bexley makes it easy to love where you live with a huge selection of shops, restaurants, cafes, recreational & leisure facilities all at your doorstep.
	Developed by the visionary team at JDC Properties and designed by Loucas Architects, The Rockford is a result of an exciting collaboration between some of Sydney's most experienced and respected names in Property.
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

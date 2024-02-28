#import spacy
#load spacy NLP library
#get text
#chech text label if it is a product
import spacy
nlp = spacy.load("en_core_web_lg")
text = "Apple"
doc = nlp(text)
#print(doc)
#loop through doc
for ent in doc.ents:
    print(ent.label_)
    if ent.label_ == "PRODUCT":
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

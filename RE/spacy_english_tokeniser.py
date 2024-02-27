import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

nlp = spacy.load('en_core_web_md')  # make sure to use larger model!
ruler = EntityRuler(nlp)
patterns = [{"label": "education", "pattern": "book"}]
ruler.add_patterns(patterns)
#nlp.add_pipe(ruler)
tokens = nlp(u'car travel education other')
mytoken = nlp('book')


for token1 in mytoken:
    for token2 in tokens:
        print(token2)
        print((token1.text), token2.text, token1.similarity(token2))

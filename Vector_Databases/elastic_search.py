from elasticsearch import Elasticsearch
from datetime import datetime

# Connect to Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Create an index
es.indices.create(index='my_index', ignore=400)

# Index a document
doc = {
    'author': 'John Doe',
    'text': 'Elasticsearch is a powerful search engine',
    'timestamp': datetime.now(),
}
es.index(index='my_index', id=1, document=doc)

# Refresh the index to make the document searchable immediately
es.indices.refresh(index='my_index')

# Search for the document
query = {'match': {'text': 'search engine'}}
res = es.search(index='my_index', query=query)
for hit in res['hits']['hits']:
    print(hit['_source'])

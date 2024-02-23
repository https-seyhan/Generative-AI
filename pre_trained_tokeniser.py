from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("Category": "Special Purpose Properties",
        "Type": "Mixed-Use Property",
        "Location": "Suburb",
        "Bedrooms": 4,
        "Bathrooms": 2,
        "Price": 123279,
        "Features": [
            "Garage"
        ])
print(tokens)

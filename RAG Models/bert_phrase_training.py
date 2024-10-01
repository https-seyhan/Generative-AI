import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 1. Dataset Preparation
# Load your dataset into a pandas DataFrame
data = {
    'phrase1': ["How are you?", "What is your name?", "Good morning", "How do you do?", "Goodbye"],
    'phrase2': ["How do you do?", "Tell me your name", "Morning", "How are you?", "Farewell"],
    'label': [1, 0, 1, 1, 0]  # 1 for match, 0 for no match
}
df = pd.DataFrame(data)

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 2. Dataset Class
class PhraseMatchingDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        phrase1 = str(self.data.iloc[index, 0])
        phrase2 = str(self.data.iloc[index, 1])
        label = self.data.iloc[index, 2]

        encoding = self.tokenizer(phrase1, phrase2, 
                                  truncation=True, padding='max_length', 
                                  max_length=self.max_len, 
                                  return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 3. Load BERT Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 4. Prepare DataLoader
def create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = PhraseMatchingDataset(df, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set parameters
BATCH_SIZE = 16
MAX_LEN = 128

train_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

# 5. Training Setup
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training function
def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(data_loader)

# 6. Evaluation Function
def eval_model(model, data_loader, device):
    model = model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels

# 7. Train the Model
EPOCHS = 3
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Training loss: {train_loss:.4f}")

# 8. Evaluate the Model
predictions, true_labels = eval_model(model, test_loader, device)

# 9. Compute Metrics
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions)

print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)


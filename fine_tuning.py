import torch
from transformers import BertForQuestionAnswering, BertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from preprocessing import preprocessed_qa_pairs

# Custom Dataset for Question Answering
class QADataset(Dataset):
    def __init__(self, tokenizer, qa_pairs):
        self.tokenizer = tokenizer
        self.qa_pairs = qa_pairs

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question = self.qa_pairs[idx]["question"]
        answer = self.qa_pairs[idx]["answer"]
        context = self.qa_pairs[idx]["context"]

        inputs = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": torch.tensor(self.tokenizer.encode(answer, add_special_tokens=False)[0]),
            "end_positions": torch.tensor(self.tokenizer.encode(answer, add_special_tokens=False)[-1])
        }

def train_model(model, train_dataloader, epochs=300, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} Loss: {loss.item()}")

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name).to(device)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(preprocessed_qa_pairs))
val_size = len(preprocessed_qa_pairs) - train_size
train_dataset, val_dataset = random_split(preprocessed_qa_pairs, [train_size, val_size])

# Create DataLoader for training and validation sets
batch_size = 8
train_dataloader = DataLoader(QADataset(tokenizer, train_dataset), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(QADataset(tokenizer, val_dataset), batch_size=batch_size, shuffle=False)

# Fine-tune the BERT model
train_model(model, train_dataloader, epochs=300, learning_rate=2e-5)

# Save the fine-tuned model
model_save_path = "fine_tuned_bert_model.pth"
torch.save(model.state_dict(), model_save_path)

print("Fine-tuning completed successfully.")

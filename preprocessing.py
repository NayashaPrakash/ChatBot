import json
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# Download NLTK resources (if not already downloaded)
nltk.download("punkt")
nltk.download("stopwords")

def read_squad_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        squad_data = json.load(f)
    return squad_data


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [re.sub(r"[^a-zA-Z0-9]", "", token) for token in tokens]
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token and token not in stop_words]
    preprocessed_text = " ".join(tokens)
    return preprocessed_text


def extract_qa_pairs(squad_data):
    qa_pairs = []
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                if "is_impossible" in qa:
                    # Handle unanswerable questions in SQuAD v2.0
                    if not qa["is_impossible"]:
                        answer = qa["answers"][0]["text"]
                        qa_pairs.append({"question": question, "answer": answer, "context": context})
                else:
                    # For SQuAD v1.1 or datasets without the "is_impossible" key, use the previous approach
                    if qa["answers"]:
                        answer = qa["answers"][0]["text"]
                        qa_pairs.append({"question": question, "answer": answer, "context": context})
    return qa_pairs

def preprocess_qa_pairs(qa_pairs, tokenizer):
    preprocessed_qa_pairs = []
    for qa_pair in tqdm(qa_pairs, desc="Preprocessing QA Pairs", unit="pair"):
        preprocessed_question = preprocess_text(qa_pair["question"])
        preprocessed_answer = preprocess_text(qa_pair["answer"])
        preprocessed_context = preprocess_text(qa_pair["context"])

        preprocessed_qa_pairs.append({
            "question": preprocessed_question,
            "answer": preprocessed_answer,
            "context": preprocessed_context
        })
    return preprocessed_qa_pairs

# Load pre-trained BERT tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Provide the path to the downloaded SQuAD JSON file
squad_file_path = "data/squad_dataset.json"
squad_data = read_squad_dataset(squad_file_path)
qa_pairs = extract_qa_pairs(squad_data)
preprocessed_qa_pairs = preprocess_qa_pairs(qa_pairs, tokenizer)

# Print a sample of preprocessed QA pairs
print("Sample Preprocessed QA Pairs:")
for i in range(min(5, len(preprocessed_qa_pairs))):
    print(f"Question: {preprocessed_qa_pairs[i]['question']}")
    print(f"Answer: {preprocessed_qa_pairs[i]['answer']}")
    print(f"Context: {preprocessed_qa_pairs[i]['context']}")
    print()

print("Preprocessing completed successfully.")



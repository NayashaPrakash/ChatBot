import random
import torch
from preprocessing import preprocessed_qa_pairs
from preprocessing import read_squad_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertForQuestionAnswering, BertTokenizer
import tkinter as tk
from tkinter import ttk
 
def answer_question(user_question, model, tokenizer, preprocessed_qa_pairs):
    vectorizer = TfidfVectorizer()
    qa_corpus = [qa_pair["question"] for qa_pair in preprocessed_qa_pairs]
    vectors = vectorizer.fit_transform(qa_corpus)
    user_question_vector = vectorizer.transform([user_question])
    similarity_scores = cosine_similarity(user_question_vector, vectors)

    most_similar_index = similarity_scores.argmax()
    most_similar_qa_pair = preprocessed_qa_pairs[most_similar_index]

    threshold = 0.2
    if similarity_scores[0, most_similar_index] < threshold:
        input_ids = tokenizer.encode(user_question, most_similar_qa_pair["context"], add_special_tokens=True)
        inputs = torch.tensor(input_ids).unsqueeze(0)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            start_logits, end_logits = outputs.start_logits, outputs.end_logits

        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)

        answer = tokenizer.decode(input_ids[start_index:end_index+1], skip_special_tokens=True)
        
        if not answer.strip():
            return "Sorry, I couldn't find an answer to your question."

        return answer

    return most_similar_qa_pair["answer"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tuned_model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)
fine_tuned_model.load_state_dict(torch.load("fine_tuned_bert_model.pth"))

class Interface:
    def __init__(self, test, preprocessed_qa_pairs):
        self.window = None
        self.test = test
        self.preprocessed_qa_pairs = preprocessed_qa_pairs

    def start(self):
        self.gui_window()
        self.gui_text_field()
        self.gui_input_text_field()
        self.window.mainloop()

    def gui_window(self):
        self.window = tk.Tk()
        self.window.geometry("1000x500")
        self.window.config(bg=self.rgb_hack((36, 41, 46)))
        self.window.title("Quest Bot")

    def gui_text_field(self):
        self.text_field = tk.Label(self.window,
                                   height=20,
                                   width=100,
                                   bg=self.rgb_hack((46, 52, 59)),
                                   fg="#98EECC",
                                   wraplength=800,
                                   justify=tk.LEFT,
                                   anchor=tk.NW,
                                   font=("Times New Roman", 12))
        self.text_field.pack(padx=10, pady=10)

    def gui_input_text_field(self):
        self.input = tk.Entry(self.window,
                              bg=self.rgb_hack((60, 70, 80)),
                              fg="white",
                              font=("Times New Roman", 12))
        self.input.pack(padx=10, pady=10, fill=tk.X)

        button = tk.Button(self.window,
                           height=2,
                           width=20,
                           text="Send",
                           bg=self.rgb_hack((50, 38, 83)),
                           fg="white",
                           activebackground=self.rgb_hack((50, 110, 156)),
                           activeforeground="white",
                           font=("Times New Roman", 12),
                           command=lambda: self.take_input())
        button.pack(padx=10, pady=10)



    def take_input(self):
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        input_text = self.input.get()
        if input_text:
            self.text_field.config(text=self.text_field.cget("text") + "Question:  " + input_text + "\n")

            answer = answer_question(input_text, self.test, tokenizer, self.preprocessed_qa_pairs)

            self.text_field.config(text=self.text_field.cget("text") + "Answer:     " + answer + "\n\n")

            self.input.delete(0, tk.END)


    def rgb_hack(self, rgb):
        return "#%02x%02x%02x" % rgb

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tuned_model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)
    fine_tuned_model.load_state_dict(torch.load("fine_tuned_bert_model.pth"))

    chatbot_interface = Interface(fine_tuned_model, preprocessed_qa_pairs)
    chatbot_interface.start()

if __name__ == "__main__":
    main()

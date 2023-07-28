import random
from preprocessing import preprocessed_qa_pairs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def answer_question(user_question, preprocessed_qa_pairs):
    # Implement information retrieval technique using TF-IDF and Cosine Similarity
    vectorizer = TfidfVectorizer()
    qa_corpus = [qa_pair["question"] for qa_pair in preprocessed_qa_pairs]
    vectors = vectorizer.fit_transform(qa_corpus)
    user_question_vector = vectorizer.transform([user_question])
    similarity_scores = cosine_similarity(user_question_vector, vectors)

    # Find the most similar question in the knowledge base
    most_similar_index = similarity_scores.argmax()
    most_similar_qa_pair = preprocessed_qa_pairs[most_similar_index]

    # If similarity is below a threshold, consider it as "not found"
    threshold = 0.2
    if similarity_scores[0, most_similar_index] < threshold:
        return "Sorry, I couldn't find an answer to your question."

    return most_similar_qa_pair["answer"]

def main():
    print("Welcome to the mini chatbot!")
    print("Type 'exit' to end the conversation.")
    while True:
        user_question = input("Ask a question: ")
        if user_question.lower() == "exit":
            break

        answer = answer_question(user_question, preprocessed_qa_pairs)
        print("Answer:", answer)

if __name__ == "__main__":
    main()

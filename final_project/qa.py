import spacy
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model and Word2Vec model for question answering
nlp = spacy.load("en_core_web_sm")
word2vec = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300-SLIM.bin", binary=True)

def preprocess_text(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
    return sentences

def get_sentence_embedding(sentence):
    doc = nlp(sentence)
    words = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
    word_vectors = [word2vec[word] for word in words if word in word2vec]
    
    if len(word_vectors) == 0:
        return np.zeros(word2vec.vector_size)
    
    embedding = np.mean(word_vectors, axis=0)
    return embedding

def calculate_similarity(question, sentences):
    question_embedding = get_sentence_embedding(question)
    sentence_embeddings = [get_sentence_embedding(sentence) for sentence in sentences]
    
    if len(sentence_embeddings) == 0:
        return "No relevant sentences found."
    
    cosine_similarities = cosine_similarity([question_embedding], sentence_embeddings).flatten()
    best_index = np.argmax(cosine_similarities)
    best_sentence = sentences[best_index]
    return best_sentence

def generate_title(context):
    doc = nlp(context)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    if not noun_phrases:
        return "No title could be generated."
    title = ' '.join(noun_phrases[:2])
    return title.capitalize()

def answer_question(context, question):
    sentences = preprocess_text(context)
    if not sentences:
        return "No context available."
    answer = calculate_similarity(question, sentences)
    return answer

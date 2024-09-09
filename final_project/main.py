import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from collections import Counter
from gensim.models import KeyedVectors

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load pre-trained Word2Vec model (ensure the file is available)
word2vec = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300-SLIM.bin", binary=True)

# Stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Tokenizes the input text into sentences.
    """
    sentences = sent_tokenize(text)
    return sentences

def get_sentence_embedding(sentence):
    """
    Converts a sentence into a vector embedding using Word2Vec.
    """
    words = word_tokenize(sentence.lower())
    words = [word for word in words if word not in stop_words]
    word_vectors = [word2vec[word] for word in words if word in word2vec]

    if len(word_vectors) == 0:
        return np.zeros(word2vec.vector_size)

    embedding = np.mean(word_vectors, axis=0)
    return embedding

def summarize_text(context, summary_ratio=0.3):
    """
    Summarizes the input text based on cosine similarity between sentence embeddings.
    """
    sentences = preprocess_text(context)
    sentence_embeddings = [get_sentence_embedding(sentence) for sentence in sentences]
    
    # Calculate the similarity matrix between sentences
    similarity_matrix = cosine_similarity(sentence_embeddings)
    sentence_scores = np.sum(similarity_matrix, axis=1)
    sentence_scores /= np.linalg.norm(sentence_scores)

    # Select top sentences
    select_len = max(1, round(len(sentences) * summary_ratio))
    selected_sentences = []
    
    while len(selected_sentences) < select_len:
        for i in np.argsort(-sentence_scores):
            if i not in selected_sentences:
                is_diverse = all(cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[j]])[0][0] < 0.75 
                                 for j in selected_sentences)
                if is_diverse:
                    selected_sentences.append(i)
                    break

    summary = " ".join([sentences[i] for i in sorted(selected_sentences)])

    # Return required four values
    original_txt = context
    len_orig_text = len(context.split())
    len_summary = len(summary.split())

    return summary, original_txt, len_orig_text, len_summary

def extract_key_phrases_tfidf(text, n=3):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=1000)
    X = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
    
    top_n = feature_array[tfidf_sorting][:n]
    return top_n

def extract_key_phrases_textrank(text, n=3):
    r = Rake()
    r.extract_keywords_from_text(text)
    ranked_phrases = r.get_ranked_phrases()[:n]
    return ranked_phrases

def extract_named_entities(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    named_entities = ne_chunk(pos_tags, binary=False)
    entities = []
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            entities.append(' '.join(c[0] for c in chunk))
    return entities

def combine_scoring_methods(context):
    tfidf_phrases = extract_key_phrases_tfidf(context, n=5)
    textrank_phrases = extract_key_phrases_textrank(context, n=5)
    named_entities = extract_named_entities(context)

    all_phrases = list(tfidf_phrases) + list(textrank_phrases) + list(named_entities)
    phrase_counts = Counter(all_phrases)
    
    most_common_phrases = phrase_counts.most_common(3)

    if most_common_phrases:
        return most_common_phrases[0][0]
    else:
        return "Untitled"

def generate_title_tfidf(context):
    title = combine_scoring_methods(context)
    return title

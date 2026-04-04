# tfidf_embeddings.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFEmbeddings:
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(corpus)
        self.words = self.vectorizer.get_feature_names_out()
    
    def get_embedding(self, word):
        if word not in self.vectorizer.vocabulary_:
            return None
        idx = self.vectorizer.vocabulary_[word]
        return self.matrix[:, idx].toarray().flatten()
    
    def get_neighbors(self, word, top_n=5):
        vector = self.get_embedding(word)
        if vector is None:
            return []
        similarities = cosine_similarity(self.matrix.T, vector.reshape(1, -1)).flatten()
        neighbors_idx = similarities.argsort()[::-1][1:top_n+1]  # skip the word itself
        neighbors = [(self.words[i], float(similarities[i])) for i in neighbors_idx]
        return neighbors
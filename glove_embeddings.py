# glove_embeddings.py
import gensim.downloader as api

class GloVeEmbeddings:
    def __init__(self):
        print("Loading GloVe model... this may take a minute first time")
        self.model = api.load("glove-wiki-gigaword-100")
    
    def get_embedding(self, word):
        try:
            return self.model[word]
        except KeyError:
            return None
    
    def get_neighbors(self, word, top_n=5):
        try:
            neighbors = self.model.most_similar(word, topn=top_n)
            return neighbors
        except KeyError:
            return []
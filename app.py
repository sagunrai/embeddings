# app.py
from flask import Flask, render_template, request, jsonify
from corpus import corpus
from tfidf_embeddings import TFIDFEmbeddings
from glove_embeddings import GloVeEmbeddings

app = Flask(__name__)

# Initialize models
tfidf_model = TFIDFEmbeddings(corpus)
glove_model = GloVeEmbeddings()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/embed", methods=["POST"])
def embed():
    data = request.json
    word = data.get("word", "").strip()
    method = data.get("method", "tfidf")
    top_n = int(data.get("top_n", 5))
    
    if word == "":
        return jsonify({"error": "Please enter a word!"})
    
    if method == "tfidf":
        embedding = tfidf_model.get_embedding(word)
        neighbors = tfidf_model.get_neighbors(word, top_n)
    else:
        embedding = glove_model.get_embedding(word)
        neighbors = glove_model.get_neighbors(word, top_n)
    
    if embedding is None:
        return jsonify({"error": f"Word '{word}' not found in {method.upper()} embeddings!"})
    
    return jsonify({
        "word": word,
        "embedding": embedding.tolist(),
        "neighbors": [{"word": w, "similarity": float(sim)} for w, sim in neighbors]
    })

if __name__ == "__main__":
    app.run(debug=True)
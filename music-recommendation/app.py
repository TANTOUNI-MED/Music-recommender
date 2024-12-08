from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model

app = Flask(__name__)

# Load saved data and model
similarity_matrix = pickle.load(open("similarity_matrix.pkl", "rb"))
filename_to_index = pickle.load(open("filename_to_index.pkl", "rb"))
index_to_filename = pickle.load(open("index_to_filename.pkl", "rb"))
best_model = load_model("best_model.h5")

MUSIC_DIR = "categories"

@app.route('/')
def index():
    # Get the list of categories
    categories = [folder for folder in os.listdir(MUSIC_DIR) if os.path.isdir(os.path.join(MUSIC_DIR, folder))]
    return render_template('home.html', categories=categories)

@app.route('/get_music/<category>')
def get_music(category):
    # Get the list of music files in the selected category
    category_path = os.path.join(MUSIC_DIR, category)
    if not os.path.exists(category_path):
        return jsonify({'error': 'Category not found'}), 404
    music_files = [file for file in os.listdir(category_path) if file.endswith('.wav')]
    return jsonify(music_files)

@app.route('/player')
def player():
    category = request.args.get('category')
    song = request.args.get('song')
    return render_template('player.html', category=category, song=song)

@app.route('/categories/<category>/<filename>')
def serve_music(category, filename):
    return send_from_directory(os.path.join(MUSIC_DIR, category), filename)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    input_filename = data.get("filename")
    num_recommendations = int(data.get("num_recommendations", 5))
    
    # Validate input
    if input_filename not in filename_to_index:
        return jsonify({"error": "Filename not found in the dataset."}), 400
    
    # Get recommendations
    input_index = filename_to_index[input_filename]
    similarity_scores = list(enumerate(similarity_matrix[input_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:num_recommendations + 1]
    recommended_indices = [i[0] for i in similarity_scores]
    recommended_filenames = [
        index_to_filename.get(idx, "Unknown") for idx in recommended_indices
    ]
    print("recommendations : ",recommended_filenames)
    return jsonify({
        "input_filename": input_filename,
        "recommendations": recommended_filenames
    })

if __name__ == '__main__':
    app.run(debug=True)
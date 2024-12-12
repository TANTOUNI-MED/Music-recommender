import os
import pickle
import numpy as np
import pandas as pd
import hashlib
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session, flash
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  

# Load saved data and model
similarity_matrix = pickle.load(open("similarity_matrix.pkl", "rb"))
filename_to_index = pickle.load(open("filename_to_index.pkl", "rb"))
index_to_filename = pickle.load(open("index_to_filename.pkl", "rb"))
best_model = load_model("best_model.h5")

# Load users dataset and interactions
users_df = pd.read_csv("users.csv")
interactions_df = pd.read_csv("user_song_table.csv")

# @app.route('/')
# def home():
#     return render_template('home.html')

MUSIC_DIR = "categories"

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        print('username :',username)
        print('password :',password)
        
        # Check credentials
        user = users_df[(users_df['name'] == username) & (users_df['password'] == password)]
        print('user :',user)

        
        if not user.empty:
            # Store user ID in session
            session['user_id'] = user.iloc[0]['user_id']
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/')
def index():

    # Get the list of categories
    categories = [folder for folder in os.listdir(MUSIC_DIR) if os.path.isdir(os.path.join(MUSIC_DIR, folder))]
    
    # Get collaborative filtering recommendations for the logged-in user
    try:
        user_recommendations = get_user_recommendations(interactions_df, session['user_id'], top_k=10)
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        user_recommendations = []
    
    return render_template('home.html', 
                           categories=categories, 
                           username=session.get('username'),
                           recommendations=user_recommendations)

# Rest of the routes remain the same as in the previous implementation
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
import os
import pickle
import numpy as np
import pandas as pd
import hashlib
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session, flash
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
graph = tf.compat.v1.get_default_graph()
print(tf.__version__)

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2



app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  

# Music Directory
MUSIC_DIR = "categories"

def load_playlist_data():
    try:
        matrix_df = pd.read_csv("playlist_user_matrix.csv", index_col=0)
        playlists_df = pd.read_csv("playlists.csv")
        return matrix_df, playlists_df
    except Exception as e:
        print(f"Error loading playlist data: {str(e)}")
        return None, None

def get_playlist_recommendations(user_id, matrix_df, playlists_df, top_n=5):
    try:
        if user_id not in matrix_df.columns:
            print(f"Error: User '{user_id}' not found in the database")
            return None
        
        user_ratings = matrix_df[user_id]
        top_playlists = user_ratings.sort_values(ascending=False)[:top_n]
        
        recommendations = []
        for playlist_name, score in top_playlists.items():
            if score > 0:
                playlist_tracks = playlists_df[playlist_name].dropna().tolist()
                
                recommendation = {
                    'playlist_name': playlist_name,
                    'score': float(score),
                    'tracks': playlist_tracks
                }
                recommendations.append(recommendation)
        
        return recommendations
    except Exception as e:
        print(f"Error generating playlist recommendations: {str(e)}")
        return None

# Recommendation System Classes and Functions
class SingleUserRecommender:
    def __init__(self, num_items, encoding_dim=50):
        self.num_items = num_items
        self.encoding_dim = encoding_dim
        self.encoder = self._build_encoder()
    
    def _build_encoder(self):
        input_layer = Input(shape=(self.num_items,))
        
        # Encoder layers
        encoded = Dense(self.encoding_dim * 2, 
                       activation='relu', 
                       kernel_regularizer=l2(0.001))(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.5)(encoded)
        
        encoded = Dense(self.encoding_dim, 
                       activation='relu', 
                       kernel_regularizer=l2(0.001))(encoded)
        encoded = BatchNormalization()(encoded)
        
        # Create encoder model
        encoder = Model(input_layer, encoded)
        
        return encoder
    
    def get_recommendations(self, user_interactions, item_names, top_k=10):
        """
        Get recommendations for a single user.
        """
        # Ensure interactions are numpy array and binary
        user_interactions = np.array(user_interactions).astype(int)
        
        # Scale interactions
        scaler = StandardScaler()
        scaled_interactions = scaler.fit_transform(user_interactions.reshape(1, -1))
        
        # Get user embedding
        user_embedding = self.encoder.predict(scaled_interactions)[0]
        
        # Get item embeddings
        item_embeddings = self.encoder.predict(np.eye(self.num_items))
        
        # Compute cosine similarity
        scores = cosine_similarity(user_embedding.reshape(1, -1), item_embeddings)[0]
        
        # Remove already interacted items
        scores[user_interactions == 1] = -np.inf
        
        # Get top K recommendations (item indices)
        top_item_indices = scores.argsort()[-top_k:][::-1]
        
        # Convert to recommended item names
        recommended_items = [item_names[idx] for idx in top_item_indices]
        
        return recommended_items

def train_recommender(interactions, item_names):
    """
    Train a recommender model.
    """
    # Convert interactions to binary matrix
    interaction_matrix = (interactions.values > 0).astype(int)
    
    # Create recommender
    recommender = SingleUserRecommender(
        num_items=interaction_matrix.shape[1],
        encoding_dim=min(50, interaction_matrix.shape[1] // 2)
    )
    
    return recommender

def get_user_recommendations(df, user_id, top_k=10):
    """
    Get recommendations for a specific user.
    """
    # Validate user_id
    if 'user_id' not in df.columns:
        raise ValueError("DataFrame must contain a 'user_id' column")

    # Prepare data
    item_columns = [col for col in df.columns if col != 'user_id']
    item_names = item_columns

    # Check if user exists
    user_row = df[df['user_id'] == user_id]
    if user_row.empty:
        print(f"No interaction data found for user {user_id}")
        return []

    # Get user interactions
    user_interactions = user_row[item_columns].values[0]

    # Check if user has any interactions
    if user_interactions.sum() == 0:
        print(f"User {user_id} has no interaction history")
        return []

    # Train recommender
    recommender = train_recommender(df[item_columns], item_names)

    # Get recommendations
    try:
        recommendations = recommender.get_recommendations(
            user_interactions, 
            item_names, 
            top_k=top_k
        )
        print('recommendations : ',recommendations)
        return recommendations
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return []

# Load saved data and model
similarity_matrix = pickle.load(open("similarity_matrix.pkl", "rb"))
filename_to_index = pickle.load(open("filename_to_index.pkl", "rb"))
index_to_filename = pickle.load(open("index_to_filename.pkl", "rb"))
best_model = load_model("best_model.h5")

# Load users dataset and interactions
users_df = pd.read_csv("users.csv")
interactions_df = pd.read_csv("user_song_table.csv")

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check credentials
        user = users_df[(users_df['name'] == username) & (users_df['password'] == password)]
        
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

# Main index route
@app.route('/')
def index():
    categories = [
        folder for folder in os.listdir(MUSIC_DIR)
        if os.path.isdir(os.path.join(MUSIC_DIR, folder))
    ]

    user_recommendations = []
    playlist_recommendations = []

    if 'user_id' in session:
        # Get existing song recommendations
        try:
            user_recommendations = get_user_recommendations(
                interactions_df, session['user_id'], top_k=10
            )
        except Exception as e:
            print(f"Error getting song recommendations: {e}")

        # Get playlist recommendations
        try:
            matrix_df, playlists_df = load_playlist_data()
            if matrix_df is not None and playlists_df is not None:
                user_id = session['user_id']
                playlist_recommendations = get_playlist_recommendations(
                    user_id, matrix_df, playlists_df
                )
        except Exception as e:
            print(f"Error getting playlist recommendations: {e}")

    return render_template(
        'home.html',
        categories=categories,
        username=session.get('username'),
        recommendations=user_recommendations,
        playlist_recommendations=playlist_recommendations
    )

@app.route('/play_playlist/<playlist_name>')
def play_playlist(playlist_name):
    try:
        _, playlists_df = load_playlist_data()
        if playlists_df is not None:
            tracks = playlists_df[playlist_name].dropna().tolist()
            return jsonify({'success': True, 'tracks': tracks})
    except Exception as e:
        print(f"Error loading playlist: {e}")
    return jsonify({'success': False, 'error': 'Failed to load playlist'})

# Get music for a specific category
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

# Recommendation endpoint
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


@app.route('/rate_song', methods=['POST'])
def rate_song():
    # Ensure user is logged in
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    try:
        # Get JSON data
        data = request.get_json()
        song = data.get('song')
        rating = data.get('rating')
        
        # Print for debugging
        print(f"Received song: {song}")
        
        # Load current interactions
        interactions_df = pd.read_csv("user_song_table.csv")
        
        # Find matching column with flexible matching
        matching_columns = [col for col in interactions_df.columns if song.split('.')[0] in col and song.split('.')[-1] == col.split('.')[-1]]

        if not matching_columns:
            print(f"No matching column found for song {song}")
            return jsonify({'success': False, 'error': f'Song {song} not found in columns'}), 404

        # If multiple matches, take the first one
        matching_column = matching_columns[0]
        
        # Standardize user_id handling
        user_id = session['user_id']
        user_id_str = f'user_{user_id}' if not str(user_id).startswith('user_') else str(user_id)

        # Find user row with flexible matching
        user_row = interactions_df[
            (interactions_df['user_id'] == user_id_str) | 
            (interactions_df['user_id'] == str(user_id)) |
            (interactions_df['user_id'] == user_id)
        ]
        
        if user_row.empty:
            return jsonify({'success': False, 'error': f'User {user_id_str} not found'}), 404
        
        user_index = user_row.index[0]
        
        # Update the rating for the specific song column
        interactions_df.at[user_index, matching_column] = float(rating)
        
        # Save updated interactions
        interactions_df.to_csv("user_song_table.csv", index=False)
        
        return jsonify({'success': True})
    
    except Exception as e:
        print(f"Error in rate_song: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
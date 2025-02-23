# Collaborative Filtering Movie Recommendation System

## File Structure
- **Collaborative filtering-movies.mat**: Dataset containing ratings and movie information.
- **collab_movie_ids.txt**: A list of movie IDs for reference.
- **Collaborative filtering.py**: The main Python script implementing the recommendation system.

## How It Works
1. **Data Input**: The dataset `collaborative fitering-movies.mat` is loaded. It contains movie ratings and user ratings. User ratings are added to the dataset, and the matrix is normalized.
2. **Matrix Factorization**: The system uses a collaborative filtering approach where the rating matrix is factorized into two matrices, `X` and `theta`, using gradient descent. The cost function is optimized to minimize the error in predicted ratings.
3. **Model Training**: Gradient descent is used for optimization with a regularization parameter `Lambda`. The model is trained over a specified number of iterations.
4. **Prediction & Recommendations**: The system predicts ratings for movies that the user hasn’t rated yet. Top movie recommendations are provided based on predicted ratings.

## Usage
1. Make sure the dataset (`collaborative fitering-movies.mat` and `collab_movie_ids.txt`) is in the same directory as the script.
2. Run the Python script:
python Collaborative filtering.py

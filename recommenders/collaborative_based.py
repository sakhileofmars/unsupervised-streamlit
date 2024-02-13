# Script dependencies
import pandas as pd
import numpy as np
import pickle
from surprise import Reader, Dataset
from sklearn.metrics.pairwise import cosine_similarity

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv', sep=',')
ratings_df = pd.read_csv('resources/data/ratings.csv')

# We make use of an SVD model trained with the full dataset.
model = pickle.load(open('resources/models/SVD.pkl', 'rb'))
# model = pickle.load(open('resources/models/lgbm_model.pkl', 'rb'))

def prediction_item(item_id, df):
    """Map a given favourite movie to users within the
    MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.
    df : pd.DataFrame
        The DataFrame containing ratings.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Filter ratings for the given movie
    movie_ratings = df[df['movieId'] == item_id]

    # Group by user and calculate average rating
    user_ratings = movie_ratings.groupby('userId')['rating'].mean()

    # Sort users by rating in descending order
    sorted_users = user_ratings.sort_values(ascending=False)

    return sorted_users.index.tolist()

def pred_movies(movie_list, df):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.
    df : pd.DataFrame
        The DataFrame containing ratings.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store = []
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for movie in movie_list:
        movie_id = movies_df[movies_df['title'] == movie]['movieId'].iloc[0]
        user_ids = prediction_item(movie_id, df)
        id_store.extend(user_ids[:10])  # Take the top 10 user ids
    return id_store

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def collab_model(df, movie_list, top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
    by the app user.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be used for collaborative filtering.
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    indices = pd.Series(df['title'])
    movie_ids = pred_movies(movie_list, ratings_df)

    # Filter ratings for selected users
    selected_ratings = ratings_df[ratings_df['userId'].isin(movie_ids)]

    # Pivot table to user-item matrix
    user_item_matrix = selected_ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

    # Calculating the cosine similarity matrix
    cosine_sim = cosine_similarity(user_item_matrix, user_item_matrix)

    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]

    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]

    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending=False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending=False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending=False)

    # Appending the names of movies
    listings = score_series_1.append(score_series_2).append(score_series_3).sort_values(ascending=False)
    recommended_movies = []

    # Choose top 50
    top_50_indexes = list(listings.iloc[1:50].index)

    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes, [idx_1, idx_2, idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(df['title'])[i])

    return recommended_movies
# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep=',')
ratings = pd.read_csv('resources/data/ratings.csv')
ratings.drop('timestamp', axis=1, inplace=True)  # We don't need timestamp for this task

def data_preprocessing(df):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be preprocessed.

    Returns
    -------
    Pandas DataFrame
        Preprocessed DataFrame.

    """
    # Merging movies and ratings
    merged_df = pd.merge(df, ratings, on='movieId')

    # Grouping movies and aggregating genres
    movie_groups = merged_df.groupby('title')['genres'].agg(lambda x: ' '.join(x))

    return movie_groups

def content_model(df, movie_list, top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be used for content-based filtering.
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : int
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(df)

    # Instantiating and generating the count matrix
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(data)

    # Calculating cosine similarity
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Getting the indices of the movies that match the titles in movie_list
    movie_indices = [df.index[df.index == title].tolist()[0] for title in movie_list]

    # Creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[movie_indices[0]]).sort_values(ascending=False)

    # Getting the indexes of the top_n most similar movies
    top_indexes = list(score_series.iloc[1:top_n + 1].index)

    # Appending the names of movies
    recommended_movies = [list(df.index)[i] for i in top_indexes]

    return recommended_movies

# Example usage:
# movie_list = ['Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)']
# recommended_movies = content_model(movies, movie_list)
# print(recommended_movies)
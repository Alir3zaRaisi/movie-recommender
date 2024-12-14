import pandas as pd
import networkx as nx
import numpy as np

def load_movielens_data(data_path, ratings_file="ratings.dat", movies_file="movies.dat"):
    """
    Load MovieLens data files with '::' as the separator.

    Parameters:
    - data_path: str, path to the folder containing MovieLens files.
    - ratings_file: str, name of the ratings file.
    - movies_file: str, name of the movies file.

    Returns:
    - ratings_df: DataFrame containing ratings data.
    - movies_df: DataFrame containing movie metadata.
    """
    ratings_path = f"{data_path}/{ratings_file}"
    movies_path = f"{data_path}/{movies_file}"
    
    # Load ratings data
    ratings_df = pd.read_csv(ratings_path, sep="::", header=None, engine='python',
                             names=["userId", "movieId", "rating", "timestamp"])
    
    # Load movies data (with explicit encoding)
    movies_df = pd.read_csv(movies_path, sep="::", header=None, engine='python', encoding="ISO-8859-1",
                            names=["movieId", "title", "genres"])
    
    return ratings_df, movies_df

def preprocess_data(ratings_df, normalize_ratings=True):
    """
    Preprocess MovieLens data and create a user-movie bipartite graph.

    Parameters:
    - ratings_df: DataFrame, contains user-item ratings.
    - normalize_ratings: bool, whether to normalize ratings between 0 and 1.
    - apply_time_decay: bool, whether to apply time decay to weights.
    - decay_factor: float, controls the time decay effect (higher means faster decay).

    Returns:
    - G: NetworkX Graph, bipartite graph of users and movies.
    """

    # Normalize ratings (optional)
    if normalize_ratings:
        min_rating = ratings_df['rating'].min()
        max_rating = ratings_df['rating'].max()
        ratings_df['normalized_rating'] = (ratings_df['rating'] - min_rating) / (max_rating - min_rating)
    else:
        ratings_df['normalized_rating'] = ratings_df['rating']

    ratings_df['weight'] = ratings_df['normalized_rating']
    # Create a bipartite graph
    G = nx.Graph()
    
    # Add nodes and edges
    for _, row in ratings_df.iterrows():

        # During graph creation, cast user and movie IDs to strings without decimals
        user_node = f"user_{int(row['userId'])}"
        movie_node = f"movie_{int(row['movieId'])}"
        G.add_node(user_node, bipartite=0)  # User nodes
        G.add_node(movie_node, bipartite=1)  # Movie nodes
        G.add_edge(user_node, movie_node, weight=row['weight'])
    
    print(f"Graph created with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G


if __name__ == "__main__":
    # Path to data folder
    data_path = "../data/ml-1m"

    # Load data
    ratings, movies = load_movielens_data(data_path)

    # Preprocess and create graph
    user_movie_graph = preprocess_data(ratings)

    # Save the graph structure for future use (optional)
    nx.write_graphml(user_movie_graph, f"{data_path}/user_movie_graph.graphml")
    print(f"Graph saved to {data_path}/user_movie_graph.graphml.")


import pandas as pd
import networkx as nx

def load_movielens_data(data_path, ratings_file="ratings.dat", movies_file="movies.dat", users_file="users.dat"):
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
    users_path = f"{data_path}/{users_file}"

    # Load ratings data
    ratings_df = pd.read_csv(ratings_path, sep="::", header=None, engine='python',
                             names=["userId", "movieId", "rating", "timestamp"])
    
    # Load movies data (with explicit encoding)
    movies_df = pd.read_csv(movies_path, sep="::", header=None, engine='python', encoding="ISO-8859-1",
                            names=["movieId", "title", "genres"])
    
    users_df = pd.read_csv(users_path, sep="::", header=None, engine='python', encoding="ISO-8859-1",
                            names=["userId", "Gender", "Age", "Occupation", "Zip"])

    return ratings_df, movies_df, users_df


def create_bipartite_graph(ratings_df):
    """
    Create a bipartite graph from a ratings DataFrame.

    Parameters:
    - ratings_df: DataFrame, contains columns ['userId', 'movieId', 'rating', 'timestamp'].

    Returns:
    - G: NetworkX bipartite graph with user and movie nodes.
    """
    # Create an empty bipartite graph
    G = nx.Graph()

    # Add user nodes
    user_nodes = [f"user_{int(user)}" for user in ratings_df['userId'].unique()]
    G.add_nodes_from(user_nodes, bipartite=0)  # bipartite=0 indicates user nodes

    # Add movie nodes
    movie_nodes = [f"movie_{int(movie)}" for movie in ratings_df['movieId'].unique()]
    G.add_nodes_from(movie_nodes, bipartite=1)  # bipartite=1 indicates movie nodes

    # Add edges with weights (normalized rating or raw rating)
    for _, row in ratings_df.iterrows():
        user_node = f"user_{int(row['userId'])}"
        movie_node = f"movie_{int(row['movieId'])}"
        G.add_edge(user_node, movie_node, weight=row['rating'])

    print(f"Graph created with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G


if __name__ == "__main__":
    # Path to data folder
    data_path = "../data/ml-1m"

    # Load data
    ratings, movies, users = load_movielens_data(data_path)

    # Preprocess and create graph
    user_movie_graph = create_bipartite_graph(ratings)

    # Save the graph structure for future use (optional)
    nx.write_graphml(user_movie_graph, f"{data_path}/user_movie_graph.graphml")
    print(f"Graph saved to {data_path}/user_movie_graph.graphml.")


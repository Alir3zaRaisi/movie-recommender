import pandas as pd
import networkx as nx

def load_movielens_data(data_path, dataset_type="1M", ratings_file=None, movies_file=None, users_file=None):
    """
    Load MovieLens data files for 1M or 100K datasets.

    Parameters:
    - data_path: str, path to the folder containing MovieLens files.
    - dataset_type: str, "1M" or "100K", indicating the dataset version.
    - ratings_file: str, name of the ratings file (optional for custom filenames).
    - movies_file: str, name of the movies file (optional for custom filenames).
    - users_file: str, name of the users file (optional for custom filenames).

    Returns:
    - ratings_df: DataFrame containing ratings data.
    - movies_df: DataFrame containing movie metadata.
    - users_df: DataFrame containing user metadata (if available).
    """
    if dataset_type == "1M":
        # Default filenames for 1M dataset
        ratings_file = ratings_file or "ratings.dat"
        movies_file = movies_file or "movies.dat"
        users_file = users_file or "users.dat"
        
        # File paths
        ratings_path = f"{data_path}/{ratings_file}"
        movies_path = f"{data_path}/{movies_file}"
        users_path = f"{data_path}/{users_file}"

        # Load ratings data
        ratings_df = pd.read_csv(ratings_path, sep="::", header=None, engine="python",
                                 names=["userId", "movieId", "rating", "timestamp"])
        
        # Load movies data
        movies_df = pd.read_csv(movies_path, sep="::", header=None, engine="python", encoding="ISO-8859-1",
                                names=["movieId", "title", "genres"])
        
        # Load users data
        users_df = pd.read_csv(users_path, sep="::", header=None, engine="python", encoding="ISO-8859-1",
                               names=["userId", "Gender", "Age", "Occupation", "Zip"])
    
    elif dataset_type == "100K":
        # Default filenames for 100K dataset
        ratings_file = ratings_file or "u.data"
        movies_file = movies_file or "u.item"
        users_file = users_file or "u.user"
        
        # File paths
        ratings_path = f"{data_path}/{ratings_file}"
        movies_path = f"{data_path}/{movies_file}"
        users_path = f"{data_path}/{users_file}"

        # Load ratings data
        ratings_df = pd.read_csv(ratings_path, sep="\t", header=None, names=["userId", "movieId", "rating", "timestamp"])
        
        # Load movies data
        movies_df = pd.read_csv(movies_path, sep="|", header=None, encoding="ISO-8859-1",
                                names=["movieId", "title", "release_date", "video_release_date", "IMDb_URL"] + 
                                      [f"genre_{i}" for i in range(19)])
        
        # Load users data
        users_df = pd.read_csv(users_path, sep="|", header=None,
                               names=["userId", "Age", "Gender", "Occupation", "Zip"])
    
    else:
        raise ValueError("Unsupported dataset type. Please specify '1M' or '100K'.")

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

    # Add user and movie nodes dynamically
    for _, row in ratings_df.iterrows():
        user_node = f"user_{row['userId']}"
        movie_node = f"movie_{row['movieId']}"
        G.add_node(user_node, bipartite=0)  # bipartite=0 indicates user nodes
        G.add_node(movie_node, bipartite=1)  # bipartite=1 indicates movie nodes
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


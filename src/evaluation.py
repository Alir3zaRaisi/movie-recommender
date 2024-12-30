def calculate_correctly_placed_pairs(recommendations, hidden_movies):
    """
    Calculate accuracy based on correctly placed pairs.

    Parameters:
    - recommendations: List of tuples (movie, score) sorted by scores (output of rank_movies_with_penalty).
    - hidden_movies: Set of ground truth hidden movies.

    Returns:
    - accuracy: Fraction of correctly placed pairs.
    """
    # Extract movie rankings from recommendations
    recommended_ranks = {movie: rank for rank, (movie, _) in enumerate(recommendations)}

    # Separate hidden and non-hidden movies
    hidden_movie_ranks = {movie: recommended_ranks[movie] for movie in hidden_movies if movie in recommended_ranks}
    non_hidden_movie_ranks = {movie: recommended_ranks[movie] for movie in recommended_ranks if movie not in hidden_movies}

    # Initialize pair counts
    correct_pairs = 0
    total_pairs = 0

    # Compare all pairs of (hidden, non-hidden)
    for _, hidden_rank in hidden_movie_ranks.items():
        for _, non_hidden_rank in non_hidden_movie_ranks.items():
            # A pair is correctly placed if the hidden movie is ranked higher (lower rank value)
            if hidden_rank < non_hidden_rank:
                correct_pairs += 1
            total_pairs += 1

    # Avoid division by zero
    accuracy = (correct_pairs / total_pairs) if total_pairs > 0 else 0
    return accuracy


def calculate_top_k_accuracy(recommendations, hidden_movies, total_movies, k=10):
    """
    Calculate Top-K accuracy for recommendations based on a percentage.

    Parameters:
    - recommendations: List of tuples (movie, score) sorted by scores (output of rank_movies_with_penalty).
    - hidden_movies: Set of ground truth movies.
    - total_movies: Total number of movies in the dataset.
    - k: Percentage of top recommendations to consider (e.g., k=10 for top 10%).

    Returns:
    - accuracy: Fraction of top-K recommendations present in hidden_movies.
    """
    # Calculate the number of movies to consider based on percentage
    top_k_count = k * len(total_movies) // 100  # Ensure at least 1 movie is considered

    # Get the top-K recommended movies
    top_k_recommendations = [movie for movie, _ in recommendations[:top_k_count]]

    # Calculate intersection with hidden movies
    correct_recommendations = set(top_k_recommendations) & hidden_movies

    # Compute accuracy
    accuracy = len(correct_recommendations) / top_k_count
    return accuracy


def calculate_recall_at_k(recommendations, hidden_movies, total_movies, k=10):
    """
    Calculate Recall@k for recommendations, adjusted by total relevant movies in the dataset.

    Parameters:
    - recommendations: List of tuples (movie, score) sorted by scores (output of rank_movies_with_penalty).
    - hidden_movies: Set of ground truth hidden movies for the user.
    - total_movies: Total number of movies in the dataset.
    - k: Percentage of top recommendations to consider (e.g., k=10 for top 10%).
    - total_relevant_movies: Total number of relevant movies in the dataset (or for the user).

    Returns:
    - recall: Fraction of relevant items retrieved in the top-k recommendations.
    """
    # Calculate the number of movies to consider based on percentage
    top_k_count = k * len(total_movies) // 100  # Ensure at least 1 movie is considered

    # Get the top-K recommended movies
    top_k_recommendations = [movie for movie, _ in recommendations[:top_k_count]]

    # Calculate the number of relevant movies in the top-K recommendations
    relevant_movies = set(top_k_recommendations) & hidden_movies

    # Calculate Recall
    recall = len(relevant_movies) / len(hidden_movies)
    return recall

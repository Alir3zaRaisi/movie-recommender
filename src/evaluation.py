def calculate_correctly_placed_pairs(recommendations, hidden_movies):
    """
    Calculate accuracy based on correctly placed pairs.

    Parameters:
    - recommendations: List of tuples (movie, score) sorted by scores (output of rank_movies_with_penalty).
    - hidden_movies: Dictionary with ground truth rankings of movies. {movie: rank}

    Returns:
    - accuracy: Fraction of correctly placed pairs.
    """
    # Filter recommendations to only include movies in the hidden_movies set
    filtered_recommendations = [movie for movie, _ in recommendations if movie in hidden_movies]

    # Count correctly placed pairs
    correct_pairs = 0
    total_pairs = 0

    for i in range(len(filtered_recommendations)):
        for j in range(i + 1, len(filtered_recommendations)):
            movie_i = filtered_recommendations[i]
            movie_j = filtered_recommendations[j]

            # Check pair order based on ground truth ranks
            if hidden_movies[movie_i] < hidden_movies[movie_j]:
                correct_pairs += 1
            total_pairs += 1

    # Avoid division by zero
    accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
    return accuracy


def calculate_top_k_accuracy(recommendations, hidden_movies, k=10):
    """
    Calculate Top-K accuracy for recommendations.

    Parameters:
    - recommendations: List of tuples (movie, score) sorted by scores (output of rank_movies_with_penalty).
    - hidden_movies: Set of ground truth movies.
    - k: Number of top recommendations to consider.

    Returns:
    - accuracy: Fraction of top-K recommendations present in hidden_movies.
    """
    # Get the top-K recommended movies
    top_k_recommendations = [movie for movie, _ in recommendations[:k]]

    # Calculate intersection with hidden movies
    correct_recommendations = set(top_k_recommendations) & hidden_movies

    # Compute accuracy
    accuracy = len(correct_recommendations) / k
    return accuracy

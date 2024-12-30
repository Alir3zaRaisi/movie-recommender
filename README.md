# MovieLens Recommender System

This repository implements a random-walk-based recommender system using the MovieLens dataset. The project is inspired by research papers that utilize random walk techniques for recommendation.

## Features
- Random Walk with Restart (RWR)
- ItemRank
- Evaluation using Top-K Accuracy, recall@k, and  Pairwise Accuracy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/movie-recommender.git
   cd movie-recommender
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3.Download the MovieLens dataset from [GroupLens](https://grouplens.org/datasets/movielens/) and place it in the `data/` folder.
## Usage
```bash
jupyter-notebook  src/main.ipynb
```
## License
Dataset License: The MovieLens dataset is governed by its own license. Please refer to [LICENSE_DATASET](LICENSE_DATASET) for more details.

Code License: The code in this repository is licensed under the MIT License. See [LICENSE_CODE](LICENSE) for details.

## Acknowledgment
This project is based on the MovieLens dataset created by GroupLens Research. We thank the authors for their contributions to recommender systems research.

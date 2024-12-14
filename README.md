# MovieLens Recommender System

This repository implements a random-walk-based recommender system using the MovieLens dataset. The project is inspired by research papers that utilize random walk techniques for recommendation.

## Features
- Random Walk with Restart (RWR)
- ItemRank
- Hybrid Recommender combining content-based, collaborative, and graph-based approaches
- Evaluation using precision@k, recall@k, and F1-score

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/movie-recommender.git
   cd movie-recommender
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
## Usage
```bash
python src/preprocess_data.py
```
## License
Dataset License: The MovieLens dataset is governed by its own license. Please refer to [LICENSE_DATASET](LICENSE_DATASET) for more details.

Code License: The code in this repository is licensed under the MIT License. See [LICENSE_CODE](LICENSE) for details.

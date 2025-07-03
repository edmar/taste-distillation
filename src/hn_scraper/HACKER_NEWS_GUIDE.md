# Hacker News Interest Classification Dataset

This tool extracts your Hacker News voting data to create a machine learning dataset for classifying your interests.

## What This Tool Does

1. **Collects Positive Examples**: Posts you upvoted and favorited
2. **Collects Negative Examples**: Recent posts you didn't interact with
3. **Enriches Data**: Adds metadata like scores, domains, categories, timestamps
4. **Creates Balanced Dataset**: Exports to CSV for machine learning

## Quick Start

```bash
# Install dependencies
poetry install

# Run the extractor
python example_usage.py
```

Enter your Hacker News username when prompted. The tool will create `hn_dataset_[username].csv`.

## Dataset Features

Each row represents a Hacker News post with these columns:

- `post_id`: HN story ID
- `title`: Post title
- `url`: Link URL (or empty for Ask HN)
- `score`: HN points/upvotes
- `descendants`: Number of comments
- `time`: Unix timestamp
- `author`: Post author
- `date`: Date (YYYY-MM-DD)
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0=Monday)
- `domain`: Website domain (e.g., "github.com")
- `category`: Post type ("ask", "show", "job", "story")
- `your_vote`: **Target variable** (1=voted, 0=not voted)
- `source`: Data source ("upvoted", "favorites", "non_voted")

## Authentication for Upvoted Posts

The tool can extract your favorites without authentication, but upvoted posts require your HN session cookies.

### Getting Session Cookies

1. Log into Hacker News in your browser
2. Open Developer Tools (F12)
3. Go to Application/Storage → Cookies → news.ycombinator.com
4. Copy the cookie values

### Using Session Cookies

```python
from src.taste.hn_scraper import HackerNewsDataExtractor

extractor = HackerNewsDataExtractor("your_username")

# Add cookies for upvoted posts access
cookies = {
    'user': 'your_user_cookie_value',
    # Add other relevant cookies
}

upvoted = extractor.get_upvoted_posts(session_cookies=cookies)
```

## Machine Learning Usage

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('hn_dataset_username.csv')

# Feature engineering
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
title_features = tfidf.fit_transform(df['title'])

# Combine with other features
numeric_features = df[['score', 'descendants', 'hour', 'day_of_week']].fillna(0)
categorical_features = pd.get_dummies(df[['category', 'domain']])

# Train classifier
X = pd.concat([
    pd.DataFrame(title_features.toarray()),
    numeric_features.reset_index(drop=True),
    categorical_features.reset_index(drop=True)
], axis=1)
y = df['your_vote']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
```

## Analysis Tools

```bash
# Analyze existing dataset
python example_usage.py analyze hn_dataset_username.csv
```

This shows:
- Class distribution (voted vs non-voted)
- Top domains and categories you engage with
- Temporal voting patterns
- Score distributions

## Configuration Options

### Sampling Parameters

```python
extractor = HackerNewsDataExtractor(
    username="your_username",
    rate_limit=1.0  # Seconds between API calls
)

df = extractor.create_dataset(
    output_file='custom_dataset.csv',
    negative_samples=2000  # More negative examples
)
```

### Custom Negative Sampling

```python
# Sample from specific time periods
negative_posts = extractor.sample_non_voted_posts(
    num_samples=1000,
    days_back=60  # Go back 60 days
)
```

## Limitations & Notes

1. **Rate Limiting**: Respects HN's servers with 1-second delays
2. **Upvoted Posts**: Requires authentication cookies
3. **Data Freshness**: Negative samples from recent top/new stories
4. **Privacy**: Only your public voting data is collected
5. **Completeness**: May not capture all historical votes

## Troubleshooting

**"No favorites found"**: Username might be incorrect or no public favorites

**"Could not fetch upvoted posts"**: Need valid session cookies

**"Too few negative samples"**: Increase `days_back` parameter

**Rate limiting errors**: Increase `rate_limit` parameter

## Next Steps

1. **Feature Engineering**: Extract more text features, add domain categories
2. **Model Selection**: Try XGBoost, neural networks, or ensemble methods  
3. **Evaluation**: Use cross-validation, precision/recall metrics
4. **Deployment**: Create a web app to classify new HN posts
5. **Continuous Learning**: Update dataset periodically with new votes

## Example Output

```
📊 Dataset shape: (2847, 16)
📈 Class distribution:
   Voted posts (positive): 1247
   Non-voted posts (negative): 1600
   Balance ratio: 44%

🔍 Quick insights:
   Most common domains you voted on: {'github.com': 89, 'arxiv.org': 23, 'blog.example.com': 18}
   Average score of posts you voted on: 156.3
   Average score of posts you didn't vote on: 67.8
```

Ready to discover what makes you click! 🎯
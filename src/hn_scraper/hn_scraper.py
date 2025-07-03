"""
Hacker News data extraction system for building interest classification datasets.
"""

import requests
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging
import json
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HackerNewsDataExtractor:
    """Extract Hacker News voting data for ML classification."""
    
    def __init__(self, username: str, rate_limit: float = 1.0):
        """
        Initialize the extractor.
        
        Args:
            username: Your Hacker News username
            rate_limit: Seconds to wait between requests
        """
        self.username = username
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # HN API endpoints
        self.hn_api_base = "https://hacker-news.firebaseio.com/v0"
        self.favorites_api = "https://hnfavs.reactual.autocode.gg"
        
        # Updated HN session cookies with proper user authentication
        self.default_cookies = {
            '_ga': 'GA1.2.1256434543.1739713836',
            'ga_CHJJ1RJL5K': 'GS2.2.s1750096818$o20$g0$t1750096818$j60$l0$h0',
            '_gid': 'GA1.2.1658197836.1750096814',
            '_sso.key': 'G2tcSNgtBxpZWdOaSJSCZJ-JAWp54ilg',
            'amp_dd1bb8': 'aCMFocrdL8Jy-urhtQv1Jw...1im5r8ar3.1im5r8ar4.0.6.6',
            'user': '3dm4r&Ss0vVLSOdNqwbqoLG294jNCgo9BI0UWb'
        }
        
    def _rate_limit_request(self):
        """Apply rate limiting between requests."""
        time.sleep(self.rate_limit)
    
    def get_favorites(self, limit_pages: int = 10) -> List[Dict]:
        """
        Get user's favorited items by scraping HN directly.
        
        Args:
            limit_pages: Number of pages to fetch (30 items per page)
            
        Returns:
            List of favorite items with id, title, link
        """
        logger.info(f"Fetching favorites for user: {self.username}")
        
        # Use session cookies for authentication
        if self.default_cookies:
            self.session.cookies.update(self.default_cookies)
        
        favorites = []
        page = 1
        
        while page <= limit_pages:
            try:
                self._rate_limit_request()
                
                url = f"https://news.ycombinator.com/favorites"
                params = {'id': self.username, 'p': page}
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find story rows
                story_rows = soup.find_all('tr', class_='athing')
                
                if not story_rows:
                    logger.info(f"No more favorites found at page {page}")
                    break
                
                for row in story_rows:
                    try:
                        story_id = row.get('id')
                        title_cell = row.find('span', class_='titleline')
                        
                        if title_cell and story_id:
                            title_link = title_cell.find('a')
                            title = title_link.text.strip() if title_link else ""
                            url = title_link.get('href', '') if title_link else ""
                            
                            favorites.append({
                                'id': story_id,
                                'title': title,
                                'link': url,  # Keep 'link' for compatibility
                                'url': url,
                                'source': 'favorites'
                            })
                    
                    except Exception as e:
                        logger.warning(f"Error parsing favorite row: {e}")
                        continue
                
                logger.info(f"Page {page}: Found {len(story_rows)} favorite posts")
                page += 1
                
            except Exception as e:
                logger.error(f"Error fetching favorites page {page}: {e}")
                break
        
        logger.info(f"Total favorites retrieved: {len(favorites)}")
        return favorites
    
    def get_upvoted_posts(self, session_cookies: Optional[Dict] = None) -> List[Dict]:
        """
        Scrape upvoted posts from HN (requires authentication).
        
        Args:
            session_cookies: Dictionary of cookies for authentication
            
        Returns:
            List of upvoted post data
        """
        logger.info(f"Fetching upvoted posts for user: {self.username}")
        
        # Use provided cookies or default ones
        cookies_to_use = session_cookies if session_cookies else self.default_cookies
        if cookies_to_use:
            self.session.cookies.update(cookies_to_use)
            logger.info("Using session cookies for authentication")
        
        upvoted_posts = []
        page = 1
        
        while True:
            try:
                self._rate_limit_request()
                
                url = f"https://news.ycombinator.com/upvoted"
                params = {'id': self.username, 'p': page}
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find story rows
                story_rows = soup.find_all('tr', class_='athing')
                
                if not story_rows:
                    logger.info(f"No more upvoted posts found at page {page}")
                    break
                
                for row in story_rows:
                    try:
                        story_id = row.get('id')
                        title_cell = row.find('span', class_='titleline')
                        
                        if title_cell and story_id:
                            title_link = title_cell.find('a')
                            title = title_link.text.strip() if title_link else ""
                            url = title_link.get('href', '') if title_link else ""
                            
                            upvoted_posts.append({
                                'id': story_id,
                                'title': title,
                                'url': url,
                                'source': 'upvoted'
                            })
                    
                    except Exception as e:
                        logger.warning(f"Error parsing story row: {e}")
                        continue
                
                logger.info(f"Page {page}: Found {len(story_rows)} upvoted posts")
                page += 1
                
                # Limit to prevent excessive requests
                if page > 50:
                    logger.info("Reached page limit (50)")
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching upvoted posts page {page}: {e}")
                break
        
        logger.info(f"Total upvoted posts retrieved: {len(upvoted_posts)}")
        return upvoted_posts
    
    def get_hn_item_details(self, item_id: str) -> Optional[Dict]:
        """
        Get detailed information about an HN item using the official API.
        
        Args:
            item_id: HN item ID
            
        Returns:
            Item details or None if not found
        """
        try:
            self._rate_limit_request()
            
            url = f"{self.hn_api_base}/item/{item_id}.json"
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.warning(f"Error fetching item {item_id}: {e}")
            return None
    
    def sample_non_voted_posts(self, num_samples: int = 1000, 
                              days_back: int = 30) -> List[Dict]:
        """
        Sample posts that the user didn't vote on for negative examples.
        
        Args:
            num_samples: Number of negative examples to collect
            days_back: How many days back to sample from
            
        Returns:
            List of non-voted post data
        """
        logger.info(f"Sampling {num_samples} non-voted posts from last {days_back} days")
        
        # Get recent story IDs from different story types
        story_types = ['topstories', 'newstories', 'beststories', 'askstories', 'showstories']
        all_story_ids = set()
        
        for story_type in story_types:
            try:
                self._rate_limit_request()
                url = f"{self.hn_api_base}/{story_type}.json"
                response = self.session.get(url)
                response.raise_for_status()
                
                story_ids = response.json()[:200]  # Get top 200 from each category
                all_story_ids.update(story_ids)
                
            except Exception as e:
                logger.warning(f"Error fetching {story_type}: {e}")
                continue
        
        logger.info(f"Found {len(all_story_ids)} total story IDs")
        
        # Sample random stories and filter by date
        sampled_stories = []
        story_ids_list = list(all_story_ids)
        random.shuffle(story_ids_list)
        
        cutoff_time = (datetime.now() - timedelta(days=days_back)).timestamp()
        
        for story_id in tqdm(story_ids_list[:num_samples * 2], desc="Sampling stories"):
            item_details = self.get_hn_item_details(str(story_id))
            
            if (item_details and 
                item_details.get('time', 0) > cutoff_time and
                item_details.get('type') == 'story'):
                
                sampled_stories.append({
                    'id': str(story_id),
                    'title': item_details.get('title', ''),
                    'url': item_details.get('url', ''),
                    'source': 'non_voted'
                })
                
                if len(sampled_stories) >= num_samples:
                    break
        
        logger.info(f"Sampled {len(sampled_stories)} non-voted posts")
        return sampled_stories
    
    def enrich_posts_with_metadata(self, posts: List[Dict]) -> List[Dict]:
        """
        Enrich posts with additional metadata from HN API.
        
        Args:
            posts: List of posts to enrich
            
        Returns:
            Enriched posts with metadata
        """
        logger.info(f"Enriching {len(posts)} posts with metadata")
        
        enriched_posts = []
        
        for post in tqdm(posts, desc="Enriching posts"):
            item_details = self.get_hn_item_details(post['id'])
            
            if item_details:
                # Extract features
                enriched_post = {
                    'post_id': post['id'],
                    'title': item_details.get('title', post.get('title', '')),
                    'url': item_details.get('url', post.get('url', '')),
                    'score': item_details.get('score', 0),
                    'descendants': item_details.get('descendants', 0),  # comment count
                    'time': item_details.get('time', 0),
                    'author': item_details.get('by', ''),
                    'type': item_details.get('type', 'story'),
                    'your_vote': 1 if post['source'] in ['upvoted', 'favorites'] else 0,
                    'source': post['source']
                }
                
                # Add time-based features
                if enriched_post['time']:
                    dt = datetime.fromtimestamp(enriched_post['time'])
                    enriched_post['date'] = dt.strftime('%Y-%m-%d')
                    enriched_post['hour'] = dt.hour
                    enriched_post['day_of_week'] = dt.weekday()
                
                # Add URL domain
                if enriched_post['url']:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(enriched_post['url']).netloc
                        enriched_post['domain'] = domain.replace('www.', '')
                    except:
                        enriched_post['domain'] = ''
                else:
                    enriched_post['domain'] = 'news.ycombinator.com'  # Self posts
                
                # Detect post category
                title_lower = enriched_post['title'].lower()
                if title_lower.startswith('ask hn:'):
                    enriched_post['category'] = 'ask'
                elif title_lower.startswith('show hn:'):
                    enriched_post['category'] = 'show'
                elif 'hiring' in title_lower or 'job' in title_lower:
                    enriched_post['category'] = 'job'
                else:
                    enriched_post['category'] = 'story'
                
                enriched_posts.append(enriched_post)
            else:
                logger.warning(f"Could not enrich post {post['id']}")
        
        logger.info(f"Successfully enriched {len(enriched_posts)} posts")
        return enriched_posts
    
    def create_dataset(self, output_file: str = 'hn_dataset.csv', 
                      negative_samples: int = 1000) -> pd.DataFrame:
        """
        Create complete dataset with positive and negative examples.
        
        Args:
            output_file: Path to save the CSV file
            negative_samples: Number of negative examples to include
            
        Returns:
            Complete dataset as pandas DataFrame
        """
        logger.info("Creating complete Hacker News dataset")
        
        # Collect positive examples
        favorites = self.get_favorites()
        upvoted = self.get_upvoted_posts()
        
        # Mark favorites
        for fav in favorites:
            fav['source'] = 'favorites'
        
        # Combine positive examples
        positive_posts = favorites + upvoted
        
        # Remove duplicates based on ID
        seen_ids = set()
        unique_positive = []
        for post in positive_posts:
            if post['id'] not in seen_ids:
                unique_positive.append(post)
                seen_ids.add(post['id'])
        
        logger.info(f"Found {len(unique_positive)} unique positive examples")
        
        # Sample negative examples
        negative_posts = self.sample_non_voted_posts(negative_samples)
        
        # Filter out any negative posts that are actually in positive set
        negative_filtered = [
            post for post in negative_posts 
            if post['id'] not in seen_ids
        ]
        
        logger.info(f"Using {len(negative_filtered)} negative examples")
        
        # Combine all posts
        all_posts = unique_positive + negative_filtered
        
        # Enrich with metadata
        enriched_posts = self.enrich_posts_with_metadata(all_posts)
        
        # Create DataFrame
        df = pd.DataFrame(enriched_posts)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Dataset saved to {output_file}")
        
        # Print summary statistics
        logger.info(f"Dataset summary:")
        logger.info(f"Total posts: {len(df)}")
        logger.info(f"Positive examples: {df['your_vote'].sum()}")
        logger.info(f"Negative examples: {len(df) - df['your_vote'].sum()}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df


def main():
    """Example usage of the HackerNewsDataExtractor."""
    
    # Initialize extractor - replace with your HN username
    extractor = HackerNewsDataExtractor("your_username_here")
    
    # Create dataset
    df = extractor.create_dataset(
        output_file='hn_interest_dataset.csv',
        negative_samples=1000
    )
    
    print("\nDataset created successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Single script to scrape Hacker News and generate dataset.
"""

import asyncio
import json
import logging
import time
import getpass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
import random

import pandas as pd
import aiohttp
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PostData:
    """Structured data for HN posts."""
    id: str
    title: str
    url: str
    source: str
    score: int = 0
    descendants: int = 0
    time: int = 0
    author: str = ""
    domain: str = ""
    category: str = ""
    your_vote: int = 0
    date: str = ""
    hour: int = 0
    day_of_week: int = 0


class HNScraper:
    """Complete HN scraper in one class."""
    
    def __init__(self, username: str, password: str, 
                 max_upvoted_pages: int = 20,
                 headless: bool = True):
        self.username = username
        self.password = password
        self.max_upvoted_pages = max_upvoted_pages
        self.headless = headless
        self.is_authenticated = False
        
    async def authenticate(self, page) -> bool:
        """Authenticate with HN."""
        logger.info(f"Authenticating user: {self.username}")
        
        try:
            await page.goto('https://news.ycombinator.com/login', timeout=30000)
            await page.fill('input[name="acct"]', self.username)
            await page.fill('input[name="pw"]', self.password)
            await page.click('input[type="submit"]')
            await page.wait_for_load_state('networkidle')
            
            # Check if login successful
            if 'login' in page.url:
                error = await page.query_selector('font[color="#ff0000"]')
                if error:
                    error_text = await error.text_content()
                    raise Exception(f"Login failed: {error_text}")
                else:
                    raise Exception("Login failed: Still on login page")
            
            # Verify with logout link
            logout_link = await page.query_selector('a[href="logout"]')
            if logout_link:
                self.is_authenticated = True
                logger.info("Authentication successful")
                return True
            else:
                logger.warning("Authentication unclear - proceeding")
                self.is_authenticated = True
                return True
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    async def extract_posts_from_page(self, page) -> List[PostData]:
        """Extract posts from current page."""
        posts = []
        
        try:
            await page.wait_for_selector('tr.athing', timeout=10000)
            story_rows = await page.query_selector_all('tr.athing')
            
            for row in story_rows:
                try:
                    story_id = await row.get_attribute('id')
                    if not story_id:
                        continue
                    
                    title_elem = await row.query_selector('.titleline a')
                    if not title_elem:
                        continue
                    
                    title = await title_elem.text_content()
                    url = await title_elem.get_attribute('href') or ''
                    
                    posts.append(PostData(
                        id=story_id,
                        title=title.strip(),
                        url=url,
                        source='',  # Will be set by caller
                        your_vote=1
                    ))
                    
                except Exception as e:
                    logger.debug(f"Error parsing row: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"No posts found on page: {e}")
        
        return posts
    
    
    async def extract_upvoted(self, context) -> List[PostData]:
        """Extract upvoted posts."""
        logger.info("Extracting upvoted posts...")
        all_upvoted = []
        
        for page_num in range(1, self.max_upvoted_pages + 1):
            try:
                page = await context.new_page()
                url = f"https://news.ycombinator.com/upvoted?id={self.username}&p={page_num}"
                await page.goto(url, timeout=30000)
                
                # Check auth
                if 'login' in page.url:
                    logger.warning(f"Re-auth needed for upvoted page {page_num}")
                    if not await self.authenticate(page):
                        await page.close()
                        break
                    await page.goto(url, timeout=30000)
                
                posts = await self.extract_posts_from_page(page)
                
                for post in posts:
                    post.source = 'upvoted'
                    post.your_vote = 1
                
                all_upvoted.extend(posts)
                logger.info(f"Upvoted page {page_num}: {len(posts)} posts")
                
                if not posts:
                    await page.close()
                    break
                
                await page.close()
                await asyncio.sleep(1)  # Rate limit
                
            except Exception as e:
                logger.error(f"Error on upvoted page {page_num}: {e}")
                if 'page' in locals():
                    await page.close()
                continue
        
        logger.info(f"Total upvoted: {len(all_upvoted)}")
        return all_upvoted
    
    async def get_recent_posts_with_vote_status(self, context, upvoted_ids: set) -> List[PostData]:
        """Get recent posts from HN front page and check if user voted on them."""
        logger.info("Getting recent posts from front page and checking vote status...")
        
        all_posts = []
        
        # Get multiple pages from front page (1-10)
        for page_num in range(1, 11):  # Pages 1-10
            try:
                page = await context.new_page()
                if page_num == 1:
                    url = "https://news.ycombinator.com"
                else:
                    url = f"https://news.ycombinator.com?p={page_num}"
                await page.goto(url, timeout=30000)
                
                posts = await self.extract_posts_from_page(page)
                
                for post in posts:
                    if post.id in upvoted_ids:
                        post.source = 'upvoted'
                        post.your_vote = 1
                    else:
                        post.source = 'non_voted'
                        post.your_vote = 0
                    
                    all_posts.append(post)
                
                logger.info(f"Got {len(posts)} posts from front page {page_num}")
                
                await page.close()
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"Error getting front page {page_num}: {e}")
                if 'page' in locals():
                    await page.close()
                continue
        
        # Remove duplicates
        seen_ids = set()
        unique_posts = []
        for post in all_posts:
            if post.id not in seen_ids:
                seen_ids.add(post.id)
                unique_posts.append(post)
        
        voted_count = sum(1 for p in unique_posts if p.your_vote == 1)
        logger.info(f"Total recent posts: {len(unique_posts)} ({voted_count} voted, {len(unique_posts) - voted_count} not voted)")
        
        return unique_posts
    
    async def _fetch_story(self, session, story_id: int, cutoff_time: float) -> Optional[PostData]:
        """Fetch single story details."""
        try:
            url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if (data and 
                        data.get('time', 0) > cutoff_time and 
                        data.get('type') == 'story' and
                        data.get('title')):
                        
                        return PostData(
                            id=str(story_id),
                            title=data.get('title', ''),
                            url=data.get('url', ''),
                            source='non_voted',
                            score=data.get('score', 0),
                            descendants=data.get('descendants', 0),
                            time=data.get('time', 0),
                            author=data.get('by', ''),
                            your_vote=0
                        )
        except:
            pass
        return None
    
    def enrich_posts(self, posts: List[PostData]) -> List[PostData]:
        """Add metadata to posts."""
        logger.info(f"Enriching {len(posts)} posts...")
        
        for post in posts:
            try:
                # Add time features
                if post.time:
                    dt = datetime.fromtimestamp(post.time)
                    post.date = dt.strftime('%Y-%m-%d')
                    post.hour = dt.hour
                    post.day_of_week = dt.weekday()
                
                # Add domain
                if post.url:
                    try:
                        post.domain = urlparse(post.url).netloc.replace('www.', '')
                    except:
                        post.domain = ''
                else:
                    post.domain = 'news.ycombinator.com'
                
                # Add category
                title_lower = post.title.lower()
                if title_lower.startswith('ask hn:'):
                    post.category = 'ask'
                elif title_lower.startswith('show hn:'):
                    post.category = 'show'
                elif 'hiring' in title_lower or 'job' in title_lower:
                    post.category = 'job'
                else:
                    post.category = 'story'
                    
            except Exception as e:
                logger.warning(f"Error enriching post {post.id}: {e}")
        
        return posts
    
    async def create_dataset(self) -> pd.DataFrame:
        """Create complete dataset."""
        logger.info("Creating HN dataset...")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            
            try:
                context = await browser.new_context()
                main_page = await context.new_page()
                
                # Authenticate
                if not await self.authenticate(main_page):
                    raise Exception("Authentication failed")
                
                # Extract upvoted posts to know what user voted on
                upvoted = await self.extract_upvoted(context)
                
                # Create set of upvoted IDs for quick lookup
                upvoted_ids = {post.id for post in upvoted}
                logger.info(f"Found {len(upvoted_ids)} upvoted posts")
                
                # Get recent posts and check vote status
                recent_posts = await self.get_recent_posts_with_vote_status(context, upvoted_ids)
                
                # Combine upvoted posts (from history) with recent posts
                # This ensures we have all upvoted posts plus recent activity
                all_posts = []
                seen_ids = set()
                
                # Add all upvoted posts first
                for post in upvoted:
                    if post.id not in seen_ids:
                        all_posts.append(post)
                        seen_ids.add(post.id)
                
                # Add recent posts (both voted and non-voted)
                for post in recent_posts:
                    if post.id not in seen_ids:
                        all_posts.append(post)
                        seen_ids.add(post.id)
                
                # Count statistics
                voted_count = sum(1 for p in all_posts if p.your_vote == 1)
                not_voted_count = len(all_posts) - voted_count
                
                logger.info(f"Total dataset: {len(all_posts)} posts ({voted_count} voted, {not_voted_count} not voted)")
                
                # Enrich all posts
                enriched_posts = self.enrich_posts(all_posts)
                
                # Create DataFrame
                df = pd.DataFrame([post.__dict__ for post in enriched_posts])
                
                logger.info("Dataset creation complete!")
                return df
                
            finally:
                await browser.close()


async def main():
    """Main function to run the scraper."""
    print("🎯 HN Dataset Creator")
    print("=" * 40)
    
    # Get credentials
    username = input("HN Username: ")
    password = getpass.getpass("HN Password: ")
    
    # Get configuration
    print("\nDataset size:")
    print("1. Small (10 upvoted pages + recent posts) ~5 min")
    print("2. Medium (20 upvoted pages + recent posts) ~10 min")
    print("3. Large (40 upvoted pages + recent posts) ~20 min")
    
    choice = input("Choose (1-3): ").strip()
    
    if choice == "1":
        max_up = 10
        size_name = "small"
    elif choice == "3":
        max_up = 40
        size_name = "large"
    else:
        max_up = 20
        size_name = "medium"
    
    # Create output path
    project_root = Path(__file__).parent.parent.parent  # Go up to main project root
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = data_dir / f"hn_export_{username}_{size_name}_{timestamp}.csv"
    
    print(f"\nConfiguration:")
    print(f"  Upvoted pages: {max_up}")
    print(f"  Recent pages: Front page (10 pages)")
    print(f"  Output: {output_file}")
    
    if input("\nProceed? (y/N): ").lower() != 'y':
        print("Cancelled.")
        return
    
    try:
        # Create scraper
        scraper = HNScraper(
            username=username,
            password=password,
            max_upvoted_pages=max_up,
            headless=choice != "1"  # Keep visible for small datasets
        )
        
        # Create dataset
        df = await scraper.create_dataset()
        
        # Save dataset
        df.to_csv(output_file, index=False)
        
        # Show results
        positive_count = df['your_vote'].sum()
        total_count = len(df)
        
        print(f"\n✅ Dataset created!")
        print(f"📁 File: {output_file}")
        print(f"📊 Total posts: {total_count}")
        print(f"📊 Positive: {positive_count}")
        print(f"📊 Negative: {total_count - positive_count}")
        print(f"📊 Balance: {positive_count/total_count*100:.1f}% positive")
        
        # Show sample
        if len(df) > 0:
            print(f"\n📝 Sample:")
            print(df[['title', 'domain', 'category', 'your_vote', 'source']].head())
        
        print(f"\n💡 Next: Copy to DSPy data directory:")
        print(f"cp '{output_file}' data/raw/export.csv")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
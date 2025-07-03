#!/usr/bin/env python3
"""
Quick test script to verify HN scraper works with your cookies.
"""

from src.hn_scraper.hn_scraper import HackerNewsDataExtractor

def test_scraper(username=None):
    # Get your username
    if not username:
        username = "edmaroferreira"  # Default for testing
    
    if not username:
        print("Username required!")
        return
    
    print(f"Testing HN scraper for user: {username}")
    
    # Initialize extractor
    extractor = HackerNewsDataExtractor(username, rate_limit=0.5)
    
    # Test favorites (no auth needed)
    print("\n1. Testing favorites extraction...")
    favorites = extractor.get_favorites(limit_pages=1)
    print(f"   Found {len(favorites)} favorites")
    if favorites:
        print(f"   Sample: {favorites[0].get('title', 'N/A')}")
    
    # Test upvoted posts (uses your cookies)
    print("\n2. Testing upvoted posts extraction...")
    upvoted = extractor.get_upvoted_posts()
    print(f"   Found {len(upvoted)} upvoted posts")
    if upvoted:
        print(f"   Sample: {upvoted[0].get('title', 'N/A')}")
    
    # Test HN API item details
    if upvoted or favorites:
        print("\n3. Testing HN API enrichment...")
        test_item = upvoted[0] if upvoted else favorites[0]
        details = extractor.get_hn_item_details(test_item['id'])
        if details:
            print(f"   Score: {details.get('score', 'N/A')}")
            print(f"   Comments: {details.get('descendants', 'N/A')}")
            print(f"   Author: {details.get('by', 'N/A')}")
        else:
            print("   Could not fetch item details")
    
    total_positive = len(favorites) + len(upvoted)
    print(f"\n✅ Total positive examples available: {total_positive}")
    
    if total_positive > 0:
        print("🎯 Ready to create your interest classification dataset!")
        print("Run: python -c \"from src.hn_scraper.hn_scraper import HackerNewsDataExtractor; extractor = HackerNewsDataExtractor('{}'); extractor.create_dataset()\"".format(username))
    else:
        print("⚠️  No data found. Check your username and cookies.")

if __name__ == "__main__":
    test_scraper()
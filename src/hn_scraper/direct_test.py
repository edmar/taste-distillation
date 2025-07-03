#!/usr/bin/env python3
"""
Direct test to check HN access with your cookies
"""

import requests
from bs4 import BeautifulSoup

def test_hn_access():
    """Test direct access to HN with cookies"""
    
    # Your browser cookies
    cookies = {
        '_ga': 'GA1.2.1256434543.1739713836',
        'ga_CHJJ1RJL5K': 'GS2.2.s1750096818$o20$g0$t1750096818$j60$l0$h0',
        '_gid': 'GA1.2.1658197836.1750096814',
        '_sso.key': 'G2tcSNgtBxpZWdOaSJSCZJ-JAWp54ilg',
        'amp_dd1bb8': 'aCMFocrdL8Jy-urhtQv1Jw...1im5r8ar3.1im5r8ar4.0.6.6'
    }
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    session.cookies.update(cookies)
    
    # Test different usernames
    test_usernames = ['edmaroferreira', 'edmar', 'edmaroferreira1']
    
    for username in test_usernames:
        print(f"\n🔍 Testing username: {username}")
        
        # Try favorites page
        try:
            fav_url = f"https://news.ycombinator.com/favorites?id={username}"
            response = session.get(fav_url)
            print(f"   Favorites page status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                stories = soup.find_all('tr', class_='athing')
                print(f"   Found {len(stories)} favorite stories")
                
                if stories:
                    first_story = stories[0]
                    title_elem = first_story.find('span', class_='titleline')
                    if title_elem:
                        title = title_elem.find('a')
                        if title:
                            print(f"   Sample favorite: {title.text[:50]}...")
            
        except Exception as e:
            print(f"   Favorites error: {e}")
        
        # Try upvoted page
        try:
            upvoted_url = f"https://news.ycombinator.com/upvoted?id={username}"
            response = session.get(upvoted_url)
            print(f"   Upvoted page status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                stories = soup.find_all('tr', class_='athing')
                print(f"   Found {len(stories)} upvoted stories")
                
                if stories:
                    first_story = stories[0]
                    title_elem = first_story.find('span', class_='titleline')
                    if title_elem:
                        title = title_elem.find('a')
                        if title:
                            print(f"   Sample upvoted: {title.text[:50]}...")
                
                # Check if login required
                if "login" in response.text.lower() or len(stories) == 0:
                    print("   ⚠️  May need authentication or user has no public votes")
            
        except Exception as e:
            print(f"   Upvoted error: {e}")
    
    # Test HN API
    print(f"\n🔍 Testing HN API access...")
    try:
        api_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        response = requests.get(api_url)
        print(f"   API status: {response.status_code}")
        
        if response.status_code == 200:
            stories = response.json()
            print(f"   Found {len(stories)} top stories")
            
            # Test getting item details
            if stories:
                item_id = stories[0]
                item_url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
                item_response = requests.get(item_url)
                if item_response.status_code == 200:
                    item_data = item_response.json()
                    print(f"   Sample story: {item_data.get('title', 'N/A')[:50]}...")
                    print(f"   Score: {item_data.get('score', 'N/A')}")
        
    except Exception as e:
        print(f"   API error: {e}")

if __name__ == "__main__":
    test_hn_access()
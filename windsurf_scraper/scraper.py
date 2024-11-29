import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from datetime import datetime
import hashlib
import json
import argparse
import re
from collections import deque
import nltk
from nltk.tokenize import sent_tokenize
import html2text
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import markdown
import subprocess
from pathlib import Path

class DocumentationScraper:
    def __init__(self, base_dir="doc-resource"):
        self.base_dir = base_dir
        self.content_dir = os.path.join(base_dir, "content")
        self.index_dir = os.path.join(base_dir, "index")
        self.embeddings_dir = os.path.join(base_dir, "embeddings")
        self.repos_dir = os.path.join(base_dir, "repos")
        
        # Create necessary directories
        for directory in [self.content_dir, self.index_dir, self.embeddings_dir, self.repos_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.visited_urls = set()
        
        # Initialize HTML to text converter
        self.text_converter = html2text.HTML2Text()
        self.text_converter.ignore_links = False
        self.text_converter.ignore_images = True
        self.text_converter.ignore_tables = False
        
        # Initialize NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initialize sentence transformer for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load or create metadata
        self.metadata_file = os.path.join(base_dir, 'metadata.json')
        if not os.path.exists(self.metadata_file):
            self._save_metadata({})
        else:
            metadata = self._load_metadata()
            for data in metadata.values():
                self.visited_urls.add(data['url'])

    def _save_metadata(self, metadata):
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _get_safe_filename(self, url, extension='.json'):
        """Generate a safe filename from URL"""
        parsed = urlparse(url)
        path = parsed.netloc + parsed.path
        if not parsed.path or parsed.path == '/':
            path += 'index'
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        safe_filename = "".join(x if x.isalnum() or x in '-._' else '_' for x in path)
        return f"{safe_filename}_{url_hash}{extension}"

    def _extract_content(self, soup, url):
        """Extract and structure content from HTML"""
        # Get main content (customize selectors based on common documentation sites)
        main_content = soup.find('div', class_='document') or soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if not main_content:
            main_content = soup

        # Remove navigation elements
        for nav in main_content.find_all(['nav', 'div'], class_=['headerlink', 'nav', 'navigation']):
            nav.decompose()
            
        # Convert to markdown-like text
        text_content = self.text_converter.handle(str(main_content))
        
        # Split into sections
        sections = []
        current_section = {"title": "", "content": [], "subsections": []}
        
        # Clean up empty lines and navigation artifacts
        lines = [line.strip() for line in text_content.split('\n') if line.strip() and not line.strip().startswith('[')]
        
        for line in lines:
            if line.startswith('# '):
                if current_section["content"] or current_section["subsections"]:
                    sections.append(current_section)
                current_section = {"title": line[2:], "content": [], "subsections": []}
            elif line.startswith('## '):
                current_section["subsections"].append({"title": line[3:], "content": []})
            else:
                if current_section["subsections"]:
                    current_section["subsections"][-1]["content"].append(line)
                else:
                    current_section["content"].append(line)
        
        if current_section["content"] or current_section["subsections"]:
            sections.append(current_section)
        
        # Create a summary from the first meaningful content
        summary_text = []
        for section in sections:
            if section["content"]:
                clean_content = [line for line in section["content"] if not line.startswith(('[', '|', '-'))]
                if clean_content:
                    summary_text.extend(clean_content[:3])
            if len(summary_text) >= 3:
                break
                
            # If we haven't found enough content, check subsections
            if len(summary_text) < 3:
                for subsection in section["subsections"]:
                    clean_content = [line for line in subsection["content"] if not line.startswith(('[', '|', '-'))]
                    if clean_content:
                        summary_text.extend(clean_content[:3-len(summary_text)])
                    if len(summary_text) >= 3:
                        break
        
        # Create structured content
        structured_content = {
            "url": url,
            "title": soup.title.string if soup.title else "No title",
            "sections": sections,
            "summary": " ".join(summary_text[:3]) if summary_text else "No summary available",
            "last_updated": datetime.now().isoformat()
        }
        
        return structured_content

    def _generate_embeddings(self, content):
        """Generate embeddings for content sections"""
        embeddings = {}
        
        # Generate embeddings for sections and their content
        for section in content["sections"]:
            # Combine section title and content
            section_text = section["title"] + "\n" + "\n".join(section["content"])
            embeddings[section["title"]] = self.model.encode(section_text)
            
            # Generate embeddings for subsections
            for subsection in section["subsections"]:
                subsection_text = subsection["title"] + "\n" + "\n".join(subsection["content"])
                embeddings[subsection["title"]] = self.model.encode(subsection_text)
        
        return embeddings

    def _save_embeddings(self, embeddings, filename):
        """Save embeddings dictionary to a file"""
        embeddings_data = {
            'vectors': np.array(list(embeddings.values())),
            'keys': list(embeddings.keys())
        }
        np.savez(filename, **embeddings_data)

    def _load_embeddings(self, filename):
        """Load embeddings dictionary from a file"""
        with np.load(filename) as data:
            return dict(zip(data['keys'], data['vectors']))

    def _is_valid_doc_link(self, url, base_url):
        """Check if a URL is a valid documentation link to follow"""
        if not url:
            return False

        url = urljoin(base_url, url)
        parsed_url = urlparse(url)
        parsed_base = urlparse(base_url)
        
        if url in self.visited_urls:
            return False
        
        if parsed_url.netloc != parsed_base.netloc:
            return False
        
        skip_patterns = [
            r'/static/',
            r'/assets/',
            r'/images/',
            r'/css/',
            r'/js/',
            r'#',
            r'\?',
            r'mailto:',
            r'tel:',
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, url):
                return False
        
        skip_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.css', '.js', '.ico']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        return True

    def scrape_url(self, url, max_depth=2, current_depth=0):
        """Scrape content from a URL and follow relevant links"""
        if current_depth > max_depth or url in self.visited_urls:
            return None
            
        try:
            print(f"Scraping (depth {current_depth}): {url}")
            self.visited_urls.add(url)
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract and structure content
            content = self._extract_content(soup, url)
            
            # Generate embeddings
            embeddings = self._generate_embeddings(content)
            
            # Save structured content
            content_file = os.path.join(self.content_dir, self._get_safe_filename(url))
            with open(content_file, 'w') as f:
                json.dump(content, f, indent=2)
            
            # Save embeddings
            embeddings_file = os.path.join(self.embeddings_dir, self._get_safe_filename(url, '.npz'))
            self._save_embeddings(embeddings, embeddings_file)
            
            # Update metadata
            metadata = self._load_metadata()
            metadata[os.path.basename(content_file)] = {
                'url': url,
                'title': content['title'],
                'date_scraped': datetime.now().isoformat(),
                'content_type': 'github' if 'github.com' in url else 'documentation',
                'depth': current_depth,
                'summary': content['summary']
            }
            self._save_metadata(metadata)
            
            print(f"Successfully scraped and processed: {url}")
            
            # Find and follow relevant links
            if current_depth < max_depth:
                links_to_follow = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if self._is_valid_doc_link(href, url):
                        full_url = urljoin(url, href)
                        links_to_follow.append(full_url)
                
                for link_url in links_to_follow:
                    self.scrape_url(link_url, max_depth, current_depth + 1)
            
            return content_file
            
        except requests.RequestException as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def clone_github_repo(self, repo_url):
        """Clone a GitHub repository and process its contents.
        
        Args:
            repo_url (str): URL of the GitHub repository (e.g., 'https://github.com/username/repo')
        """
        # Extract repo name from URL
        repo_name = repo_url.rstrip('/').split('/')[-1]
        repo_path = os.path.join(self.repos_dir, repo_name)
        
        # Clone or update the repository
        if os.path.exists(repo_path):
            print(f"Repository {repo_name} exists, pulling latest changes...")
            subprocess.run(['git', '-C', repo_path, 'pull'], check=True)
        else:
            print(f"Cloning repository {repo_name}...")
            subprocess.run(['git', 'clone', repo_url, repo_path], check=True)
        
        # Process repository contents
        repo_files = []
        for root, _, files in os.walk(repo_path):
            if '.git' in root:  # Skip .git directory
                continue
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Generate a unique ID for this file
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_id = self._get_safe_filename(f"{repo_url}/{relative_path}")
                    
                    # Store the content
                    content_path = os.path.join(self.content_dir, file_id)
                    with open(content_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'url': f"{repo_url}/blob/main/{relative_path}",
                            'path': relative_path,
                            'content': content,
                            'type': 'github_file'
                        }, f, indent=2)
                    
                    # Update metadata
                    metadata = self._load_metadata()
                    metadata[file_id] = {
                        'url': f"{repo_url}/blob/main/{relative_path}",
                        'title': relative_path,
                        'type': 'github_file',
                        'timestamp': datetime.now().isoformat(),
                        'repo_url': repo_url
                    }
                    self._save_metadata(metadata)
                    
                    repo_files.append(file_id)
                except UnicodeDecodeError:
                    # Skip binary files
                    continue
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return repo_files

    def detect_url_type(self, url):
        """Detect the type of URL (GitHub repo, documentation site, etc.)
        
        Returns:
            tuple: (type, parsed_url)
            where type is one of: 'github_repo', 'documentation', 'unknown'
        """
        parsed = urlparse(url)
        
        # GitHub repository detection
        if parsed.netloc == 'github.com':
            path_parts = [p for p in parsed.path.split('/') if p]
            if len(path_parts) >= 2:  # username/repo format
                return 'github_repo', url
        
        # Common documentation sites
        doc_domains = {
            'docs.python.org', 'docs.github.com', 'docs.docker.com', 
            'developer.mozilla.org', 'docs.aws.amazon.com', 'kubernetes.io',
            'docs.microsoft.com', 'cloud.google.com'
        }
        if parsed.netloc in doc_domains:
            return 'documentation', url
            
        # Try to detect if it's a documentation page by checking response headers and content
        try:
            response = self.session.head(url, allow_redirects=True)
            content_type = response.headers.get('content-type', '')
            
            if 'text/html' in content_type:
                # Make a GET request to check content
                response = self.session.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Common documentation indicators
                doc_indicators = [
                    'documentation', 'docs', 'api reference', 'developer guide',
                    'manual', 'reference', 'guide', 'tutorial'
                ]
                
                # Check title and meta description
                title = soup.title.string.lower() if soup.title else ''
                meta_desc = soup.find('meta', {'name': 'description'})
                desc = meta_desc.get('content', '').lower() if meta_desc else ''
                
                if any(indicator in title.lower() or indicator in desc.lower() 
                       for indicator in doc_indicators):
                    return 'documentation', url
        except Exception:
            pass
            
        return 'unknown', url

    def process_url(self, url, max_depth=2):
        """Process any URL automatically by detecting its type"""
        url_type, processed_url = self.detect_url_type(url)
        
        print(f"Detected URL type: {url_type}")
        
        if url_type == 'github_repo':
            return self.clone_github_repo(processed_url)
        elif url_type == 'documentation':
            return self.scrape_url(processed_url, max_depth=max_depth)
        else:
            print(f"Warning: Unable to determine URL type for {url}. Treating as documentation...")
            return self.scrape_url(processed_url, max_depth=max_depth)

    def refresh_docs(self, urls=None):
        """Refresh existing documentation or specific URLs"""
        metadata = self._load_metadata()
        urls_to_refresh = set()

        if urls:
            urls_to_refresh.update(urls)
        else:
            for data in metadata.values():
                urls_to_refresh.add(data['url'])

        if not urls_to_refresh:
            print("No documents to refresh.")
            return

        self.visited_urls.clear()
        
        print(f"Refreshing {len(urls_to_refresh)} documents...")
        for url in urls_to_refresh:
            print(f"\nRefreshing: {url}")
            original_depth = 0
            for data in metadata.values():
                if data['url'] == url:
                    original_depth = data.get('depth', 2)
                    break
            self.scrape_url(url, max_depth=original_depth)

    def semantic_search(self, query, top_k=5):
        """Search through documentation using semantic similarity"""
        print(f"Searching for: {query}")
        query_embedding = self.model.encode(query)
        
        results = []
        metadata = self._load_metadata()
        
        for filename in os.listdir(self.embeddings_dir):
            if not filename.endswith('.npz'):
                continue
                
            embeddings = self._load_embeddings(os.path.join(self.embeddings_dir, filename))
            content_file = os.path.join(self.content_dir, filename.replace('.npz', '.json'))
            
            if not os.path.exists(content_file):
                continue
                
            with open(content_file, 'r') as f:
                content = json.load(f)
            
            # Calculate similarities for each section
            for section_title, section_embedding in embeddings.items():
                similarity = np.dot(query_embedding, section_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(section_embedding)
                )
                
                results.append({
                    'url': content['url'],
                    'title': content['title'],
                    'section': section_title,
                    'similarity': similarity,
                    'summary': content['summary']
                })
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def list_scraped_pages(self):
        """List all scraped pages with their metadata"""
        metadata = self._load_metadata()
        for filename, data in metadata.items():
            print(f"\nFile: {filename}")
            print(f"Title: {data['title']}")
            print(f"URL: {data['url']}")
            print(f"Scraped: {data['date_scraped']}")
            print(f"Type: {data['content_type']}")
            print(f"Depth: {data.get('depth', 0)}")
            if 'summary' in data:
                print(f"Summary: {data['summary']}")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Scrape documentation and GitHub pages for local reference')
    parser.add_argument('urls', nargs='*', help='URLs to scrape (automatically detects GitHub repos and documentation sites)')
    parser.add_argument('-d', '--directory', default='doc-resource', help='Directory to store scraped content (default: doc-resource)')
    parser.add_argument('--depth', type=int, default=2, help='Maximum depth to follow links (default: 2)')
    parser.add_argument('--refresh', action='store_true', help='Refresh existing documentation')
    parser.add_argument('--refresh-urls', nargs='*', help='Refresh specific URLs')
    parser.add_argument('--search', help='Search documentation')
    parser.add_argument('--limit', type=int, default=5, help='Limit search results (default: 5)')
    parser.add_argument('-l', '--list', action='store_true', help='List all scraped pages')
    
    args = parser.parse_args()
    scraper = DocumentationScraper(args.directory)
    
    if args.search:
        results = scraper.semantic_search(args.search, limit=args.limit)
        for result in results:
            print("\n---")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Relevance: {result['similarity']:.2f}")
            print(f"Summary: {result['summary']}")
    elif args.refresh or args.refresh_urls:
        scraper.refresh_docs(args.refresh_urls)
    elif args.list:
        metadata = scraper._load_metadata()
        for doc_id, data in metadata.items():
            print(f"\nDocument ID: {doc_id}")
            print(f"Title: {data['title']}")
            print(f"URL: {data['url']}")
            print(f"Last Updated: {data['timestamp']}")
    elif args.urls:
        for url in args.urls:
            scraper.process_url(url, max_depth=args.depth)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

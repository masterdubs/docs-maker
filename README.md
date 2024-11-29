# AI-Friendly Documentation Scraper

A Python script to scrape and save documentation in an AI-optimized format for easy searching and reference.

## Features

- Scrapes web content from documentation sites and GitHub
- Converts HTML to structured, semantic content
- Generates embeddings for semantic search
- Follows and saves relevant documentation links
- Maintains searchable metadata and content indices
- Smart filtering of links to focus on documentation
- Refresh capability to update existing documentation

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The first run will download required NLTK data and the sentence transformer model.

## Directory Structure

- `doc-resource/`: Main directory for all scraped content
  - `content/`: Structured JSON content files
  - `embeddings/`: Neural embeddings for semantic search
  - `index/`: Search indices and metadata
  - `metadata.json`: Global metadata about all scraped pages

## Usage

### Basic Scraping
```bash
# Scrape a single URL (default depth: 2)
python scraper.py https://docs.python.org/3/

# Scrape with custom depth
python scraper.py --depth 3 https://docs.python.org/3/

# Scrape multiple URLs
python scraper.py https://docs.python.org/3/ https://docs.github.com/
```

### Semantic Search
```bash
# Search across all documentation
python scraper.py --search "how to handle exceptions in Python"

# Get more results
python scraper.py --search "async programming" --top-k 10
```

### Refreshing Documentation
```bash
# Refresh all existing documentation
python scraper.py --refresh

# Refresh specific URLs
python scraper.py --refresh-urls https://docs.python.org/3/ https://docs.github.com/
```

### Other Commands
```bash
# List all scraped pages
python scraper.py --list

# Change storage directory
python scraper.py -d custom-docs https://docs.python.org/3/
```

## AI-Friendly Features

1. **Structured Content Storage**
   - Content is parsed and stored in structured JSON format
   - Sections and subsections are properly identified
   - Maintains document hierarchy and relationships

2. **Semantic Search**
   - Uses sentence transformers for semantic understanding
   - Generates embeddings for each content section
   - Enables natural language queries
   - Returns relevant sections with similarity scores

3. **Smart Content Processing**
   - Extracts meaningful summaries
   - Preserves document structure
   - Removes noise (ads, navigation, etc.)
   - Maintains cross-references

4. **Metadata and Indexing**
   - Rich metadata for each document
   - Section-level granularity
   - Quick content retrieval
   - Semantic similarity search

## Notes

- Content is stored in both raw and processed formats
- Each section has its own embedding for precise search
- Links are followed only within the same domain
- Asset files (images, CSS, JS) are automatically filtered out
- Semantic search works across all scraped content

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

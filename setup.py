import nltk

def setup():
    """Download required NLTK data"""
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    print("Setup complete!")

if __name__ == "__main__":
    setup()

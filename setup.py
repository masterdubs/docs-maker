from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='windsurf-scraper',
    version='0.1.0',
    description='A tool for scraping and indexing documentation and GitHub repositories',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='dubs',
    author_email='',  # Add your email if desired
    url='https://github.com/dubs-subnet/windsurf-scraper',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'windsurf-scraper=windsurf_scraper.scraper:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
)

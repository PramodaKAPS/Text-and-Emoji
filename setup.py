import subprocess
import sys

def install_requirements():
    packages = [
        "datasets==2.18.0",
        "huggingface_hub==0.21.2", 
        "fsspec==2023.9.2",
        "transformers==4.53.2",
        "imbalanced-learn",
        "pandas",
        "numpy",
        "emoji==2.2.0",
        "gensim==4.3.2",
        "nltk==3.8.1",
        "scipy==1.11.4"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
    print(" All required packages installed successfully!")
    print(" Please restart your runtime: Runtime > Restart runtime")

def setup_cache_directory():
    import os
    cache_dir = "/root/huggingface_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Created cache directory: {cache_dir}")
    else:
        print(f"Using cache directory: {cache_dir}")
    return cache_dir

if __name__ == "__main__":
    install_requirements()

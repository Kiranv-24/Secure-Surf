
# Helper script to create test URL files
import os

def create_sample_test_file():
    """Create a sample test file with various URLs"""
    
    sample_urls = [
        "# Sample URLs for phishing detection testing",
        "# Format: url - yes/no (yes=phishing, no=legitimate)",
        "# Or just url for unlabeled testing",
        "",
        "# Legitimate websites",
        "https://google.com - no",
        "https://github.com - no", 
        "https://stackoverflow.com - no",
        "https://wikipedia.org - no",
        "https://microsoft.com - no",
        "",
        "# Suspicious/Phishing examples (replace with real test URLs)",
        "# https://fake-bank-login.com - yes",
        "# https://paypal-security-update.net - yes", 
        "# https://amazon-winner-prize.org - yes",
        "",
        "# Unlabeled URLs for testing",
        "https://reddit.com",
        "https://youtube.com",
        "https://twitter.com"
    ]
    
    filename = "sample_test_urls.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sample_urls))
    
    print(f"Created sample test file: {filename}")
    print("Edit this file with your actual test URLs before running batch_test.py")

if __name__ == "__main__":
    create_sample_test_file()

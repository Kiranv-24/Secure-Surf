
import requests
import Feature_extraction_ff1 as fex  
import numpy as np
import os
import joblib
import result as ress
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

model_path = "best_rf_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

def create_session_with_retry():
    """Create a requests session with enhanced retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1,
        raise_on_redirect=False,
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def preprocess_url(domain):
    if "." not in domain:
        raise ValueError("Invalid URL format")
    if not domain.startswith("http://") and not domain.startswith("https://"):
        domain = "https://" + domain
    if domain.startswith("https://www."):
        domain = "https://" + domain[12:]
    elif domain.startswith("http://www."):
        domain = "http://" + domain[11:]
    return domain

def fetch_url_content(url):
    """Enhanced URL fetching with comprehensive error handling"""
    session = create_session_with_retry()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        # Handle redirects manually to avoid infinite loops
        response = session.get(url, timeout=10, headers=headers, verify=False, allow_redirects=False)
        
        redirect_count = 0
        max_redirects = 5
        
        while response.status_code in [301, 302, 303, 307, 308] and redirect_count < max_redirects:
            redirect_url = response.headers.get('Location')
            if not redirect_url:
                break
            
            if redirect_url.startswith('/'):
                from urllib.parse import urljoin
                redirect_url = urljoin(url, redirect_url)
            
            response = session.get(redirect_url, timeout=10, headers=headers, verify=False, allow_redirects=False)
            redirect_count += 1
            url = redirect_url
        
        if redirect_count >= max_redirects:
            raise ConnectionError("Redirect loop detected")
        
        if response.status_code == 404:
            raise ConnectionError("Page not found (404)")
        elif response.status_code == 403:
            raise ConnectionError("Access forbidden (403)")
        elif response.status_code >= 400:
            raise ConnectionError(f"HTTP {response.status_code} error")
        
        return response
        
    except requests.exceptions.Timeout:
        raise ConnectionError("Request timeout")
    except requests.exceptions.ConnectionError as e:
        if "getaddrinfo failed" in str(e) or "failed to resolve" in str(e):
            raise ConnectionError("Domain not found")
        raise ConnectionError(f"Connection failed: {str(e)}")
    except requests.exceptions.TooManyRedirects:
        raise ConnectionError("Too many redirects")
    except Exception as e:
        raise ConnectionError(f"Connection error: {str(e)}")

def extract_features(domain):
    features = fex.data_set_list_creation(domain)
    if features is None or not isinstance(features, list):
        raise ValueError("Feature extraction failed or returned invalid data")
    return features

def predict_phishing(features):
    if not features:
        raise ValueError("Features are empty or None")
    features = np.array([features])
    prediction = model.predict(features)
    if prediction is None:
        raise ValueError("Prediction failed or returned None")
    
    return prediction[0] == 1

def process_url_input(domain_input):
    try:
        cleaned_domain = preprocess_url(domain_input)
        print(f"Cleaned domain: {cleaned_domain}")
        
        try:
            fetch_url_content(cleaned_domain)
            print(f"URL fetched successfully: {cleaned_domain}")
            features = extract_features(cleaned_domain)
            print(f"Extracted features: {features}")
            print(len(features))
            
            is_phishing = predict_phishing(features)
            result_text = "The URL is predicted as a phishing domain🔴." if is_phishing else "The URL is predicted as a legitimate domain🟢."
            
            additional_info = {
                "domain": str(ress._url_domain(cleaned_domain)),
                "ip": str(ress.has_ip(ress._url_domain(cleaned_domain))), 
                "Domain Age": str(ress.domain_age(ress._url_domain(cleaned_domain))),
                "num_sub_domains": str(ress.number_of_subdomains(preprocess_url(cleaned_domain))), 
                "domain_reg_length": str(ress.domain_registration_length(ress._url_domain(cleaned_domain))),  
                "ip_counts": str(ress.get_ip_count(ress._url_domain(cleaned_domain))), 
                "ssl_update_age(In Days)": str(ress.get_ssl_update_age(ress._url_domain(cleaned_domain))),  
                "num_smtp_servers": str(ress.number_of_smtp_servers(ress._url_domain(cleaned_domain).removeprefix("www.")))  
            }
            
            return {
                "result_text": result_text,
                "additional_info": additional_info
            }
            
        except ConnectionError as e:
            return {
                "result_text": f"⚠️ Connection Failed: {str(e)} - Limited analysis performed",
                "additional_info": {"warning": "Domain-only analysis due to connection failure"}
            }

    except Exception as e:
        return {
            "result_text": f"Error: {e}",
            "additional_info": {}
        }

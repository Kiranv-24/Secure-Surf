
import requests
import Feature_extraction_ff1 as fex  
import numpy as np
import os
import joblib
import result as ress
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket

model_path = r"C:\Users\Darshan.v\OneDrive\Desktop\HTF-Final-Judge 2\HTF-Final-Judge\best_rf_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

def create_session_with_retry():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1
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
    """Enhanced URL fetching with better error handling"""
    session = create_session_with_retry()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # First try the original URL
        response = session.get(url, timeout=15, headers=headers, verify=False, allow_redirects=True)
        response.raise_for_status()
        return response
    except requests.exceptions.SSLError:
        # If SSL fails, try without SSL verification
        try:
            response = session.get(url, timeout=15, headers=headers, verify=False, allow_redirects=True)
            response.raise_for_status()
            return response
        except Exception as e:
            raise ConnectionError(f"SSL and non-SSL connection failed: {str(e)}")
    except requests.exceptions.Timeout:
        raise ConnectionError("Request timeout - the website is taking too long to respond")
    except requests.exceptions.ConnectionError as e:
        if "Name or service not known" in str(e) or "getaddrinfo failed" in str(e):
            raise ConnectionError("Domain not found - this website may not exist or may be offline")
        else:
            raise ConnectionError(f"Connection failed: {str(e)}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            raise ConnectionError("Access forbidden - the website is blocking automated requests")
        elif e.response.status_code == 404:
            raise ConnectionError("Page not found - this URL may not exist")
        elif e.response.status_code == 429:
            raise ConnectionError("Too many requests - the website is rate limiting")
        else:
            raise ConnectionError(f"HTTP error {e.response.status_code}: {str(e)}")
    except Exception as e:
        raise ConnectionError(f"Unexpected error: {str(e)}")

def extract_features(domain):
    try:
        features = fex.data_set_list_creation(domain)
        if features is None or not isinstance(features, list):
            raise ValueError("Feature extraction failed or returned invalid data")
        return features
    except Exception as e:
        raise ValueError(f"Feature extraction error: {str(e)}")

def predict_phishing(features):
    if not features:
        raise ValueError("Features are empty or None")
    try:
        features = np.array([features])
        prediction = model.predict(features)
        if prediction is None:
            raise ValueError("Prediction failed or returned None")
        
        return prediction[0] == 1
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")

def analyze_url_without_connection(domain_input):
    """Analyze URL using only domain-based features when connection fails"""
    try:
        cleaned_domain = preprocess_url(domain_input)
        domain = ress._url_domain(cleaned_domain)
        
        # Extract basic domain features that don't require website access
        basic_features = {
            "domain": str(domain),
            "Domain Age": str(ress.domain_age(domain)),
            "num_sub_domains": str(ress.number_of_subdomains(cleaned_domain)), 
            "domain_reg_length": str(ress.domain_registration_length(domain)),  
            "ip_counts": str(ress.get_ip_count(domain)), 
            "ssl_update_age(In Days)": str(ress.get_ssl_update_age(domain)),  
            "num_smtp_servers": str(ress.number_of_smtp_servers(domain.removeprefix("www."))),
            "has_ip": str(ress.has_ip(domain))
        }
        
        # Simple heuristic analysis based on domain characteristics
        suspicious_indicators = 0
        
        # Check for suspicious domain patterns
        if len(domain.split('.')) > 3:  # Multiple subdomains
            suspicious_indicators += 1
        if any(char.isdigit() for char in domain):  # Numbers in domain
            suspicious_indicators += 1
        if len(domain) > 30:  # Very long domain
            suspicious_indicators += 1
        if any(suspicious in domain.lower() for suspicious in ['login', 'secure', 'bank', 'verify', 'update']):
            suspicious_indicators += 1
        
        is_likely_phishing = suspicious_indicators >= 2
        
        return {
            "result_text": f"⚠️ Limited Analysis (Connection Failed): Based on domain characteristics, this URL appears {'SUSPICIOUS - Potential phishing domain' if is_likely_phishing else 'LEGITIMATE - Domain characteristics look normal'}",
            "additional_info": {
                **basic_features,
                "analysis_type": "Limited - Connection Failed",
                "suspicious_indicators": str(suspicious_indicators),
                "warning": "Full analysis requires website access"
            }
        }
    except Exception as e:
        return {
            "result_text": f"Analysis Failed: Unable to analyze domain due to: {str(e)}",
            "additional_info": {"error": "Complete analysis failure"}
        }

def process_url_input(domain_input):
    try:
        cleaned_domain = preprocess_url(domain_input)
        print(f"Cleaned domain: {cleaned_domain}")
        
        try:
            # Try to fetch the URL
            fetch_url_content(cleaned_domain)
            print(f"URL fetched successfully: {cleaned_domain}")
            
            # Extract features and predict
            features = extract_features(cleaned_domain)
            print(f"Extracted features: {features}")
            
            is_phishing = predict_phishing(features)
            result_text = "🔴 THREAT DETECTED: This URL is predicted as a phishing domain." if is_phishing else "🟢 VERIFIED SAFE: This URL is predicted as a legitimate domain."
            
            try:
                additional_info = {
                    "domain": str(ress._url_domain(cleaned_domain)),
                    "ip": str(ress.has_ip(ress._url_domain(cleaned_domain))), 
                    "Domain Age": str(ress.domain_age(ress._url_domain(cleaned_domain))),
                    "num_sub_domains": str(ress.number_of_subdomains(preprocess_url(cleaned_domain))), 
                    "domain_reg_length": str(ress.domain_registration_length(ress._url_domain(cleaned_domain))),  
                    "ip_counts": str(ress.get_ip_count(ress._url_domain(cleaned_domain))), 
                    "ssl_update_age(In Days)": str(ress.get_ssl_update_age(ress._url_domain(cleaned_domain))),  
                    "num_smtp_servers": str(ress.number_of_smtp_servers(ress._url_domain(cleaned_domain).removeprefix("www."))),
                    "analysis_type": "Full Analysis"
                }
            except Exception as e:
                print(f"Error extracting additional info: {str(e)}")
                additional_info = {"info": "Additional analysis failed", "analysis_type": "Basic Analysis"}
            
            return {
                "result_text": result_text,
                "additional_info": additional_info
            }
            
        except ConnectionError as conn_error:
            print(f"Connection failed, attempting limited analysis: {str(conn_error)}")
            # Fall back to domain-only analysis
            return analyze_url_without_connection(domain_input)
            
    except ValueError as e:
        return {
            "result_text": f"❌ ANALYSIS ERROR: {str(e)}",
            "additional_info": {"error_type": "Validation Error"}
        }
    except Exception as e:
        return {
            "result_text": f"❌ SYSTEM ERROR: {str(e)}",
            "additional_info": {"error_type": "System Error"}
        }

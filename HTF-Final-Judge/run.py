
import requests
import Feature_extraction_ff1 as fex  
import numpy as np
import os
import joblib
import result as ress
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket
import time

model_path = r"C:\Users\Darshan.v\OneDrive\Desktop\HTF-Final-Judge 2\HTF-Final-Judge\best_rf_model.joblib"

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
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # Try with allow_redirects=False first to handle redirect loops
        response = session.get(url, timeout=10, headers=headers, verify=False, allow_redirects=False)
        
        # Handle redirects manually to avoid infinite loops
        redirect_count = 0
        max_redirects = 5
        
        while response.status_code in [301, 302, 303, 307, 308] and redirect_count < max_redirects:
            redirect_url = response.headers.get('Location')
            if not redirect_url:
                break
            
            # Handle relative redirects
            if redirect_url.startswith('/'):
                from urllib.parse import urljoin
                redirect_url = urljoin(url, redirect_url)
            
            response = session.get(redirect_url, timeout=10, headers=headers, verify=False, allow_redirects=False)
            redirect_count += 1
            url = redirect_url
        
        if redirect_count >= max_redirects:
            raise ConnectionError("Too many redirects - possible redirect loop detected")
        
        # Check final response status
        if response.status_code == 404:
            raise ConnectionError("Page not found (404) - This URL may not exist or has been removed")
        elif response.status_code == 403:
            raise ConnectionError("Access forbidden (403) - The website is blocking automated requests")
        elif response.status_code == 429:
            raise ConnectionError("Rate limited (429) - Too many requests to this website")
        elif response.status_code >= 400:
            raise ConnectionError(f"HTTP {response.status_code} error - Server returned an error response")
        
        return response
        
    except requests.exceptions.SSLError as e:
        raise ConnectionError(f"SSL certificate error - The website's security certificate is invalid: {str(e)}")
    except requests.exceptions.Timeout:
        raise ConnectionError("Connection timeout - The website is taking too long to respond (>10 seconds)")
    except requests.exceptions.ConnectionError as e:
        error_str = str(e).lower()
        if "name or service not known" in error_str or "getaddrinfo failed" in error_str or "failed to resolve" in error_str:
            raise ConnectionError("Domain not found - This website may not exist, be offline, or the domain may be invalid")
        elif "connection refused" in error_str:
            raise ConnectionError("Connection refused - The website server is not accepting connections")
        elif "max retries exceeded" in error_str:
            raise ConnectionError("Connection failed after multiple attempts - The website may be temporarily unavailable")
        else:
            raise ConnectionError(f"Network connection failed: {str(e)}")
    except requests.exceptions.TooManyRedirects:
        raise ConnectionError("Too many redirects - The website has a redirect loop")
    except Exception as e:
        raise ConnectionError(f"Unexpected connection error: {str(e)}")

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
    """Enhanced domain-only analysis when connection fails"""
    try:
        cleaned_domain = preprocess_url(domain_input)
        domain = ress._url_domain(cleaned_domain)
        
        # Extract basic domain features that don't require website access
        try:
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
        except Exception as e:
            basic_features = {
                "domain": str(domain),
                "analysis_note": "Limited feature extraction due to domain resolution issues"
            }
        
        # Enhanced heuristic analysis
        suspicious_indicators = 0
        warning_flags = []
        
        # Check for suspicious domain patterns
        if len(domain.split('.')) > 3:
            suspicious_indicators += 1
            warning_flags.append("Multiple subdomains detected")
        
        if any(char.isdigit() for char in domain):
            suspicious_indicators += 1
            warning_flags.append("Numbers in domain name")
        
        if len(domain) > 30:
            suspicious_indicators += 1
            warning_flags.append("Unusually long domain name")
        
        # Check for suspicious keywords
        phishing_keywords = ['login', 'secure', 'bank', 'verify', 'update', 'account', 'signin', 'auth', 'validation']
        if any(keyword in domain.lower() for keyword in phishing_keywords):
            suspicious_indicators += 1
            warning_flags.append("Contains suspicious keywords")
        
        # Check for URL shorteners or suspicious TLDs
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf']
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            suspicious_indicators += 1
            warning_flags.append("Uses suspicious top-level domain")
        
        risk_level = "HIGH" if suspicious_indicators >= 3 else "MEDIUM" if suspicious_indicators >= 2 else "LOW"
        is_likely_phishing = suspicious_indicators >= 2
        
        return {
            "result_text": f"⚠️ LIMITED ANALYSIS (Connection Failed): Risk Level: {risk_level} - {'SUSPICIOUS - Potential phishing indicators detected' if is_likely_phishing else 'APPEARS NORMAL - Domain characteristics look legitimate'}",
            "additional_info": {
                **basic_features,
                "analysis_type": "Domain-Only Analysis",
                "risk_level": risk_level,
                "suspicious_indicators": str(suspicious_indicators),
                "warning_flags": ", ".join(warning_flags) if warning_flags else "None detected",
                "limitation": "Full analysis requires successful website connection"
            }
        }
    except Exception as e:
        return {
            "result_text": f"❌ ANALYSIS FAILED: Unable to analyze domain - {str(e)}",
            "additional_info": {"error": "Domain analysis completely failed", "error_details": str(e)}
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
                    "analysis_type": "Full ML Analysis",
                    "confidence": "High (Website accessible)"
                }
            except Exception as e:
                print(f"Error extracting additional info: {str(e)}")
                additional_info = {
                    "analysis_type": "Basic ML Analysis", 
                    "note": "Some additional features unavailable",
                    "confidence": "Medium"
                }
            
            return {
                "result_text": result_text,
                "additional_info": additional_info
            }
            
        except ConnectionError as conn_error:
            print(f"Connection failed, performing domain-only analysis: {str(conn_error)}")
            # Enhanced fallback analysis
            fallback_result = analyze_url_without_connection(domain_input)
            # Add connection error details
            fallback_result["additional_info"]["connection_error"] = str(conn_error)
            return fallback_result
            
    except ValueError as e:
        return {
            "result_text": f"❌ INPUT ERROR: {str(e)}",
            "additional_info": {"error_type": "Invalid URL Format", "suggestion": "Please check the URL format and try again"}
        }
    except Exception as e:
        return {
            "result_text": f"❌ SYSTEM ERROR: {str(e)}",
            "additional_info": {"error_type": "System Error", "suggestion": "Please try again or contact support"}
        }

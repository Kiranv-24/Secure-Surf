
import requests
import Feature_extraction_ff1 as fex  
import numpy as np
import os
import joblib
import result as ress
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

model_path = r"C:\Users\Darshan.v\OneDrive\Desktop\HTF-Final-Judge 2\HTF-Final-Judge\best_rf_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

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
    """Simple URL fetching with basic error handling"""
    try:
        response = requests.get(url, timeout=10, verify=False, allow_redirects=True)
        if response.status_code >= 400:
            raise ConnectionError("Site is not active")
        return response
    except Exception:
        raise ConnectionError("Site is not active")

def extract_features(domain):
    try:
        features = fex.data_set_list_creation(domain)
        if features is None or not isinstance(features, list):
            raise ValueError("Feature extraction failed")
        return features
    except Exception as e:
        raise ValueError(f"Feature extraction error: {str(e)}")

def predict_phishing(features):
    if not features:
        raise ValueError("Features are empty")
    try:
        features = np.array([features])
        prediction = model.predict(features)
        return prediction[0] == 1
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")

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
            
        except ConnectionError:
            return {
                "result_text": "⚠️ Site is not active - Unable to perform analysis",
                "additional_info": {"status": "Site not accessible"}
            }
            
    except ValueError as e:
        return {
            "result_text": f"❌ INPUT ERROR: {str(e)}",
            "additional_info": {"error_type": "Invalid URL Format"}
        }
    except Exception as e:
        return {
            "result_text": f"❌ SYSTEM ERROR: {str(e)}",
            "additional_info": {"error_type": "System Error"}
        }

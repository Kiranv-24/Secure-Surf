
import requests
import Feature_extraction_ff1 as fex  
import numpy as np
import os
import joblib
import result as ress
from datetime import datetime
import traceback

# Load the trained model
model_path = "best_rf_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

def preprocess_url(domain):
    """Preprocess URL to ensure proper format"""
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
    """Fetch URL content with timeout"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        raise ConnectionError(f"Cannot connect to URL: {str(e)}")

def extract_features(domain):
    """Extract features from domain"""
    try:
        features = fex.data_set_list_creation(domain)
        if features is None or not isinstance(features, list):
            raise ValueError("Feature extraction failed or returned invalid data")
        return features
    except Exception as e:
        raise ValueError(f"Feature extraction error: {str(e)}")

def predict_phishing(features):
    """Predict if URL is phishing"""
    if not features:
        raise ValueError("Features are empty or None")
    try:
        features = np.array([features])
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        if prediction is None:
            raise ValueError("Prediction failed or returned None")
        
        is_phishing = prediction[0] == 1
        confidence = max(probability[0]) * 100
        
        return is_phishing, confidence
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")

def process_single_url(url, actual_label=None):
    """Process a single URL and return results"""
    try:
        cleaned_url = preprocess_url(url.strip())
        print(f"Processing: {cleaned_url}")
        
        # Fetch URL content
        fetch_url_content(cleaned_url)
        
        # Extract features
        features = extract_features(cleaned_url)
        
        # Make prediction
        is_phishing, confidence = predict_phishing(features)
        
        result = {
            'url': cleaned_url,
            'predicted': 'yes' if is_phishing else 'no',
            'confidence': confidence,
            'status': 'success',
            'error': None
        }
        
        if actual_label is not None:
            result['actual'] = actual_label
            result['correct'] = (result['predicted'] == actual_label)
        
        return result
        
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return {
            'url': url.strip(),
            'predicted': 'error',
            'confidence': 0,
            'status': 'error',
            'error': str(e),
            'actual': actual_label if actual_label else None,
            'correct': False if actual_label else None
        }

def parse_input_file(file_path):
    """Parse input file and extract URLs with labels"""
    urls_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                
                # Check if line contains label (format: url - yes/no)
                if ' - ' in line:
                    parts = line.split(' - ', 1)
                    url = parts[0].strip()
                    label = parts[1].strip().lower()
                    
                    if label not in ['yes', 'no']:
                        print(f"Warning: Invalid label '{label}' on line {line_num}. Expected 'yes' or 'no'")
                        continue
                    
                    urls_data.append({'url': url, 'actual': label})
                else:
                    # Just URL without label
                    urls_data.append({'url': line, 'actual': None})
        
        return urls_data
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading input file: {str(e)}")

def calculate_accuracy(results):
    """Calculate accuracy metrics"""
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    errors = total - successful
    
    # Calculate accuracy only for results with actual labels
    labeled_results = [r for r in results if r.get('actual') is not None and r['status'] == 'success']
    
    if not labeled_results:
        return {
            'total_urls': total,
            'successful_predictions': successful,
            'errors': errors,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None
        }
    
    correct = sum(1 for r in labeled_results if r['correct'])
    accuracy = (correct / len(labeled_results)) * 100 if labeled_results else 0
    
    # Calculate precision, recall, F1-score
    tp = sum(1 for r in labeled_results if r['predicted'] == 'yes' and r['actual'] == 'yes')
    fp = sum(1 for r in labeled_results if r['predicted'] == 'yes' and r['actual'] == 'no')
    fn = sum(1 for r in labeled_results if r['predicted'] == 'no' and r['actual'] == 'yes')
    tn = sum(1 for r in labeled_results if r['predicted'] == 'no' and r['actual'] == 'no')
    
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return {
        'total_urls': total,
        'successful_predictions': successful,
        'errors': errors,
        'labeled_urls': len(labeled_results),
        'correct_predictions': correct,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def batch_test_urls(input_file_path, output_file_path=None):
    """Main function to test multiple URLs"""
    
    # Generate output filename if not provided
    if output_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = f"batch_test_results_{timestamp}.txt"
    
    print(f"Starting batch testing...")
    print(f"Input file: {input_file_path}")
    print(f"Output file: {output_file_path}")
    print("-" * 50)
    
    try:
        # Parse input file
        urls_data = parse_input_file(input_file_path)
        print(f"Found {len(urls_data)} URLs to process")
        
        # Process each URL
        results = []
        for i, url_data in enumerate(urls_data, 1):
            print(f"Processing {i}/{len(urls_data)}: {url_data['url']}")
            result = process_single_url(url_data['url'], url_data['actual'])
            results.append(result)
        
        # Calculate accuracy metrics
        accuracy_metrics = calculate_accuracy(results)
        
        # Write results to output file
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            # Write header
            output_file.write("PHISHING DETECTION BATCH TEST RESULTS\n")
            output_file.write("=" * 50 + "\n")
            output_file.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            output_file.write(f"Input File: {input_file_path}\n")
            output_file.write(f"Model Used: {model_path}\n\n")
            
            # Write accuracy summary
            output_file.write("ACCURACY SUMMARY\n")
            output_file.write("-" * 20 + "\n")
            output_file.write(f"Total URLs: {accuracy_metrics['total_urls']}\n")
            output_file.write(f"Successful Predictions: {accuracy_metrics['successful_predictions']}\n")
            output_file.write(f"Errors: {accuracy_metrics['errors']}\n")
            
            if accuracy_metrics['accuracy'] is not None:
                output_file.write(f"Labeled URLs: {accuracy_metrics['labeled_urls']}\n")
                output_file.write(f"Correct Predictions: {accuracy_metrics['correct_predictions']}\n")
                output_file.write(f"Accuracy: {accuracy_metrics['accuracy']:.2f}%\n")
                output_file.write(f"Precision: {accuracy_metrics['precision']:.2f}%\n")
                output_file.write(f"Recall: {accuracy_metrics['recall']:.2f}%\n")
                output_file.write(f"F1-Score: {accuracy_metrics['f1_score']:.2f}%\n\n")
                
                output_file.write("CONFUSION MATRIX\n")
                output_file.write("-" * 16 + "\n")
                output_file.write(f"True Positives: {accuracy_metrics['true_positives']}\n")
                output_file.write(f"False Positives: {accuracy_metrics['false_positives']}\n")
                output_file.write(f"True Negatives: {accuracy_metrics['true_negatives']}\n")
                output_file.write(f"False Negatives: {accuracy_metrics['false_negatives']}\n\n")
            else:
                output_file.write("No labeled data found - accuracy metrics not available\n\n")
            
            # Write detailed results
            output_file.write("DETAILED RESULTS\n")
            output_file.write("-" * 16 + "\n")
            output_file.write(f"{'URL':<50} {'Predicted':<10} {'Actual':<8} {'Confidence':<12} {'Status':<10}\n")
            output_file.write("-" * 100 + "\n")
            
            for result in results:
                url = result['url'][:47] + "..." if len(result['url']) > 50 else result['url']
                predicted = result['predicted']
                actual = result.get('actual', 'N/A')
                confidence = f"{result['confidence']:.1f}%" if result['confidence'] > 0 else "N/A"
                status = result['status']
                
                output_file.write(f"{url:<50} {predicted:<10} {actual:<8} {confidence:<12} {status:<10}\n")
                
                if result['error']:
                    output_file.write(f"    Error: {result['error']}\n")
            
            output_file.write("\n" + "=" * 50 + "\n")
            output_file.write("END OF REPORT\n")
        
        # Print summary to console
        print("\n" + "=" * 50)
        print("BATCH TEST COMPLETED")
        print("=" * 50)
        print(f"Results saved to: {output_file_path}")
        print(f"Total URLs processed: {accuracy_metrics['total_urls']}")
        print(f"Successful predictions: {accuracy_metrics['successful_predictions']}")
        print(f"Errors: {accuracy_metrics['errors']}")
        
        if accuracy_metrics['accuracy'] is not None:
            print(f"Overall Accuracy: {accuracy_metrics['accuracy']:.2f}%")
        else:
            print("No labeled data - accuracy not calculated")
        
        return results, accuracy_metrics
        
    except Exception as e:
        print(f"Error during batch testing: {str(e)}")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Example usage
    input_file = "test_urls.txt"  # Change this to your input file path
    output_file = "prediction_results.txt"  # Change this to your desired output file path
    
    print("Phishing Detection Batch Tester")
    print("=" * 30)
    print("\nUsage:")
    print("1. Create a text file with URLs (one per line)")
    print("2. For labeled data, use format: url - yes/no")
    print("   Example: https://malicious-site.com - yes")
    print("   Example: https://google.com - no")
    print("3. Run this script\n")
    
    # Check if example input file exists
    if not os.path.exists(input_file):
        print(f"Creating example input file: {input_file}")
        with open(input_file, 'w') as f:
            f.write("# Example input file for batch testing\n")
            f.write("# Format: url - yes/no (yes=phishing, no=legitimate)\n")
            f.write("# Or just: url (for unlabeled testing)\n\n")
            f.write("https://google.com - no\n")
            f.write("https://facebook.com - no\n")
            f.write("https://suspicious-phishing-site.com - yes\n")
            f.write("https://github.com\n")  # Unlabeled example
        print(f"Example file created. Edit {input_file} with your URLs and run again.")
    else:
        # Run batch testing
        results, metrics = batch_test_urls(input_file, output_file)

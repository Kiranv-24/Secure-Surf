
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import run  # Import the necessary functions from run.py

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/dev')
def dev():
    return render_template('dev.html')

@app.route('/check_phishing', methods=['POST'])
def check_phishing():
    try:
        if not request.is_json:
            print("Request Content-Type is not application/json")
            print("Raw data received:", request.data)
            return jsonify({"error": "Content-Type must be application/json", "result_text": "Invalid request format"}), 415
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received", "result_text": "No data provided"}), 400
            
        domain = data.get('url')
        if not domain:
            return jsonify({"error": "URL parameter missing", "result_text": "Please provide a URL"}), 400

        print(f"Processing URL: {domain}")
        result = run.process_url_input(domain)
        print(f"Result: {result}")
        
        # Ensure the result has the required structure
        if not isinstance(result, dict):
            result = {"result_text": str(result), "additional_info": {}}
        
        # Add a message field for the home.html template
        result["message"] = result.get("result_text", "Analysis complete")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in check_phishing: {str(e)}")
        error_response = {
            "error": str(e),
            "result_text": f"Error processing request: {str(e)}",
            "message": f"Error: {str(e)}",
            "additional_info": {}
        }
        return jsonify(error_response), 500

if __name__ == '__main__':
    app.run(debug=True, port=2002, host='127.0.0.1')

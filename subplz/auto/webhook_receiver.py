# webhook_receiver.py

import json
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# (Optional but Recommended) Add a root/homepage route
@app.route('/', methods=['GET'])
def index():
    """
    A simple homepage to show the server is alive.
    """
    # Returns a simple HTML message
    return "<h1>Webhook Server is Running</h1><p>Send POST requests from Bazarr to /webhook.</p>"


@app.route('/webhook', methods=['GET', 'POST'])  # <-- ADDED 'GET' METHOD
def handle_webhook():
    """
    This endpoint now handles both GET and POST requests.
    - GET: Returns a simple status message.
    - POST: Receives data from Bazarr, prints it, and returns success.
    """
    # Check if the request is a POST (from Bazarr)
    if request.method == 'POST':
        if not request.json:
            print("❌ Received non-JSON request")
            return jsonify({"status": "error", "message": "Invalid request, expected JSON"}), 400

        payload = request.json
        print("="*50)
        print(f"✅ POST received from Bazarr at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Pretty-print the JSON to the console log.
        print(json.dumps(payload, indent=2))
        print("="*50)

        # Return a success message to Bazarr
        return jsonify({"status": "success", "message": "Payload received and logged"}), 200

    # If the request is a GET (from a web browser)
    if request.method == 'GET':
        # Return a simple JSON status message
        return jsonify({
            "status": "ok",
            "message": "Webhook server is running. Ready to receive POST requests from Bazarr.",
            "timestamp": datetime.now().isoformat()
        }), 200

if __name__ == '__main__':
    print("🚀 Starting local webhook receiver...")
    print(f"🌍 Homepage available at http://127.0.0.1:5775/")
    print(f"👂 Listening for GET/POST at http://127.0.0.1:5775/webhook")
    print("📝 Press Ctrl+C to stop the server.")
    app.run(host='0.0.0.0', port=5775)

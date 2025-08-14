# revised webhook_receiver.py
import os
import subprocess
from flask import Flask, request, jsonify

MEDIA_ROOT_IN_CONTAINER = "/media"
app = Flask(__name__)

def find_path_in_payload(payload):
    """
    Searches the payload for a valid subtitle file path.
    This makes it resilient to changes in Bazarr's payload structure.
    """
    # First, check common top-level keys
    common_keys = ['filepath', 'subtitles_path', 'destination_path', 'path']
    for key in common_keys:
        path = payload.get(key)
        if isinstance(path, str) and path.startswith(MEDIA_ROOT_IN_CONTAINER):
            return path

    # If not found, do a deeper search through the payload values
    for value in payload.values():
        if isinstance(value, str) and value.startswith(MEDIA_ROOT_IN_CONTAINER) and value.endswith(('.srt', '.ass', '.vtt')):
             return value
    return None

def process_subtitles(subtitle_path):
    # ... (This function remains the same as the previous answer)
    target_directory = os.path.dirname(subtitle_path)
    script_path = "/app/helpers/subplz.sh"
    print(f"🚀 Triggering SubPlz batch process for directory: {target_directory}", flush=True)
    try:
        subprocess.run([script_path, target_directory], capture_output=True, text=True, check=True)
        print(f"✅ Successfully processed subtitles for: {target_directory}", flush=True)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running SubPlz script for {target_directory}", flush=True)
        print("Stderr:", e.stderr, flush=True)
        return False, e.stderr
    except FileNotFoundError:
        print(f"❌ Error: Script not found at {script_path}", flush=True)
        return False, "Script not found"


@app.route('/webhook', methods=['POST'])
def handle_webhook():
    if not request.json:
        return jsonify({"status": "error", "message": "Invalid request, expected JSON"}), 400

    payload = request.json
    print(f"ℹ️ Received webhook payload: {payload}", flush=True)

    subtitle_file_path = find_path_in_payload(payload)

    if subtitle_file_path:
        print(f"✅ Found subtitle path in payload: {subtitle_file_path}", flush=True)
        success, message = process_subtitles(subtitle_file_path)
        if success:
            return jsonify({"status": "success", "message": "SubPlz triggered"}), 200
        else:
            return jsonify({"status": "error", "message": message}), 500
    else:
        print("❌ Could not find a valid subtitle file path in the payload.", flush=True)
        return jsonify({"status": "error", "message": "Subtitle path not found in payload"}), 400

if __name__ == '__main__':
    print("🚀 Starting SubPlz Webhook Receiver on port 5000...", flush=True)
    app.run(host='0.0.0.0', port=5000)
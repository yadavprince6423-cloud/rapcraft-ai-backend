import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # Is line se CORS issue fixed!

HF_TOKEN = os.environ.get("HF_API_KEY")

@app.route("/")
def home():
    return jsonify({"status": "RapCraft AI Backend is Live!"})

@app.route("/generate-rap", methods=["POST"])
def generate_rap():
    data = request.json or {}
    topic = data.get("topic", "hip hop life")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": f"Write a catchy 8-line rap verse about: {topic}",
        "parameters": {"max_new_tokens": 200}
    }

    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return jsonify({"success": True, "rap": response.json()})
        else:
            return jsonify({"success": False, "error": response.text}), response.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    

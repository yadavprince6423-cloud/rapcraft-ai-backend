from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy.io.wavfile
import os
import uuid

app = Flask(__name__)
CORS(app)

print("Loading Meta Hip-Hop MusicGen AI Engine...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
print("Hip-Hop AI Engine Loaded Successfully!")

@app.route('/generate-rap', methods=['POST'])
def generate_rap():
    try:
        data = request.json
        lyrics = data.get('lyrics', '')
        
        if not lyrics:
            return jsonify({'error': 'Lyrics are required'}), 400

        session_id = str(uuid.uuid4())[:8]
        output_file = f"rap_{session_id}.wav"

        # Dedicated Indian Underground Hip-Hop & Rap Beat Prompt Engine
        hiphop_prompt = f"heavy 808 bass underground desi hip hop rap beat, fast hi-hats, hard drum kit, rhythmic rap flow style, tempo 140bpm, aggressive melodic chorus, lyrics: {lyrics}"

        inputs = processor(
            text=[hiphop_prompt],
            padding=True,
            return_tensors="pt"
        )

        # Generate Hip-Hop Audio Track (~15 Seconds)
        audio_values = model.generate(**inputs, max_new_tokens=768)

        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()
        scipy.io.wavfile.write(output_file, rate=sampling_rate, data=audio_data)

        return send_file(output_file, mimetype="audio/wav", as_attachment=True)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
  

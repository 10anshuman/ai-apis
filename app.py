from flask import Flask, request, jsonify
from extraction import get_ai_msg
from transcription import process_b64_str, decode_base64_to_audio_file, transcribe_audio

app = Flask(__name__)

@app.route('/transcription', methods=['POST'])
def transcribe_endpoint():
    try:
        data = request.get_json()
        if not data or 'b64_str' not in data:
            return jsonify({'error': 'Missing base64 string (b64_str) in the request'}), 400
        b64_str = data['b64_str']
        transcription = process_b64_str(b64_str)

        return jsonify({'transcription': transcription})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/extraction", methods=["POST"])
def process_prompt():
    try:
        patient_prompt = request.json.get("patient_prompt")
        if not patient_prompt:
            return jsonify({"error": "Patient prompt is required."}), 400

        ai_response = get_ai_msg(patient_prompt)
        return jsonify(ai_response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/conversation', methods=['POST'])
def process_audio():
    try:
        data = request.json
        b64_str = data.get('b64_str', None)

        if not b64_str:
            return jsonify({"error": "Base64 string is required."}), 400

        # Decode and transcribe
        audio_stream = decode_base64_to_audio_file(b64_str)
        transcription = transcribe_audio(audio_stream)

        # Extract details
        structured_data = get_ai_msg(transcription)

        return jsonify(structured_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000,debug=True)
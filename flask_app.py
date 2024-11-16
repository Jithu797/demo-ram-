from flask import Flask, request, jsonify, send_file
import openai
import requests
import config
import boto3
import speech_recognition as sr
from flask_cors import CORS
import os
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI API Key from Config
openai.api_key = config.OPENAI_API_KEY

# Initialize AWS Polly Client for Text-to-Speech using environment variables
polly_client = boto3.Session(
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name="us-east-1"  # Choose your preferred region
).client('polly')

# Speech Recognition Setup
recognizer = sr.Recognizer()

# Default route for root URL
@app.route('/')
def home():
    return "Welcome to the Juliee Voice Assistant API!", 200

# Handle favicon.ico request
@app.route('/favicon.ico')
def favicon():
    return "", 204

# OpenAI GPT-3 Endpoint
@app.route('/ask', methods=['POST'])
def ask_juliee():
    try:
        data = request.json
        user_query = data.get('query', '')
        ai_model = data.get('model', 'ChatGPT')

        if not user_query:
            return jsonify({"error": "Query is required."}), 400

        if ai_model == 'ChatGPT':
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_query}],
                max_tokens=150  # Adjust max_tokens as needed
            )
            result = response.choices[0].message['content'].strip()  # Extract response message
        elif ai_model == 'Gemini':
            result = get_gemini_response(user_query)
        else:
            return jsonify({"error": "AI model not supported."}), 400

        return jsonify({"response": result})
    except openai.OpenAIError as e:  # Updated error handling for new API
        logger.error(f"OpenAI API Error: {e}")
        return jsonify({"error": f"OpenAI API Error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        return jsonify({"error": str(e)}), 500

# Define the get_gemini_response function
def get_gemini_response(query):
    headers = {
        'Authorization': f'Bearer {config.GEMINI_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "query": query
    }
    try:
        response = requests.post("https://api.gemini.com/v1/query", json=data, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Gemini API Error: {response.status_code}")
            return {"error": f"Failed to get response from Gemini API: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error: {e}")
        return {"error": f"Request Error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        return {"error": str(e)}

# Endpoint for speech-to-text using Google Speech Recognition
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "Audio file is required."}), 400
        
        # Process the audio file
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        logger.info("Speech to text conversion successful")
        return jsonify({"text": text})
    except sr.UnknownValueError:
        logger.error("Google Speech Recognition could not understand audio.")
        return jsonify({"error": "Google Speech Recognition could not understand audio."}), 400
    except sr.RequestError as e:
        logger.error(f"Google Speech Recognition service error: {e}")
        return jsonify({"error": f"Could not request results from Google Speech Recognition service; {e}"}), 500
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        return jsonify({"error": str(e)}), 500

# Endpoint for text-to-speech using Amazon Polly
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    text = request.json.get('text', '')
    if not text:
        return jsonify({"error": "Text is required for conversion."}), 400
    try:
        # Synthesize speech using Amazon Polly
        response = polly_client.synthesize_speech(
            VoiceId='Joanna',  # Choose your preferred voice
            OutputFormat='mp3',  # You can use other formats like 'ogg_vorbis'
            Text=text
        )

        # Save the audio file temporarily
        audio_file_path = 'speech.mp3'
        with open(audio_file_path, 'wb') as file:
            file.write(response['AudioStream'].read())

        logger.info("Text to speech conversion successful")
        
        # Close the file handle before returning it
        file.close()
        
        # Return audio file as response
        return send_file(audio_file_path, as_attachment=True)

    except boto3.exceptions.Boto3Error as e:
        logger.error(f"Amazon Polly Error: {e}")
        return jsonify({"error": f"Amazon Polly Error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary file after sending response
        if os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
            except PermissionError as pe:
                logger.error(f"Permission Error: {pe}")
                return jsonify({"error": "Failed to delete the temporary file."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

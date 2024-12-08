# requirements.txt
"""
Flask==2.3.3
Flask-CORS==4.0.0
python-multipart==0.0.6
gunicorn==21.2.0
google-cloud-speech==2.21.0
google-cloud-texttospeech==2.14.1
google-generativeai==0.3.0
python-dotenv==1.0.0
"""

# main.py = version-finale-claide-flask
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os
from google.cloud import speech
from google.cloud import texttospeech
import google.generativeai as genai
import io
import wave
from typing import List, Dict
import json
import tempfile


# Configuration de Flask
app = Flask(__name__)
CORS(app)  # Activation de CORS pour toutes les routes

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Accéder aux variables d'environnement
GOOGLE_CREDENTIALS_PATH = {
    "type": os.getenv('GOOGLE_CREDENTIALS_TYPE'),
    "project_id": os.getenv('GOOGLE_CREDENTIALS_PROJECT_ID'),
    "private_key_id": os.getenv('GOOGLE_CREDENTIALS_PRIVATE_KEY_ID'),
    "private_key": os.getenv('GOOGLE_CREDENTIALS_PRIVATE_KEY'),
    "client_email": os.getenv('GOOGLE_CREDENTIALS_CLIENT_EMAIL'),
    "client_id": os.getenv('GOOGLE_CREDENTIALS_CLIENT_ID'),
    "auth_uri": os.getenv('GOOGLE_CREDENTIALS_AUTH_URI'),
    "token_uri": os.getenv('GOOGLE_CREDENTIALS_TOKEN_URI'),
    "auth_provider_x509_cert_url": os.getenv('GOOGLE_CREDENTIALS_AUTH_PROVIDER_X509_CERT_URL'),
    "client_x509_cert_url": os.getenv('GOOGLE_CREDENTIALS_CLIENT_X509_CERT_URL'),
    "universe_domain": os.getenv('GOOGLE_CREDENTIALS_UNIVERSE_DOMAIN')
}

GENAI_API_KEY = os.getenv('GENAI_API_KEY')

# Sauvegarder le contenu JSON dans un fichier temporaire
with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
    json.dump(GOOGLE_CREDENTIALS_PATH, f)
    temp_file_path = f.name

# Configurer les API en utilisant ce fichier temporaire
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path
genai.configure(api_key=GENAI_API_KEY)

# Initialisation des clients Google Cloud
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

# Variable globale pour l'historique des conversations
conversation_history: List[Dict] = []

@app.route('/')
def read_root():
    return jsonify({"message": "Hello World"})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Aucun fichier fourni"}), 400
            
        file = request.files['file']
        contents = file.read()
        
        audio = speech.RecognitionAudio(content=contents)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="fr-FR"
        )
        
        response = speech_client.recognize(config=config, audio=audio)
        if not response.results:
            return jsonify({"error": "Aucune transcription obtenue"}), 400
            
        transcript = response.results[0].alternatives[0].transcript
        return jsonify({"text": transcript})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-response', methods=['POST'])
def generate_response():
    try:
        data = request.json
        text = data.get('text')
        if not text:
            return jsonify({"error": "Aucun texte fourni"}), 400

        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        
        # Création du contexte avec l'historique
        context = "Historique de la conversation:\n"
        for exchange in conversation_history:
            context += f"Q: {exchange['question']}\nR: {exchange['response']}\n"
        
        instructions = f"""
        Tu es un assistant vocal professionnel engagé dans une conversation continue. 
        Utilise l'historique de la conversation pour maintenir le contexte.
        
        {context}
        
        Important:
        - NE COMMENCE PAS par "Bonjour" ou autres formules d'introduction sauf pour la première interaction
        - Utilise le contexte précédent pour comprendre les références
        - N'utilise PAS de symboles spéciaux
        - Privilégie un vocabulaire professionnel mais accessible
        - Utilise des phrases concises et bien structurées
        - Maintiens un ton courtois et professionnel tout en restant chaleureux
        
        Question actuelle de l'utilisateur: {text}
        """
        
        response = model.generate_content(instructions)
        clean_response = response.text.replace('*', '').replace('-', '').replace('#', '')
        
        # Ajout à l'historique
        conversation_history.append({
            'question': text,
            'response': clean_response
        })
        
        return jsonify({"text": clean_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/text-to-speech', methods=['POST'])
def convert_text_to_speech():
    try:
        data = request.json
        text = data.get('text')
        if not text:
            return jsonify({"error": "Aucun texte fourni"}), 400

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="fr-FR",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Création d'un stream pour renvoyer l'audio
        audio_content = io.BytesIO(response.audio_content)
        audio_content.seek(0)
        
        return send_file(
            audio_content,
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name="response.mp3"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset-conversation', methods=['DELETE'])
def reset_conversation():
    global conversation_history
    conversation_history = []
    return jsonify({"message": "Historique de conversation réinitialisé"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

# Procfile
"""
web: gunicorn main:app
"""

# runtime.txt
"""
python-3.9.18
"""
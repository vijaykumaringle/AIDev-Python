from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from flask_cors import CORS
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define the prompts
guess_prompt_template = "Guess the song based on the following clues: {clues}. Mention movie/album. Mention Artist. Add a one line description of the song."
lyrics_prompt_template = "Get the lyrics first paragraph for the song {song_name}"
meaning_prompt_template = "Get the meaning of the first paragraph of the song {song_name}. And also mention a fun fact about the song."

@app.route('/guess', methods=['POST'])
def guess_song():
    try:
        data = request.get_json()
        clues = data.get('clues')

        if not clues:
            return jsonify({'error': 'Clues are required'}), 400

        prompt = guess_prompt_template.format(clues=clues)
        logging.debug(f"Generated prompt for guessing song: {prompt}")
        guess = llm.invoke(prompt)
        logging.debug(f"Received response for guessing song: {guess}")

        # Extract the text from the response object 
        if hasattr(guess, 'content'): 
            guess_text = guess.content 
        else: 
            raise ValueError(f"Unexpected response format: {guess}")

        return jsonify({'guess': guess_text})

    except Exception as e:
        logging.error(f"Error in /guess endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_song():
    try:
        data = request.get_json()
        song_name = data.get('song_name')

        if not song_name:
            return jsonify({'error': 'Song name is required'}), 400

        lyrics_prompt = lyrics_prompt_template.format(song_name=song_name)
        meaning_prompt = meaning_prompt_template.format(song_name=song_name)

        logging.debug(f"Generated lyrics prompt: {lyrics_prompt}")
        logging.debug(f"Generated meaning prompt: {meaning_prompt}")

        lyrics_response = llm.invoke(lyrics_prompt)
        meaning_response = llm.invoke(meaning_prompt)

        logging.debug(f"Received lyrics response: {lyrics_response}")
        if not lyrics_response:
            raise ValueError("Received empty lyrics response from LLM")
        if not meaning_response:
            raise ValueError("Received empty meaning response from LLM")

        #lyrics = lyrics_response.strip()
        #meaning = meaning_response.strip()
        
        # Extract the text from the response objects 
        if hasattr(lyrics_response, 'content'): 
            lyrics = lyrics_response.content 
        else: 
            lyrics = 'Unexpected response format for lyrics' 
            
        if hasattr(meaning_response, 'content'): 
            meaning = meaning_response.content 
        else: 
            meaning = 'Unexpected response format for meaning'
        
        return jsonify({'lyrics': lyrics, 'meaning': meaning})

    except Exception as e:
        logging.error(f"Error in /analyze endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
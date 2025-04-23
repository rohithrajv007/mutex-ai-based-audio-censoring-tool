# mutex-ai-based-audio-censoring-tool


Here’s a suggested README for your project:

---

# **Audio Censorship Web Application**

## **Project Description**

This web application allows users to upload audio files, transcribe them into text, and automatically detect and censor offensive or inappropriate language. It utilizes a combination of speech recognition and natural language processing techniques to identify bad words, replace them in the transcription with placeholders, and apply censorship to the audio. The application also uses the Vosk model for speech-to-text conversion, enabling accurate recognition of words and timestamps.

The key features include:

- **Audio Upload**: Users can upload audio files in various formats (e.g., MP3, WAV).
- **Transcription**: The uploaded audio is transcribed into text using Google Speech Recognition.
- **Bad Word Detection**: A pre-defined set of offensive words is used to identify bad words in the transcribed text.
- **Audio Censorship**: If bad words are found, the application censors them in both the transcribed text (by replacing them with "____") and the audio (using either beeping or silence insertion).
- **Efficient Processing**: The application uses background threading and caching techniques to improve performance, especially for audio processing and transcription.
- **Vosk Model Integration**: For more advanced word timestamp detection and forced alignment, the Vosk model is used to provide precise word timings, enhancing the censorship accuracy.

## **Technologies Used**

- **Flask**: A micro web framework for building the web application.
- **Speech Recognition**: Google Speech API and Vosk for speech-to-text transcription.
- **Pydub**: Used for audio file manipulation, including conversion between formats and applying audio effects.
- **Python**: Backend logic, including audio processing, file handling, and word filtering.
- **Wave**: For handling audio files in the WAV format.
- **Threading & Concurrency**: To handle background tasks such as model loading and audio conversion without blocking the main application.

## **Features**

1. **Upload Audio File**: Users can upload an audio file for processing.
2. **Transcribe Audio**: The uploaded audio is transcribed into text.
3. **Bad Word Detection**: Offensive words are detected in the transcription and flagged.
4. **Censoring**: The application offers two censoring methods—beeping or silencing bad words in the audio.
5. **Results Display**: After processing, the application displays the censored transcription and provides the option to download the censored audio file.

## **Setup Instructions**

1. Clone the repository:


2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the Vosk model:
   - Download the Vosk model from [Vosk GitHub](https://github.com/alphacep/vosk-api/blob/master/doc/models.md).
   - Extract the model and specify the model path in the application.

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Access the application in your browser at:
   ```
   http://127.0.0.1:5000/
   ```

## **Contributing**

If you'd like to contribute to this project, please fork the repository, create a new branch for your changes, and submit a pull request. Contributions are welcome to enhance the functionality or improve the accuracy of the model.


---

This README should give a concise and clear overview of the project while including the necessary details for setup and contribution. Let me know if you'd like to add or modify anything!

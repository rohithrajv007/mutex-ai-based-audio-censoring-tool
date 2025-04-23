from flask import Flask, request, send_from_directory, render_template
import os
import logging
import uuid
from pydub import AudioSegment
from pydub.generators import Sine
import speech_recognition as sr
from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import wave
import threading
import concurrent.futures
import numpy as np
import io
from functools import lru_cache

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
SetLogLevel(-1)
MODEL_PATH = r"C:\Users\rajro\Downloads\vosk-model-small-en-us-0.15"

# Load model in a background thread to avoid blocking app startup
model = None
model_loaded = False

def load_model_background():
    global model, model_loaded
    try:
        model = Model(MODEL_PATH)
        model_loaded = True
        logger.info("Loaded Vosk model successfully")
    except Exception as e:
        logger.error(f"Failed to load Vosk model: {e}")
        logger.warning("Audio word timing features will be disabled")
        model_loaded = False

# Start model loading in background
threading.Thread(target=load_model_background, daemon=True).start()

# Cache the bad words set for fast lookup
BAD_WORDS = {
    "hate", "abuse", "stupid", "idiot", "fool", "damn", "jerk", "moron", "loser", "dumb", 
    "bastard", "asshole", "bitch", "shit", "fuck", "cunt", "prick", "dick", "pussy", "fucking","shitiest","nigga",
    "whore", "slut", "faggot", "nigger", "chink", "gook", "spic", "kike", "wop", "dago", 
    "retard", "cripple", "freak", "tranny", "dyke", "queer", "savage", "ape", "mongrel", 
    "paki", "raghead", "towelhead", "cracker", "redneck", "hillbilly", "trash", "scum", 
    "pig", "dog", "coward", "sissy", "twat", "wanker", "arse", "bollocks", "bloody", 
    "piss", "arsehole", "shithead", "douche", "motherfucker", "ass", "cock", "tits", 
    "coon", "jap", "wetback", "beaner", "gringo", "yid", "heeb", "mick", "kraut", "limey",
    "skank", "slag", "tard", "gimpy", "midget", "halfwit", "dolt", "cretin", "numbskull", 
    "arsewipe", "tosser", "git", "prat", "knob", "bellend", "shite", "bugger", "sod", 
    "poof", "pansy", "nancy", "fruit", "chav", "yokel", "bogan", "pleb", "peasant", 
    "mong", "spaz", "lame", "goon", "thug", "brat", "snot", "puke", "sleaze", "perv", 
    "nonce", "wog", "boong", "abbo", "darky", "slope", "slant", "cameljockey", "jihadi", 
    "papist", "heathen", "infidel", "zionist", "commie", "pinko", "nazi", "femi-nazi", 
    "bimbo", "ditz", "cow", "hag", "crone", "harpy", "shrew", "frigid", "prude", "cuck"
}

@app.route('/')
def upload_form():
    return render_template('index.html')

# Optimize file handling with threading
def save_file_async(file, path):
    file.save(path)
    logger.info(f"Saved uploaded file to {path}")
    return path

@lru_cache(maxsize=16)
def generate_beep(duration_ms, sample_rate=44100):
    """Cache beep generation for common durations"""
    return Sine(1000).to_audio_segment(duration=duration_ms, volume=-3)

def convert_to_wav(audio_path, output_path):
    """Optimized audio conversion"""
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path

def fast_word_check(words):
    """Quickly find bad words using set operations"""
    bad_word_indices = []
    bad_words_found = []
    
    for i, word in enumerate(words):
        cleaned = ''.join(ch for ch in word.lower() if ch.isalpha())
        if cleaned in BAD_WORDS:
            bad_word_indices.append(i)
            bad_words_found.append(word)
    
    return bad_word_indices, bad_words_found

@app.route('/convert', methods=['POST'])
def convert_audio():
    if 'audio' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['audio']
    if file.filename == '':
        return "No selected file", 400
    
    try:
        censor_type = request.form.get('censor_type', 'beep')
        padding_ms = min(max(int(request.form.get('padding', 200)), 50), 500)
    except ValueError:
        padding_ms = 200

    file_extension = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    audio_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    try:
        # Save file asynchronously
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(save_file_async, file, audio_path)
            audio_path = future.result()
        
        # Convert to WAV if necessary
        wav_path = audio_path
        if file_extension != '.wav':
            try:
                wav_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.wav")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(convert_to_wav, audio_path, wav_path)
                    wav_path = future.result()
                logger.info(f"Converted to WAV: {wav_path}")
            except Exception as e:
                logger.error(f"Failed to convert audio format: {e}")
                return f"Error converting audio format: {str(e)}", 500

        # Transcribe the audio
        text, confidence = transcribe_audio(wav_path)
        if not text:
            return "Could not transcribe the audio. Please try again with a clearer recording.", 400

        words = text.split()
        display_words = words.copy()
        
        # Efficiently find bad words
        bad_word_indices, bad_words_found = fast_word_check(words)
        
        # Replace bad words with placeholder in display text
        for i in bad_word_indices:
            display_words[i] = "____"

        # Process audio if bad words found
        if bad_words_found:
            logger.info(f"Abusive words found: {bad_words_found}")
            
            # Get word timestamps using the most efficient method available
            timestamps = []
            if model_loaded:
                timestamps = get_word_timestamps(wav_path, bad_words_found, padding_ms)
                logger.info(f"Vosk found {len(timestamps)} word timestamps")
            
            if not timestamps:
                logger.warning("Vosk failed to find timestamps, using forced alignment")
                timestamps = forced_alignment(wav_path, text, bad_words_found, bad_word_indices, padding_ms)
            
            if not timestamps:
                logger.warning("Forced alignment failed, using time-based estimation")
                audio_duration = AudioSegment.from_wav(wav_path).duration_seconds
                timestamps = estimate_word_timestamps(len(words), bad_words_found, bad_word_indices, audio_duration, padding_ms)
            
            # Apply energy analysis for refinement
            timestamps = refine_timestamps_with_energy_analysis(wav_path, timestamps)
            
            # Censor words in audio
            output_filename = f"censored_{uuid.uuid4()}.wav"
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)
            success = censor_words_in_audio(wav_path, output_path, timestamps, censor_type)
            
            if not success:
                return "Failed to censor words in audio", 500
        else:
            # No censoring needed, just copy the file
            output_filename = f"processed_{uuid.uuid4()}.wav"
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)
            with open(wav_path, 'rb') as src_file:
                with open(output_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())
            logger.info("No abusive words found; output audio remains unchanged.")

        # Return results data for the template
        return render_template(
            'results.html', 
            display_words=display_words, 
            output_filename=output_filename, 
            bad_words_found=bad_words_found, 
            confidence=int(confidence * 100)
        )
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        return f"Error processing audio: {str(e)}", 500

def transcribe_audio(audio_path):
    """Optimized transcription function"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            # Use a timeout to prevent hanging on API calls
            text = recognizer.recognize_google(audio_data, show_all=True)
            if not text:
                return "", 0
                
            if isinstance(text, dict) and 'alternative' in text:
                best_result = text['alternative'][0]
                transcript = best_result['transcript']
                confidence = best_result.get('confidence', 0.8)
                return transcript, confidence
            elif isinstance(text, list) and len(text) > 0:
                return text[0], 0.8
            else:
                return text, 0.8
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand audio")
            return "", 0
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service: {e}")
            return "", 0

def get_word_timestamps(audio_path, target_words, padding_ms=200):
    """Optimized timestamp detection with parallel processing"""
    if not model_loaded:
        return []
        
    timestamps = []
    target_words_lower = [word.lower() for word in target_words]
    
    try:
        with wave.open(audio_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != 'NONE':
                logger.warning("Audio file not in required format for Vosk")
                return []
                
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            
            # Use larger chunks for faster processing
            chunk_size = 8192
            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    process_results(result, target_words_lower, timestamps, padding_ms)
                    
            result = json.loads(rec.FinalResult())
            process_results(result, target_words_lower, timestamps, padding_ms)
            
        return timestamps
    except Exception as e:
        logger.error(f"Error in Vosk processing: {e}", exc_info=True)
        return []

def process_results(result, target_words_lower, timestamps, padding_ms=200):
    """Process recognition results more efficiently"""
    if 'result' not in result:
        return
        
    # Create a set for faster lookups
    target_set = set(target_words_lower)
    
    for word_data in result['result']:
        word = word_data.get('word', '').lower()
        
        # Check if this word matches any target
        if word in target_set or any(target in word for target in target_set) or any(word in target for target in target_set):
            start_time = int(word_data['start'] * 1000)
            end_time = int(word_data['end'] * 1000)
            
            # Apply padding
            start_time = max(0, start_time - padding_ms)
            end_time = end_time + padding_ms
            
            # Find which target word matched
            matching_target = next((target for target in target_words_lower if target in word or word in target), word)
            timestamps.append((start_time, end_time, matching_target))

def forced_alignment(audio_path, transcript, target_words, target_indices, padding_ms=200):
    """More efficient forced alignment implementation"""
    try:
        # Load audio with memory-efficient method
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            total_duration_ms = int((frames / rate) * 1000)
            
        words = transcript.split()
        avg_word_duration = total_duration_ms / len(words)
        padding_ms_factor = padding_ms / 100
        word_padding = avg_word_duration * padding_ms_factor
        
        # Process all timestamps at once
        timestamps = []
        for i, word_idx in enumerate(target_indices):
            word = target_words[i]
            word_start = word_idx * avg_word_duration
            word_end = (word_idx + 1) * avg_word_duration
            word_start = max(0, word_start - word_padding)
            word_end = min(total_duration_ms, word_end + word_padding)
            timestamps.append((int(word_start), int(word_end), word))
            
        return timestamps
    except Exception as e:
        logger.error(f"Error in forced alignment: {e}", exc_info=True)
        return []

def estimate_word_timestamps(total_words, muted_words, mute_indices, audio_duration, padding_ms=200):
    """Faster timestamp estimation"""
    logger.info("Using estimated timestamps as last resort")
    
    word_duration = (audio_duration * 1000) / total_words
    padding_factor = padding_ms / 100
    padding = word_duration * padding_factor
    
    # Calculate all timestamps at once
    timestamps = [
        (max(0, int((idx * word_duration) - padding)), 
         int(((idx + 1) * word_duration) + padding), 
         muted_words[i])
        for i, idx in enumerate(mute_indices)
    ]
    
    return timestamps

def refine_timestamps_with_energy_analysis(audio_path, timestamps):
    """More efficient energy analysis"""
    try:
        # Use NumPy for faster audio processing
        audio = AudioSegment.from_wav(audio_path)
        audio_array = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        ms_per_sample = 1000 / sample_rate
        
        refined = []
        chunk_size = 20  # ms
        threshold = -35  # dB
        
        for start_time, end_time, word in timestamps:
            # Convert ms to samples
            analysis_start = max(0, start_time - 150)
            analysis_end = min(len(audio), end_time + 150)
            
            # Get segment
            segment = audio[analysis_start:analysis_end]
            segment_array = np.array(segment.get_array_of_samples())
            
            # Calculate chunk boundaries
            chunk_size_samples = int(chunk_size * sample_rate / 1000)
            num_chunks = len(segment_array) // chunk_size_samples
            
            # Calculate energy for each chunk
            energies = []
            for i in range(num_chunks):
                chunk = segment_array[i*chunk_size_samples:(i+1)*chunk_size_samples]
                # Convert to dB (avoid log of 0)
                rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0
                db = 20 * np.log10(max(rms, 1)) 
                energies.append(db)
            
            # Find first and last speech
            first_speech = next((i for i, e in enumerate(energies) if e > threshold), 0)
            last_speech = next((i for i in range(len(energies)-1, -1, -1) if energies[i] > threshold), len(energies)-1)
            
            # Calculate refined boundaries
            refined_start = analysis_start + (first_speech * chunk_size)
            refined_end = analysis_start + ((last_speech + 1) * chunk_size)
            
            # Apply safety padding
            refined_start = max(0, refined_start - 100)
            refined_end = min(len(audio), refined_end + 100)
            
            refined.append((refined_start, refined_end, word))
            
        return refined
    except Exception as e:
        logger.error(f"Error in energy analysis: {e}")
        return timestamps

def censor_words_in_audio(input_path, output_path, timestamps, censor_type='beep'):
    """Optimized audio censoring function"""
    try:
        original_audio = AudioSegment.from_wav(input_path)
        original_length = len(original_audio)
        
        # Sort and merge timestamps to reduce processing
        timestamps_sorted = sorted(timestamps, key=lambda x: x[0])
        merged_timestamps = []
        
        # Efficiently merge overlapping timestamps
        for start_time, end_time, word in timestamps_sorted:
            start_time = max(0, min(start_time, original_length))
            end_time = max(0, min(end_time, original_length))
            
            if start_time >= end_time:
                continue
                
            if merged_timestamps and start_time <= merged_timestamps[-1][1]:
                merged_timestamps[-1] = (
                    merged_timestamps[-1][0],
                    max(merged_timestamps[-1][1], end_time),
                    f"{merged_timestamps[-1][2]}, {word}"
                )
            else:
                merged_timestamps.append((start_time, end_time, word))
        
        # Build segments in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            segment_data = []
            
            # First segment
            if merged_timestamps and merged_timestamps[0][0] > 0:
                segment_data.append((0, merged_timestamps[0][0], None))
            
            # Middle segments (alternating original and censored)
            for i, (start_time, end_time, word) in enumerate(merged_timestamps):
                segment_data.append((start_time, end_time, word))
                
                # Add segment after this censored part
                if i < len(merged_timestamps) - 1 and end_time < merged_timestamps[i+1][0]:
                    segment_data.append((end_time, merged_timestamps[i+1][0], None))
            
            # Last segment
            if merged_timestamps and merged_timestamps[-1][1] < original_length:
                segment_data.append((merged_timestamps[-1][1], original_length, None))
            
            # If no timestamps, keep original
            if not merged_timestamps:
                segment_data = [(0, original_length, None)]
            
            # Process each segment
            for start, end, word in segment_data:
                if word is None:
                    # Original audio segment
                    futures.append(executor.submit(lambda s, e: original_audio[s:e], start, end))
                else:
                    # Censored segment
                    futures.append(executor.submit(
                        lambda s, e, w: generate_censor_segment(original_audio, s, e, w, censor_type),
                        start, end, word
                    ))
            
            # Collect results in order
            segments = [future.result() for future in futures]
        
        # Join segments
        if segments:
            censored_audio = sum(segments[1:], segments[0])
        else:
            censored_audio = original_audio
        
        # Ensure correct length
        if len(censored_audio) != original_length:
            if len(censored_audio) > original_length:
                censored_audio = censored_audio[:original_length]
            else:
                censored_audio = censored_audio + AudioSegment.silent(duration=original_length - len(censored_audio))
        
        # Export result
        censored_audio.export(output_path, format="wav")
        logger.info(f"Created censored audio file: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error censoring words in audio: {e}", exc_info=True)
        return False

def generate_censor_segment(original_audio, start_time, end_time, word, censor_type):
    """Helper function to create censored segments"""
    segment_duration = end_time - start_time
    
    if censor_type == 'beep':
        # Generate or fetch cached beep
        beep = generate_beep(segment_duration)
        beep = beep.fade_in(min(30, segment_duration // 4)).fade_out(min(30, segment_duration // 4))
        
        # Match volume to surrounding audio
        segment_before = original_audio[max(0, start_time-500):start_time]
        if len(segment_before) > 0:
            target_dbfs = segment_before.dBFS
            if target_dbfs > -float('inf'):
                beep = beep.apply_gain(target_dbfs - beep.dBFS + 5)
        
        # Ensure correct length
        if len(beep) != segment_duration:
            beep = beep[:segment_duration] if len(beep) > segment_duration else beep + AudioSegment.silent(duration=segment_duration - len(beep))
        
        logger.info(f"Replaced word(s) '{word}' with beep at {start_time}-{end_time}ms")
        return beep
    else:
        # Simple silence
        silence = AudioSegment.silent(duration=segment_duration)
        logger.info(f"Replaced word(s) '{word}' with silence at {start_time}-{end_time}ms")
        return silence

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.abspath(UPLOAD_FOLDER), filename)

if __name__ == '__main__':
    # Use threaded=True for better handling of multiple requests
    app.run(debug=False, threaded=True)
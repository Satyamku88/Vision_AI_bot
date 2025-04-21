import cv2
import face_recognition
import os
import numpy as np
import sounddevice as sd
import queue
import vosk
import json
import threading
import customtkinter as ctk
from PIL import Image
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import ollama
import logging
from ollama import Client
import requests
import time 
import os
import platform
import playsound
from ultralytics import YOLO
yolo_model = YOLO("yolov8n.pt")
os.environ["PATH"] += os.pathsep + "C:\ffmpeg-master-latest-win64-gpl\bin"
def query_ollama(prompt):
    try:
        client = Client(timeout=30)  # 30-second timeout
        response = client.generate(
            model='mistral',
            prompt=prompt,
            stream=False
        )
        return response['response']
    except requests.exceptions.Timeout:
        return "Sorry, my response took too long. Please try a simpler question."
    except Exception as e:
        return f"I'm having trouble: {str(e)}"
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("VisionBot")

# Initialize global states
GREETED = False
PROCESSING = False

# Initialize Vosk model
MODEL_PATH = "model/vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    logger.error(f"Vosk model missing at {MODEL_PATH}")
    exit(1)

model = vosk.Model(MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, 16000)
audio_queue = queue.Queue(maxsize=10)

# Load known faces
KNOWN_FACES = []
KNOWN_NAMES = []
known_faces_dir = "known_faces"

if os.path.exists(known_faces_dir):
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = face_recognition.load_image_file(f"{known_faces_dir}/{filename}")
            encoding = face_recognition.face_encodings(image)
            if encoding:
                KNOWN_FACES.append(encoding[0])
                KNOWN_NAMES.append(os.path.splitext(filename)[0])

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Cannot access webcam")
    exit(1)

# GUI Setup
ctk.set_appearance_mode("dark")
app = ctk.CTk()
app.title("Vision Bot")
app.geometry("800x600")

video_label = ctk.CTkLabel(app, text="")
video_label.pack()

status_label = ctk.CTkLabel(app, text="Status: Ready", text_color="green")
status_label.pack(pady=10)

# Replace the speak() function with this:
# Replace the speak() function with this:
def speak(text):
    """Convert text to speech using gTTS and play it with playsound (simpler)"""
    global PROCESSING
    try:
        logger.debug(f"Speaking: {text}")
        
        tts = gTTS(text=text, lang='en')
        tts.save("response.mp3")

        # Play MP3 safely using playsound
        playsound.playsound("response.mp3", block=True)

    except Exception as e:
        logger.error(f"Speech error: {e}")
    finally:
        if os.path.exists("response.mp3"):
            os.remove("response.mp3")
        PROCESSING = False

def audio_callback(indata, frames, time, status):
    """Audio input callback"""
    # Suppress overflow warnings by not logging them
    if status and status.input_overflow:
        return  # Skip overflow warnings
    audio_queue.put(bytes(indata))

def process_audio():
    """Process audio input continuously"""
    global PROCESSING
    logger.debug("Audio processor started")
    
    while True:
        try:
            # Clear queue if overflow occurs
            while audio_queue.qsize() > 5:
                audio_queue.get_nowait()
                
            data = audio_queue.get(timeout=0.5)
            
            # Process in small chunks
            chunk_size = 4000
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                if len(chunk) < 100:
                    continue
                
                if recognizer.AcceptWaveform(chunk):
                    result = json.loads(recognizer.Result())
                    text = result.get('text', '').strip().lower()
                    
                    if text and not PROCESSING:
                        logger.info(f"Command received: {text}")
                        PROCESSING = True
                        handle_command(text)
                        break  # Process one command at a time
                        
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Audio error: {e}")
            time.sleep(0.1)

def handle_command(command):
    """Handle recognized commands"""
    global GREETED

    status_label.configure(text=f"Processing: {command}", text_color="yellow")

    try:
        if "exit" in command or "stop" in command:
            speak("Goodbye! Shutting down.")
            app.after(1000, app.destroy)

        elif "what is this" in command :
            speak("wait ")

            ret, frame = cap.read()
            if not ret:
                speak("Sorry, I can't see anything right now.")
                return

            objects = detect_objects(frame)

            if objects:
                description = f"i think this is : {', '.join(objects)} maybe."
                speak(description)
                
                
            else:
                speak("I don't see anything I can recognize.")

        elif "who am i" in command or "who is this" in command:
                name = recognize_face()
                if name:
                    speak(f"Hi {name}, nice to see you!")
                else:
                    speak("Sorry, I don't recognize this person.") 

        elif "hello" in command and not GREETED:
            speak("Hello Satyam! How can I help you?")
            GREETED = True

        else:
            response = query_ollama(command)
            speak(response)

    except Exception as e:
        logger.error(f"Command error: {e}")
        speak("Sorry, I encountered an error")

    status_label.configure(text="Status: Ready", text_color="green")

def recognize_face():
    """Detect a face and return the recognized name"""
    ret, frame = cap.read()
    if not ret:
        return None

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(KNOWN_FACES, face_encoding)
        face_distances = face_recognition.face_distance(KNOWN_FACES, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            return KNOWN_NAMES[best_match_index]

    return None

def detect_objects(frame):
    results = yolo_model.predict(frame, verbose=False)
    labels = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = r.names.get(cls_id, "unknown")
            labels.append(label)
    logger.info(f"Detected: {labels}")
    return labels


  
def update_camera():
    """Update camera feed and detect face with greeting"""
    global GREETED

    try:
        ret, frame = cap.read()
        if not ret:
            return

        # Face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(KNOWN_FACES, face_encoding)
            face_distances = face_recognition.face_distance(KNOWN_FACES, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = KNOWN_NAMES[best_match_index]
                if not GREETED:
                    speak(f"Hi {name}, good to see you!")
                    GREETED = True
                break

        # Update GUI
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        ctk_img = ctk.CTkImage(light_image=img, size=(800, 600))
        video_label.configure(image=ctk_img)

    except Exception as e:
        logger.error(f"Camera error: {e}")

    app.after(100, update_camera)

        
       

# Start threads
audio_thread = threading.Thread(target=process_audio, daemon=True)
audio_thread.start()

# Start audio input
# Start audio input
# Modify the audio input section:
with sd.RawInputStream(
    samplerate=16000,
    blocksize=8000,
    dtype='int16',
    channels=1,
    callback=audio_callback,
    device=7,  
    latency='low'
):

    logger.info("System ready - Listening...")
    update_camera()
    app.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
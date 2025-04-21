# 🤖 VisionBot - Offline AI Voice & Vision Assistant

**VisionBot** is a smart AI assistant that runs fully offline on your Windows 11 PC. It can **see objects**, **recognize faces**, **listen to voice commands**, and respond using an **offline LLM** like DeepSeek. It even has a built-in **chat UI** that logs your interactions!

---

## 📹 What It Can Do

- 🎤 Voice command like “What do you see?”
- 👀 Detect objects from **live camera** (YOLOv8)
- 😊 Recognize known faces and greet them (e.g., "Hi Satyam!")
- 🧠 Answer intelligently using **DeepSeek** model via **Ollama**
- 💬 Chat UI with history (web-based interface)
- 📦 Runs fully offline — no internet needed after setup

---

## 📷 Testing Video

VIDEO LINK-
https://youtu.be/2Z7ZGd3dJwA

---

## 🛠️ Tech Stack

| Component      | Technology                        |
|----------------|------------------------------------|
| 🎥 Camera Feed | OpenCV                            |
| 🧠 Object AI   | YOLOv8 (Ultralytics)              |
| 😎 Face ID     | face_recognition + dlib           |
| 🎤 Voice Input | Vosk (offline speech recognition) |
| 🗣️ Voice Output| pyttsx3 (TTS) or gTTS             |
| 🌐 Web UI      | Flask (API), HTML + JS (frontend) |
| 🧠 LLM Model   | Ollama + DeepSeek 14B             |

---

## 🧾 Requirements

- Windows 11 PC
- Python 3.10
- Camera + Microphone
- Git & Ollama installed

---

## 📁 Folder Structure

```
vision_bot/
│
├── app.py                 # Main AI bot logic
├── chat_api.py           # Flask backend for chat UI
├── chat.html             # Frontend UI (opens in browser)
├── known_faces/          # Store face images (e.g., satyam.jpg)
├── vosk-model/           # Vosk voice recognition model
├── templates/            # Flask HTML (optional)
├── static/               # Custom styles/scripts (optional)
├── requirements.txt
```

---

## 🔽 Offline Model Downloads

### 🧠 DeepSeek via Ollama (LLM)
- [🔗 Ollama Site](https://ollama.com)
- Install Ollama → run model:
```bash
ollama run deepseek-r1:14b
```

### 🎤 Vosk Voice Model (Offline)
- [🔗 Download Vosk Small Model](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip)
- Extract to: `vosk-model-small-en-us-0.15`

---

## 🚀 How to Run

### 1. Clone and activate environment
```bash
git clone https://github.com/yourusername/visionbot.git
cd visionbot
python -m venv venv
venv\Scripts\activate
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

> If `dlib` fails, download prebuilt `.whl` here:  
> [🔗 Dlib Windows Binaries - Gohlke](https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib)

### 3. Run the bot
```bash
python app.py
```

### 4. Start web interface (optional)
```bash
python chat_api.py
```
Then open `chat.html` in your browser to view chat history.

---

## ✅ Features in Action

| Feature              | Status |
|----------------------|--------|
| Voice input (Vosk)   | ✅     |
| Object detection     | ✅     |
| Face recognition     | ✅     |
| AI response (LLM)    | ✅     |
| Web chat interface   | ✅     |
| Fully offline mode   | ✅     |

---

## 📷 Known Faces Setup

Put your face images in the `known_faces/` folder with meaningful names like:

```
known_faces/
├── satyam.jpg
├── rohan.png
```

The bot will greet detected faces by filename!

---

## 🧠 Model Info

### DeepSeek LLM (used with Ollama)

- Name: `deepseek-r1:14b`
- Size: ~9 GB
- Offline model used for reasoning based on detected objects/faces
- Works fully offline using [Ollama](https://ollama.com/)

### YOLOv8 (Ultralytics)

- Used to detect real-world objects (pen, bottle, person, etc.)
- Model file: `yolov8n.pt` (auto-downloaded)

---

## 📦 To-Do / Ideas

- [ ] Add emotion detection  
- [ ] Enable command training  
- [ ] Export to .exe app  
- [ ] Add face registration from webcam  
- [ ] Run on Raspberry Pi with Coral TPU  
- [ ] Add Telegram bot interface  

---

## 🙋‍♂️ Author

**Satyam Kumar Kasyap**  
🎓 BTech CSE | 2nd Year @ CUSAT  
📬 [GitHub](https://github.com/yourusername) • [LinkedIn](https://linkedin.com/in/yourusername)

---

## 📄 License

MIT License © 2025 [Satyam Kumar Kasyap](https://github.com/yourusername)

---

> ⚡ Feel free to fork, modify, or contribute — let’s build offline AI assistants that can see, hear, and think!

```

---

Let me know if you want this file exported as `.md` and placed in your project folder directly. I can also help you generate a `requirements.txt` and upload demo screenshots or videos!

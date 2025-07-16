# Sign_language
🤟 Real-Time Indian Sign Language (ISL) Gesture Recognition
This project uses MediaPipe, OpenCV, and Python to recognize a set of Indian Sign Language (ISL) gestures from webcam input in real time. The recognized gestures are displayed on-screen and logged with timestamps.

✨ Features
Real-time hand detection using MediaPipe Hands

Gesture smoothing for improved accuracy

Recognition of common ISL gestures like:

HELLO 👋

I LOVE YOU 🤟

YES ✊

STOP ✋

FPS display for performance monitoring

Logging of recognized gestures to a file (gesture_log.txt)

📸 Example Gestures
Gesture	Meaning
✋	HELLO
🤟	I LOVE YOU
✊	YES
✌️	STOP

🧰 Requirements
Python 3.7+

OpenCV

MediaPipe

NumPy

Install Dependencies
bash
Copy
Edit
pip install opencv-python mediapipe numpy
🚀 Usage
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/isl-gesture-recognition.git
cd isl-gesture-recognition
Run the program:

bash
Copy
Edit
python isl_gesture_recognition.py
Use your webcam and show gestures in front of the camera.

Press q to quit the program.

📁 Output
The program displays the detected gesture and FPS on the video stream.

All confirmed gestures are logged in gesture_log.txt with timestamps.

⚙️ Code Structure
isl_gesture_recognition.py — Main script for gesture recognition.

gesture_log.txt — Output file with logged gestures.

Uses deque buffer and landmark smoothing for robustness.

🙋‍♂️ Contributing
Pull requests and suggestions are welcome! Please fork the repo and open an issue or PR to discuss improvements or new features.

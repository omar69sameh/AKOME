# Multi-Modal Interaction System & Draw the Answer Game

A comprehensive, multi-modal drawing and interaction system that integrates various computer vision and hardware interfaces (eye gaze, hand gestures, facial recognition, and laser pointers) into a unified C# interactive educational game application. 

## 🌟 Overview
This project bridges the gap between advanced Python-based computer vision pipelines and a rich C# user interface. Originally featuring a "Draw the Answer" game alongside other educational tools, it allows users to interact with the screen using their eyes, their hands, or a physical laser pointer. Under the hood, custom Machine Learning models process and classify user inputs in real-time.

## 🏗 System Architecture

The project is split into two main layers:
1. **Python Vision/ML Backend**: A suite of independent python modules that analyze webcam/camera feeds and perform real-time tracking and inference (using libraries like OpenCV and MediaPipe).
2. **C# Interactive Frontend (`gui/`)**: A TUIO-compatible C# application that listens to the Python modules via Socket Communication. It features various mini-games, dynamic UI scaling, and multi-modal drawing support.

## 🧩 Core Components

- **Gaze Tracking (`gaze_tracking/`)**: Analyzes eye movement and dwell-time to dynamically scale UI elements (GazeDwellScaler) and navigate menus hands-free.
- **Gesture Recognition (`gesture/`, `hand_recorder/`)**: Detects hand landmarks using MediaPipe to map hand movements to on-screen drawing or pointer actions.
- **Laser Tracking (`laser_server/`)**: Uses color/intensity thresholds to track a physical laser pointer on a projected screen and digitize the stroke.
- **Face Processing (`face_emotion/`, `face_recognition/`)**: Identifies the current user and their emotional state to adapt game difficulty or record user reactions.
- **Object Tracking (`Objectt/`)**: Visual object detection for physical manipulative interactions.
- **Hardware Integration (`bluetooth/`)**: Manages external bluetooth hardware connections for auxiliary devices.
- **GUI & Games (`gui/`)**: The main application. Includes:
  - *DrawTheAnswerGame.cs*
  - *SpellingGame.cs*
  - *WhatsInTheImageGame.cs*
  - *BuildSentenceGame.cs*

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ (for ML/Vision servers)
- .NET Framework (for the C# GUI)
- Webcams (for vision tracking)

### Installation
1. Clone the repository.
2. Activate your Python environment (e.g., `.venv`).
3. Install the required Python packages for each tracker (usually `pip install opencv-python mediapipe numpy`).
4. Build the C# solution located in `gui/TUIO_CSHARP.sln`.

### Running the System
You can launch the entire stack—all python tracking servers and the C# GUI—simultaneously using the provided batch script. 

Ensure your virtual environment is set up correctly, then run:

```cmd
run_all.bat
```

This will spin up the socket listeners, start the camera feeds, and launch the interactive games.

## 🤝 How the Communication Works
The Python modules calculate the exact screen coordinates or classifications (e.g., gaze `(x, y)` or gesture `INDEX_FINGER_UP`). They send this data over local TCP/UDP sockets to the C# application, which uses bridges (like `GazeTrackingBridge.cs` and `FaceSocketListener.cs`) to update the cursor position, draw lines on a canvas, or change the UI state in real-time.

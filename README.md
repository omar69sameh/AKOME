# Multi-Modal Interaction System

This project is a multi-modal tracking and interaction system that combines various computer vision and hardware interfaces to provide a rich user experience.

## Components

- **Gaze Tracking (`gaze_tracking/`)**: Systems for tracking user eye movement and attention.
- **Gesture Recognition (`gesture/`, `hand_recorder/`)**: Hand gesture detection and recording using MediaPipe.
- **Face Processing (`face_emotion/`, `face_recognition/`)**: Facial feature recognition and emotion analysis.
- **Hardware Interfaces (`bluetooth/`, `laser_server/`)**: Connectivity and laser pointer tracking servers.
- **Object Tracking (`Objectt/`)**: Visual object detection and tracking.
- **GUI (`gui/`)**: The main user interface, integrating the different input modalities.

## Running the System

You can run all necessary services simultaneously using the provided batch script:
```bash
./run_all.bat
```

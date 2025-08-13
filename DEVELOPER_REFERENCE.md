# Developer Reference: Tech Stack & Functions

This document provides an in-depth overview of the technologies and key functions used in the Lost & Found Advanced Face Recognition System. It is intended for developers contributing to or maintaining the project.

---

## 🛠️ Tech Stack Details

### Backend
- **Flask**: Lightweight Python web framework for RESTful APIs and web interface.
- **OpenCV**: Image and video processing, camera access, frame extraction.
- **PyTorch**: Deep learning framework, GPU acceleration for face recognition.
- **facenet-pytorch**: Pre-trained MTCNN (face detection) and FaceNet (embedding extraction).
- **Pinecone**: Vector database for fast similarity search and face matching.
- **JSON**: Metadata storage for enrolled faces and recognition stats.

### Frontend
- **HTML5**: Structure for web interface and camera preview.
- **CSS3**: Styling, responsive design, status indicators.
- **JavaScript**: Camera control, AJAX requests, UI feedback, error handling.

### Hardware
- **NVIDIA CUDA**: GPU acceleration (RTX 4060 Laptop GPU recommended).

---
## 🧩 Key Python Functions & Classes

### app.py
- `output_file()`: Streams video files with HTTP range support for browser preview.
- `camera_stream()`: Streams MJPEG frames from laptop camera for live recognition.
- `start_camera()`, `stop_camera()`: API endpoints to control camera feed.
- `enroll_face()`: Handles new face enrollment and metadata update.
- `process_media()`: Processes uploaded images/videos for face detection and recognition.
- `get_analytics()`, `get_performance()`: Returns system stats and performance metrics.

### model.py / model3.py
- `FaceRecognitionModel`: Loads MTCNN and FaceNet models, manages device selection (CPU/GPU).
- `get_face_embedding(image)`: Extracts 512-dim embedding from detected face.
- `match_face(embedding)`: Searches Pinecone for closest match, returns metadata.
- `update_recognition_count(face_id)`: Tracks recognition frequency for analytics.
- `load_metadata()`, `save_metadata()`: Reads/writes face metadata from/to JSON.

### Camera & Video
- `cv2.VideoCapture`: Accesses camera hardware, supports multiple backends (DirectShow, Media Foundation, CAP_ANY).
- `cv2.imencode`: Encodes frames for MJPEG streaming.

---

## 🧩 Key JavaScript Functions

### face_recognition_advanced.html
- `startCamera()`: Initiates live camera feed via API, displays MJPEG stream.
- `stopCamera()`: Stops camera feed and releases resources.
- `showMessage(msg, type)`: Displays status/error messages to user.
- `enrollFace()`: Sends AJAX request to enroll new face.
- `updateFaceDatabase()`: Refreshes face database UI after changes.
- `handleRecognitionResult(data)`: Updates UI with recognition results.

---

## 🗂️ Data & Metadata
- **enrolled_faces/metadata.json**: Stores face IDs, names, image paths, recognition counts, and other metadata.
- **Pinecone Vector DB**: Stores face embeddings for fast similarity search.

---

## 🧑‍💻 Developer Notes
- All endpoints return JSON for easy integration.
- Camera access uses multiple backends for Windows compatibility.
- GPU acceleration is auto-detected; falls back to CPU if unavailable.
- Error handling and status feedback are integrated throughout frontend and backend.
- Modular code structure for easy extension and maintenance.
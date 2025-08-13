# Lost & Found - Advanced Face Recognition System

## Overview
Lost & Found is a best-in-class face recognition system designed for real-time identification, video analysis, and database management. Built with modern AI and web technologies, it provides a seamless workflow for uploading, processing, and recognizing faces from images, videos, and live camera feeds.

---

## 🚀 Workflow
1. **Upload Media**: Users can upload images or videos via the web interface.
2. **Face Enrollment**: New faces are enrolled into the database with metadata and images.
3. **Live Camera Recognition**: Activate the laptop camera for real-time face recognition.
4. **Face Matching**: The system detects and matches faces using deep learning and vector search.
5. **Database Management**: View, update, and manage enrolled faces and their metadata.
6. **Analytics & Monitoring**: Access system health, performance, and recognition analytics.

---

## 🛠️ Tech Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: PyTorch, facenet-pytorch (MTCNN, FaceNet)
- **Database**: Pinecone (vector database for face embeddings)
- **Image/Video Processing**: OpenCV
- **GPU Acceleration**: NVIDIA CUDA (RTX 4060 Laptop GPU)

---

## 🧩 Key Functions & Endpoints
- `/` : Main web interface
- `/api/health` : System health & GPU status
- `/api/enroll` : Enroll new faces
- `/api/process` : Process uploaded media
- `/api/batch-process` : Batch file processing
- `/api/camera/start` : Start live camera feed
- `/api/analytics` : View analytics
- `/api/performance` : Performance monitoring
- `/api/cleanup` : Smart file cleanup
- `/api/export` : Data export & backup
- `/api/face-database` : Face database management
- `/api/optimize-settings` : Smart optimization
- `/api/video-enhance` : Video enhancement
- `/api/security-status` : Security overview

---

## ✨ Features
- **Video Preview**: Stream and preview videos with scrubbing support
- **Live Camera**: Real-time face recognition from laptop camera
- **Face Enrollment**: Add new faces with images and metadata
- **Face Matching**: Fast, accurate recognition using deep learning and Pinecone
- **Database Management**: Update, view, and manage enrolled faces
- **Analytics Dashboard**: Monitor recognition stats and system health
- **Performance Monitoring**: GPU status, memory usage, and optimization
- **Security Overview**: System security status and recommendations
- **Responsive UI**: Mobile-friendly, modern web interface
- **Error Handling**: Comprehensive feedback and status messages

---

## 📦 Directory Structure
```
app.py                  # Main Flask backend
index.html              # Main web interface
face_recognition_advanced.html # Advanced camera & recognition UI
model.py, model3.py     # Face recognition models & Pinecone integration
__pycache__/            # Python cache files
enrolled_faces/         # Face images and metadata
    metadata.json       # Face metadata and recognition stats
```

---

## 📝 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Set Pinecone API key: `$env:PINECONE_API_KEY='your-key'` (PowerShell)
3. Start the server: `python app.py`
4. Open [http://localhost:5000](http://localhost:5000) in your browser

---

## 🏆 Credits
Developed by Omkar Deshpande and contributors.

---

## 📖 License
MIT License

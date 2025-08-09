import cv2
import torch
import numpy as np
import os
import logging
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import tempfile
import shutil

try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
    from ultralytics import YOLO
    from pinecone import Pinecone, ServerlessSpec
    import torchvision.transforms as transforms
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    logging.error(f"Missing required package: {e}")
    print(f"Please install missing package: {e}")
    print("Run: pip install -r requirements.txt")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionModel:
    """Face Recognition Model using FaceNet and Pinecone for vector storage"""
    
    def __init__(self, pinecone_api_key: str, pinecone_env: str = "us-east-1"):
        """
        Initialize the face recognition model
        
        Args:
            pinecone_api_key: Pinecone API key
            pinecone_env: Pinecone environment
        """
        # Check CUDA availability and setup device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device('cpu')
            logger.info("⚠️  No GPU detected, using CPU")
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.INDEX_NAME = "face-recognition-index"
        self.EMBEDDING_DIM = 512
        self.threshold = 0.6
        
        self._setup_pinecone()
        self._load_models()
        self._setup_transforms()
    
    def _setup_pinecone(self):
        """Setup Pinecone index"""
        try:
            # Create index if it doesn't exist
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.INDEX_NAME not in existing_indexes:
                self.pc.create_index(
                    name=self.INDEX_NAME,
                    dimension=self.EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                logger.info(f"Created Pinecone index: {self.INDEX_NAME}")
            
            self.index = self.pc.Index(self.INDEX_NAME)
            logger.info("Connected to Pinecone index")
        except Exception as e:
            logger.error(f"Failed to setup Pinecone: {e}")
            raise
    
    def _load_models(self):
        """Load face detection and recognition models"""
        try:
            logger.info("Loading face recognition models...")
            
            # Face recognition model (FaceNet)
            self.facenet_model = InceptionResnetV1(
                pretrained='vggface2'
            ).eval().to(self.device)
            
            # Face detection model (MTCNN)
            self.mtcnn = MTCNN(
                keep_all=True, 
                device=self.device,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False
            )
            
            # Optimize for GPU if available
            if torch.cuda.is_available():
                # Enable optimizations for GPU
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Warm up the GPU
                logger.info("Warming up GPU...")
                dummy_input = torch.randn(1, 3, 160, 160).to(self.device)
                with torch.no_grad():
                    _ = self.facenet_model(dummy_input)
                logger.info("✅ GPU warmed up successfully")
            
            logger.info("✅ Models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    
    def get_embedding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding from cropped face image
        
        Args:
            face_bgr: BGR face image as numpy array
            
        Returns:
            Face embedding or None if invalid
        """
        try:
            # Validate crop size
            if face_bgr.shape[0] < 20 or face_bgr.shape[1] < 20:
                return None
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # Preprocess and get embedding
            face_tensor = self.preprocess(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.facenet_model(face_tensor)
                # Move to CPU immediately to free GPU memory
                embedding = embedding.cpu().numpy()[0]
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Clear GPU cache on error too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
    
    def enroll_face(self, image_path: str, person_id: str, name: str) -> bool:
        """
        Enroll a new face in the database
        
        Args:
            image_path: Path to the image file
            person_id: Unique person ID
            name: Person's name
            
        Returns:
            True if enrollment successful, False otherwise
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Detect faces
            boxes, _ = self.mtcnn.detect(img)
            if boxes is None or len(boxes) == 0:
                logger.error(f"No faces detected in {image_path}")
                return False
            
            # Process the first detected face
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box)
            
            # Add padding to the bounding box
            h, w = img.shape[:2]
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            face_crop = img[y1:y2, x1:x2]
            embedding = self.get_embedding(face_crop)
            
            if embedding is not None:
                # Store in Pinecone
                self.index.upsert(vectors=[(
                    person_id, 
                    embedding.tolist(), 
                    {"name": name, "image_path": image_path}
                )])
                logger.info(f"Enrolled {name} with ID {person_id}")
                return True
            else:
                logger.error("Failed to get embedding for face")
                return False
                
        except Exception as e:
            logger.error(f"Error enrolling face: {e}")
            return False
    
    def recognize_face(self, face_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a face from its embedding
        
        Args:
            face_embedding: Face embedding vector
            
        Returns:
            Tuple of (recognized_name, confidence_score)
        """
        try:
            # Query Pinecone
            query_result = self.index.query(
                vector=face_embedding.tolist(), 
                top_k=1, 
                include_metadata=True
            )
            
            if query_result['matches'] and len(query_result['matches']) > 0:
                best_match = query_result['matches'][0]
                score = best_match['score']
                metadata = best_match.get('metadata', {})
                
                if score > self.threshold:
                    name = metadata.get('name', "Unknown")
                    return name, score
            
            return "Unknown", 0.0
            
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return "Unknown", 0.0
    
    def process_video(self, video_path: str, output_path: str = None) -> str:
        """
        Process video for face recognition
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)
            
        Returns:
            Path to output video
        """
        if output_path is None:
            output_path = "recognized_output.mp4"
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup video writer with better codec for web compatibility
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for better compatibility
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error("Failed to open video writer")
                raise ValueError("Could not initialize video writer")
            
            logger.info(f"Processing {total_frames} frames on {self.device}...")
            logger.info(f"Video specs: {width}x{height} @ {fps}fps")
            
            frame_count = 0
            faces_detected_total = 0
            recognized_faces = set()
            batch_size = 8 if torch.cuda.is_available() else 1  # Process multiple frames at once on GPU
            
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Detect faces
                    boxes, _ = self.mtcnn.detect(frame)
                    
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Add padding
                            padding = 5
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding)
                            x2 = min(width, x2 + padding)
                            y2 = min(height, y2 + padding)
                            
                            face_crop = frame[y1:y2, x1:x2]
                            embedding = self.get_embedding(face_crop)
                            
                            if embedding is not None:
                                faces_detected_total += 1
                                name, score = self.recognize_face(embedding)
                                
                                if name != "Unknown":
                                    recognized_faces.add(name)
                                
                                # Draw bounding box and label with better styling
                                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                thickness = 3
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                                
                                # Prepare label
                                label = f"{name}"
                                if name != "Unknown":
                                    confidence_percent = int(score * 100)
                                    label += f" ({confidence_percent}%)"
                                
                                # Calculate text size and background
                                font_scale = 0.7
                                font_thickness = 2
                                (text_width, text_height), baseline = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                                )
                                
                                # Draw background rectangle for text
                                text_y = y1 - 10
                                if text_y - text_height - 10 < 0:  # If too close to top, put text below
                                    text_y = y2 + text_height + 10
                                
                                cv2.rectangle(
                                    frame, 
                                    (x1, text_y - text_height - 5), 
                                    (x1 + text_width + 10, text_y + 5), 
                                    color, -1
                                )
                                
                                # Draw text
                                cv2.putText(
                                    frame, label, (x1 + 5, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness
                                )
                    
                    out.write(frame)
                    frame_count += 1
                    pbar.update(1)
            
            cap.release()
            out.release()
            
            logger.info(f"✅ Processed {frame_count} frames. Output saved to: {output_path}")
            logger.info(f"📊 Statistics: {faces_detected_total} faces detected, {len(recognized_faces)} unique people recognized")
            if recognized_faces:
                logger.info(f"👥 Recognized people: {', '.join(recognized_faces)}")
            
            return output_path, {
                'total_frames': frame_count,
                'faces_detected': faces_detected_total,
                'unique_people': len(recognized_faces),
                'recognized_names': list(recognized_faces)
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
    
    def process_image(self, image_path: str, output_path: str = None) -> tuple:
        """
        Process single image for face recognition
        
        Args:
            image_path: Path to input image
            output_path: Path to output image (optional)
            
        Returns:
            Tuple of (output_path, statistics_dict)
        """
        if output_path is None:
            output_path = "recognized_image.jpg"
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            faces_detected = 0
            recognized_faces = set()
            
            # Detect faces
            boxes, _ = self.mtcnn.detect(img)
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Add padding
                    h, w = img.shape[:2]
                    padding = 10
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    
                    face_crop = img[y1:y2, x1:x2]
                    embedding = self.get_embedding(face_crop)
                    
                    if embedding is not None:
                        faces_detected += 1
                        name, score = self.recognize_face(embedding)
                        
                        if name != "Unknown":
                            recognized_faces.add(name)
                        
                        # Draw bounding box and label with better styling
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        thickness = 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                        
                        # Prepare label
                        label = f"{name}"
                        if name != "Unknown":
                            confidence_percent = int(score * 100)
                            label += f" ({confidence_percent}%)"
                        
                        # Calculate text size and background
                        font_scale = 0.9
                        font_thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                        )
                        
                        # Draw background rectangle for text
                        text_y = y1 - 15
                        if text_y - text_height - 10 < 0:  # If too close to top, put text below
                            text_y = y2 + text_height + 20
                        
                        cv2.rectangle(
                            img, 
                            (x1, text_y - text_height - 8), 
                            (x1 + text_width + 15, text_y + 8), 
                            color, -1
                        )
                        
                        # Draw text
                        cv2.putText(
                            img, label, (x1 + 8, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness
                        )
            
            cv2.imwrite(output_path, img)
            logger.info(f"✅ Processed image saved to: {output_path}")
            logger.info(f"📊 Statistics: {faces_detected} faces detected, {len(recognized_faces)} people recognized")
            if recognized_faces:
                logger.info(f"👥 Recognized people: {', '.join(recognized_faces)}")
            
            return output_path, {
                'faces_detected': faces_detected,
                'unique_people': len(recognized_faces),
                'recognized_names': list(recognized_faces)
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    def get_enrolled_faces(self) -> List[Dict[str, Any]]:
        """
        Get list of enrolled faces
        
        Returns:
            List of enrolled face metadata
        """
        try:
            # Query all vectors (this is a simplified approach)
            # In production, you might want to use a separate metadata store
            stats = self.index.describe_index_stats()
            return {"total_faces": stats.total_vector_count}
        except Exception as e:
            logger.error(f"Error getting enrolled faces: {e}")
            return {"total_faces": 0}
    
    def delete_face(self, person_id: str) -> bool:
        """
        Delete a face from the database
        
        Args:
            person_id: Person ID to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            self.index.delete(ids=[person_id])
            logger.info(f"Deleted face with ID: {person_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting face: {e}")
            return False
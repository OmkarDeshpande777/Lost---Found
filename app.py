from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import os
import sys
import uuid
from datetime import datetime
import logging
from pathlib import Path
import json
from werkzeug.utils import secure_filename

# Import our face recognition model
from model3 import FaceRecognitionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['STATIC_FOLDER'] = 'static'

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['STATIC_FOLDER']]:
    Path(folder).mkdir(exist_ok=True)

# Initialize face recognition model
# You need to set your Pinecone API key as an environment variable
# Set it in PowerShell: $env:PINECONE_API_KEY="your-api-key-here"
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

if not PINECONE_API_KEY or PINECONE_API_KEY == 'your-pinecone-api-key-here':
    logger.warning("⚠️ PINECONE_API_KEY not set! Using demo mode.")
    logger.warning("To enable full functionality:")
    logger.warning("1. Get API key from https://pinecone.io")
    logger.warning("2. Set environment variable: $env:PINECONE_API_KEY='your-key'")
    face_model = None
else:
    try:
        face_model = FaceRecognitionModel(pinecone_api_key=PINECONE_API_KEY)
        logger.info("✅ Face recognition model initialized successfully with GPU support")
    except Exception as e:
        logger.error(f"❌ Failed to initialize face recognition model: {e}")
        face_model = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov', 'wmv'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image(filename):
    """Check if file is an image"""
    image_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions

def is_video(filename):
    """Check if file is a video"""
    video_extensions = {'mp4', 'avi', 'mov', 'wmv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open('face_recognition.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>Face Recognition System</h1>
        <p>Please ensure the HTML file is present in the same directory.</p>
        <p>The system is ready to accept API calls at:</p>
        <ul>
            <li>POST /api/enroll - Enroll a new face</li>
            <li>POST /api/process - Process image/video</li>
            <li>GET /api/faces - Get enrolled faces</li>
            <li>DELETE /api/faces/&lt;id&gt; - Delete a face</li>
        </ul>
        """

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    import torch
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'gpu_available': True,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
            'gpu_memory_allocated': f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            'gpu_memory_cached': f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
        }
    else:
        gpu_info = {'gpu_available': False, 'device': 'CPU'}
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': face_model is not None,
        'timestamp': datetime.now().isoformat(),
        'gpu_info': gpu_info,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'torch_version': torch.__version__
    })

@app.route('/api/enroll', methods=['POST'])
def enroll_face():
    """Enroll a new face"""
    if not face_model:
        return jsonify({'success': False, 'message': 'Face recognition model not initialized'}), 500

    try:
        # Validate request
        if 'enrollImage' not in request.files:
            return jsonify({'success': False, 'message': 'No image file provided'}), 400
        
        file = request.files['enrollImage']
        person_name = request.form.get('personName', '').strip()
        person_id = request.form.get('personId', '').strip()
        
        if not person_name or not person_id:
            return jsonify({'success': False, 'message': 'Person name and ID are required'}), 400
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if not allowed_file(file.filename) or not is_image(file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type. Please upload an image.'}), 400

        # Save uploaded file
        filename = secure_filename(f"{person_id}_{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[1].lower()}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        logger.info(f"Processing enrollment for {person_name} (ID: {person_id})")

        # Enroll face
        success = face_model.enroll_face(filepath, person_id, person_name)
        
        if success:
            return jsonify({
                'success': True, 
                'message': f'Successfully enrolled {person_name}',
                'person_id': person_id,
                'person_name': person_name
            })
        else:
            # Clean up file on failure
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'message': 'No faces detected in the image or enrollment failed'}), 400

    except Exception as e:
        logger.error(f"Error enrolling face: {e}")
        return jsonify({'success': False, 'message': f'Internal server error: {str(e)}'}), 500

@app.route('/api/process', methods=['POST'])
def process_media():
    """Process image or video for face recognition"""
    if not face_model:
        return jsonify({'success': False, 'message': 'Face recognition model not initialized'}), 500

    try:
        # Validate request
        if 'mediaFile' not in request.files:
            return jsonify({'success': False, 'message': 'No media file provided'}), 400
        
        file = request.files['mediaFile']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type'}), 400

        # Save uploaded file
        filename = secure_filename(f"process_{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[1].lower()}")
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        # Generate output filename
        output_filename = f"output_{uuid.uuid4().hex}.{'mp4' if is_video(file.filename) else 'jpg'}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        logger.info(f"Processing {'video' if is_video(file.filename) else 'image'}: {filename}")

        # Process media
        if is_video(file.filename):
            result_path, stats = face_model.process_video(input_path, output_path)
            faces_detected = stats.get('faces_detected', 0)
            additional_info = {
                'total_frames': stats.get('total_frames', 0),
                'unique_people': stats.get('unique_people', 0),
                'recognized_names': stats.get('recognized_names', [])
            }
        else:
            result_path, stats = face_model.process_image(input_path, output_path)
            faces_detected = stats.get('faces_detected', 0)
            additional_info = {
                'unique_people': stats.get('unique_people', 0),
                'recognized_names': stats.get('recognized_names', [])
            }

        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)

        return jsonify({
            'success': True,
            'message': 'Media processed successfully',
            'output_path': f'/outputs/{output_filename}',
            'faces_detected': faces_detected,
            'media_type': 'video' if is_video(file.filename) else 'image',
            'statistics': additional_info
        })

    except Exception as e:
        logger.error(f"Error processing media: {e}")
        # Clean up files on error
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        return jsonify({'success': False, 'message': f'Processing failed: {str(e)}'}), 500

@app.route('/api/faces', methods=['GET'])
def get_enrolled_faces():
    """Get list of enrolled faces"""
    if not face_model:
        return jsonify({'success': False, 'message': 'Face recognition model not initialized'}), 500

    try:
        # Get enrolled faces info
        faces_info = face_model.get_enrolled_faces()
        
        # This is a simplified response since Pinecone doesn't easily allow listing all vectors
        # In a production system, you'd maintain a separate metadata store
        return jsonify({
            'success': True,
            'faces': [],  # Would contain actual face data in production
            'total_count': faces_info.get('total_faces', 0),
            'message': f"Total enrolled faces: {faces_info.get('total_faces', 0)}"
        })

    except Exception as e:
        logger.error(f"Error getting enrolled faces: {e}")
        return jsonify({'success': False, 'message': f'Failed to retrieve faces: {str(e)}'}), 500

@app.route('/api/faces/<face_id>', methods=['DELETE'])
def delete_face(face_id):
    """Delete an enrolled face"""
    if not face_model:
        return jsonify({'success': False, 'message': 'Face recognition model not initialized'}), 500

    try:
        success = face_model.delete_face(face_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Face {face_id} deleted successfully'
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to delete face'}), 400

    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        return jsonify({'success': False, 'message': f'Deletion failed: {str(e)}'}), 500

@app.route('/api/convert-video', methods=['POST'])
def convert_video_for_web():
    """Convert video to web-compatible format using ffmpeg if available"""
    try:
        input_path = request.json.get('input_path')
        if not input_path or not os.path.exists(input_path):
            return jsonify({'success': False, 'message': 'Invalid input path'}), 400
        
        # Try to convert using ffmpeg for better web compatibility
        output_path = input_path.replace('.mp4', '_web.mp4')
        
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-i', input_path, 
                '-c:v', 'libx264', '-preset', 'fast', 
                '-c:a', 'aac', '-movflags', '+faststart',
                '-y', output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return jsonify({
                    'success': True,
                    'output_path': output_path.replace('outputs/', '/outputs/'),
                    'message': 'Video converted for web compatibility'
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': 'Video conversion failed, using original',
                    'output_path': input_path.replace('outputs/', '/outputs/')
                })
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return jsonify({
                'success': False, 
                'message': 'FFmpeg not available, using original video',
                'output_path': input_path.replace('outputs/', '/outputs/')
            })
            
    except Exception as e:
        logger.error(f"Error converting video: {e}")
        return jsonify({'success': False, 'message': f'Conversion error: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    """Serve processed output files"""
    response = send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    
    # Add headers for better video streaming
    if filename.endswith('.mp4'):
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Type'] = 'video/mp4'
    
    return response

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download processed files"""
    try:
        return send_from_directory(
            app.config['OUTPUT_FOLDER'], 
            filename, 
            as_attachment=True,
            download_name=f"processed_{filename}"
        )
    except FileNotFoundError:
        return jsonify({'success': False, 'message': 'File not found'}), 404

@app.route('/static/<filename>')
def static_file(filename):
    """Serve static files"""
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'success': False, 'message': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found errors"""
    return jsonify({'success': False, 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Face Recognition System...")
    print("📋 Available endpoints:")
    print("   - GET  /               : Main interface")
    print("   - GET  /api/health     : Health check")
    print("   - POST /api/enroll     : Enroll new face")
    print("   - POST /api/process    : Process media")
    print("   - GET  /api/faces      : Get enrolled faces")
    print("   - DELETE /api/faces/<id> : Delete face")
    print()
    
    if not face_model:
        print("⚠️  WARNING: Face recognition model failed to initialize!")
        print("   Please check your Pinecone API key and dependencies.")
    else:
        print("✅ Face recognition model initialized successfully!")
    
    print(f"🌐 Server starting on http://localhost:5000")
    print("📁 Make sure to place the HTML file as 'face_recognition.html' in the same directory")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
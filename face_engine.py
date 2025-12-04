import os
import cv2
import numpy as np
import onnxruntime as ort
import urllib.request
from pathlib import Path
from config import INPUT_SIZE, FACE_MARGIN, MIN_FACE_SIZE

class FaceEngine:
    def __init__(self, model_path="models/arcface_r50.onnx"):
        self.model_path = Path(model_path)
        self.model_dir = self.model_path.parent

        # Create model directory if it doesn't exist
        self.model_dir.mkdir(exist_ok=True)

        # Check if model exists and is valid
        self.model_available = False
        self.session = None

        try:
            # Try to download model if missing
            if not self.model_path.exists():
                self._download_model()

            # Try to load ONNX model
            if self.model_path.exists():
                self.session = ort.InferenceSession(str(self.model_path))
                self.model_available = True
                print("✅ ArcFace model loaded successfully!")
            else:
                print("❌ ArcFace model not found")
        except Exception as e:
            print(f"⚠️  ArcFace model loading failed: {e}")
            print("Face recognition will not work until model is fixed.")

        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # ArcFace preprocessing constants
        self.input_size = INPUT_SIZE
        self.mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
        self.std = np.array([127.5, 127.5, 127.5], dtype=np.float32)

    def _download_model(self):
        """Download ArcFace R50 model from InsightFace repository"""
        print("🤖 ArcFace model not found. Please download it manually:")
        print("   1. Go to: https://github.com/onnx/models/tree/main/vision/body_analysis/arcface")
        print("   2. Download: arcfaceresnet100-11.onnx")
        print("   3. Save as: models/arcface_r50.onnx")
        print("   4. Restart the application")
        print()
        print("Or use this direct link:")
        print("   curl -L 'https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-11.onnx' -o models/arcface_r50.onnx")
        print()

        # Don't raise exception, just warn and continue
        print("⚠️  Continuing without model. Face recognition will not work until model is downloaded.")
        return

    def preprocess_image(self, image):
        """Preprocess image for ArcFace model"""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3 and image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Resize to 112x112
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Normalize
        image = (image - self.mean) / self.std

        # Convert to NCHW format (batch_size, channels, height, width)
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension

        return image

    def detect_face_bgr(self, bgr):
        """Detect face in BGR image and return bbox"""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE,
            maxSize=(300, 300)
        )

        if len(faces) == 0:
            return None

        # Return the largest detected face (by area) to focus on la cara principal
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (x, y, x + w, y + h)  # Return as (x1, y1, x2, y2)

    def crop_align(self, bgr, bbox):
        """Crop and align face from BGR image using bbox"""
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        # Add margin
        margin = int(FACE_MARGIN * max(x2 - x1, y2 - y1))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(bgr.shape[1], x2 + margin)
        y2 = min(bgr.shape[0], y2 + margin)

        # Crop face
        face_img = bgr[y1:y2, x1:x2]

        # Convert to RGB for ArcFace
        if face_img.shape[2] == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        return face_img

    def embed(self, face):
        """Extract L2-normalized 512-D embedding from face image"""
        if face is None:
            return None
        # If model/session is not available, gracefully return None
        if not self.model_available or self.session is None:
            return None

        # Preprocess the face image
        processed_image = self.preprocess_image(face)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        result = self.session.run([output_name], {input_name: processed_image})
        embedding = result[0][0]  # Remove batch dimension

        # L2 normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def average_embeddings(self, embeddings):
        """Average multiple embeddings and L2 normalize"""
        if not embeddings:
            return None

        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        return avg_embedding

    def cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

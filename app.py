import os
import json
import base64
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from uuid import uuid4

import cv2
import numpy as np
import hnswlib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, Session, create_engine, select
from PIL import Image
import uvicorn

from face_engine import FaceEngine
from config import SIM_THRESHOLD, FRAME_INTERVAL_MS, SEARCH_LIMIT, CURRENCY, MAX_ELEMENTS, EF_CONSTRUCTION, M_INDEX

# Database models
class Student(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str
    grade: str
    balance: int = Field(default=0)
    photo_path: str = ""
    embedding: Optional[bytes] = None

class Product(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    price: int
    stock: int = Field(default=0)

class Transaction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: str
    product_id: int
    amount: int
    created_at: datetime = Field(default_factory=datetime.now)

# Pydantic models for API
class StudentCreate(BaseModel):
    name: str
    grade: str
    balance: int = 0

class StudentResponse(BaseModel):
    id: str
    name: str
    grade: str
    balance: int
    photo_path: str

class ProductCreate(BaseModel):
    name: str
    price: int
    stock: int = 0

class ProductUpdate(BaseModel):
    name: str
    price: int
    stock: int

class RecognitionResult(BaseModel):
    match: bool
    student: Optional[StudentResponse] = None
    score: float = 0.0

class ChargeRequest(BaseModel):
    student_id: str
    product_id: int

# Global variables
app = FastAPI(title="Cantina Face Recognition System")
face_engine = FaceEngine()

# Base directories (absolute paths)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Database setup
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "db.sqlite"
FACES_DIR = DATA_DIR / "faces"
INDEX_PATH = DATA_DIR / "index.bin"
INDEX_LABELS_PATH = DATA_DIR / "index_labels.json"

DATA_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SQLModel.metadata.create_all(engine)


def ensure_product_stock_column():
    """Ensure 'stock' column exists on Product table (SQLite migrations)."""
    with engine.connect() as conn:
        result = conn.exec_driver_sql("PRAGMA table_info(product)").all()
        # row[1] is column name
        if not any(row[1] == "stock" for row in result):
            conn.exec_driver_sql(
                "ALTER TABLE product ADD COLUMN stock INTEGER NOT NULL DEFAULT 0"
            )


# HNSW index settings
DIMENSION = 512

# Global index variables
index = None
index_labels = {}  # student_id -> index_position

def init_hnsw_index():
    """Initialize or load HNSW index"""
    global index, index_labels

    if INDEX_PATH.exists() and INDEX_LABELS_PATH.exists():
        # Load existing index
        index = hnswlib.Index(space='cosine', dim=DIMENSION)
        index.load_index(str(INDEX_PATH))

        with open(INDEX_LABELS_PATH, 'r') as f:
            index_labels = json.load(f)
    else:
        # Create new index
        index = hnswlib.Index(space='cosine', dim=DIMENSION)
        index.init_index(max_elements=MAX_ELEMENTS, ef_construction=EF_CONSTRUCTION, M=M_INDEX)
        index_labels = {}
        save_index()

def save_index():
    """Save HNSW index and labels to disk"""
    if index is not None:
        index.save_index(str(INDEX_PATH))
        with open(INDEX_LABELS_PATH, 'w') as f:
            json.dump(index_labels, f)

def rebuild_index():
    """Rebuild index from database"""
    global index, index_labels

    with Session(engine) as session:
        students = session.exec(select(Student)).all()

    # Create new index
    index = hnswlib.Index(space='cosine', dim=DIMENSION)
    index.init_index(max_elements=MAX_ELEMENTS, ef_construction=EF_CONSTRUCTION, M=M_INDEX)
    index_labels = {}

    embeddings = []
    student_ids = []

    for student in students:
        if student.embedding:
            # Convert bytes to numpy array
            emb = np.frombuffer(student.embedding, dtype=np.float32)
            embeddings.append(emb)
            student_ids.append(student.id)

    if embeddings:
        embeddings = np.array(embeddings)
        int_labels = np.arange(len(embeddings))
        index.add_items(embeddings, int_labels)
        # Map student_id -> int_label
        for i, student_id in enumerate(student_ids):
            index_labels[student_id] = int(i)

    save_index()

def search_similar(embedding, k=1):
    """Search for similar embeddings in index"""
    if index is None or index.get_current_count() == 0:
        return [], []

    # Search for k nearest neighbors
    labels, distances = index.knn_query(embedding.reshape(1, -1), k=k)

    # Convert cosine distance to similarity score
    similarities = 1 - distances[0]

    # Build inverse label map: int_label -> student_id
    inv_map = {v: k for k, v in index_labels.items()}

    # Filter by similarity threshold and translate labels
    results = []  # student_ids
    scores = []
    for label, similarity in zip(labels[0], similarities):
        sid = inv_map.get(int(label))
        if sid and similarity >= SIM_THRESHOLD:
            results.append(sid)
            scores.append(float(similarity))

    return (results[:k], scores[:k]) if results else ([], [])

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    ensure_product_stock_column()
    init_hnsw_index()

    # Rebuild index if empty but we have students in DB
    with Session(engine) as session:
        student_count = session.exec(select(Student)).all()

    if index.get_current_count() == 0 and len(student_count) > 0:
        rebuild_index()

@app.post("/api/products")
async def create_product(product: ProductCreate):
    """Create a new product"""
    with Session(engine) as session:
        # Check if product with same name already exists
        existing = session.exec(
            select(Product).where(Product.name == product.name)
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail="Product with this name already exists")
        
        # Create new product
        new_product = Product(
            name=product.name,
            price=product.price,
            stock=product.stock,
        )
        
        session.add(new_product)
        session.commit()
        session.refresh(new_product)
        
        return {
            "id": new_product.id,
            "name": new_product.name,
            "price": new_product.price,
            "stock": new_product.stock,
            "message": "Product created successfully",
        }

@app.get("/api/products")
async def list_products():
    """List all products"""
    with Session(engine) as session:
        products = session.exec(select(Product)).all()
        return [
            {
                "id": p.id,
                "name": p.name,
                "price": p.price,
                "stock": p.stock,
            }
            for p in products
        ]

@app.put("/api/products/{product_id}")
async def update_product(product_id: int, product: ProductUpdate):
    """Update an existing product (name, price, stock)."""
    with Session(engine) as session:
        db_product = session.get(Product, product_id)
        if not db_product:
            raise HTTPException(status_code=404, detail="Product not found")

        db_product.name = product.name
        db_product.price = product.price
        db_product.stock = product.stock
        session.add(db_product)
        session.commit()
        session.refresh(db_product)

        return {
            "id": db_product.id,
            "name": db_product.name,
            "price": db_product.price,
            "stock": db_product.stock,
        }

@app.post("/api/seed")
async def seed_products():
    """Seed demo products with IDs 1-9"""
    demo_products = [
        {"id": 1, "name": "Sandwich", "price": 350, "stock": 999},
        {"id": 2, "name": "Apple", "price": 100, "stock": 999},
        {"id": 3, "name": "Orange Juice", "price": 200, "stock": 999},
        {"id": 4, "name": "Yogurt", "price": 250, "stock": 999},
        {"id": 5, "name": "Cookie", "price": 150, "stock": 999},
        {"id": 6, "name": "Banana", "price": 80, "stock": 999},
        {"id": 7, "name": "Milk", "price": 180, "stock": 999},
        {"id": 8, "name": "Croissant", "price": 220, "stock": 999},
        {"id": 9, "name": "Water", "price": 50, "stock": 999},
    ]

    with Session(engine) as session:
        # Clear existing products
        session.exec("DELETE FROM product")

        for product_data in demo_products:
            product = Product(**product_data)
            session.add(product)

        session.commit()

    return {"message": f"Seeded {len(demo_products)} products"}

@app.get("/api/students")
async def search_students(query: str = Query("", description="Search query for student name"), limit: int = Query(SEARCH_LIMIT, description="Maximum results")):
    """Search students by name"""
    with Session(engine) as session:
        if query:
            students = session.exec(
                select(Student).where(Student.name.ilike(f"%{query}%")).limit(limit)
            ).all()
        else:
            students = session.exec(select(Student).limit(limit)).all()

        return [
            {
                "id": s.id,
                "name": s.name,
                "grade": s.grade,
                "balance": s.balance,
                "photo_url": f"/data/faces/{Path(s.photo_path).name}" if s.photo_path else '/default-avatar.png'
            }
            for s in students
        ]

@app.post("/api/enroll_one_shot")
async def enroll_one_shot(
    name: str = Form(...),
    grade: str = Form(...),
    frame: UploadFile = File(...)
):
    """Enroll student with single frame"""
    try:
        # Read and decode image
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Detect face
        bbox = face_engine.detect_face_bgr(img)
        if bbox is None:
            raise HTTPException(status_code=400, detail="No face detected")

        # Crop and align face
        face = face_engine.crop_align(img, bbox)
        if face is None:
            raise HTTPException(status_code=400, detail="Could not crop face")

        # Extract embedding
        embedding = face_engine.embed(face)
        if embedding is None:
            raise HTTPException(status_code=400, detail="Could not extract embedding")

        # Save thumbnail
        student_id = str(uuid4())
        photo_filename = f"{student_id}_thumb.jpg"
        photo_path = FACES_DIR / photo_filename

        # Resize to thumbnail
        thumb = cv2.resize(img, (120, 120), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(photo_path), thumb)

        # Create student record
        with Session(engine) as session:
            student = Student(
                id=student_id,
                name=name,
                grade=grade,
                balance=0,
                photo_path=f"data/faces/{photo_filename}",
                embedding=embedding.tobytes()  # Store as bytes
            )
            session.add(student)
            session.commit()

            # Add to HNSW index
            if index is not None:
                current_count = index.get_current_count()
                if current_count < MAX_ELEMENTS:
                    new_label = int(current_count)
                    index.add_items(embedding.reshape(1, -1), [new_label])
                    index_labels[student_id] = new_label
                    save_index()
                else:
                    print(f"⚠️ HNSW index full ({current_count}/{MAX_ELEMENTS}). Skipping index update.")

        return {
            "id": student_id,
            "name": name,
            "grade": grade,
            "photo_path": f"data/faces/{photo_filename}",
            "embedding_computed": True
        }

    except HTTPException as e:
        # Propagate intended HTTP errors (e.g., 400 No face detected)
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enroll error: {str(e)}")

@app.post("/api/enroll_burst")
async def enroll_burst(
    name: str = Form(...),
    grade: str = Form(...),
    frames: List[UploadFile] = File(...)
):
    """Enroll student with multiple frames (3-5)"""
    if len(frames) < 3 or len(frames) > 5:
        raise HTTPException(status_code=400, detail="Provide 3-5 frames")

    try:
        embeddings = []
        best_frame = None
        best_bbox = None
        max_area = 0

        # Process each frame
        for frame in frames:
            contents = await frame.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # Detect face
            bbox = face_engine.detect_face_bgr(img)
            if bbox is not None:
                # Calculate face area
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > max_area:
                    max_area = area
                    best_frame = img
                    best_bbox = bbox

                # Crop and align face
                face = face_engine.crop_align(img, bbox)
                if face is not None:
                    embedding = face_engine.embed(face)
                    if embedding is not None:
                        embeddings.append(embedding)

        if len(embeddings) < 3:
            raise HTTPException(status_code=400, detail="Could not detect faces in enough frames")

        # Average embeddings
        avg_embedding = face_engine.average_embeddings(embeddings)
        if avg_embedding is None:
            raise HTTPException(status_code=400, detail="Could not compute average embedding")

        # Save best frame as thumbnail
        student_id = str(uuid4())
        photo_filename = f"{student_id}_thumb.jpg"
        photo_path = FACES_DIR / photo_filename

        if best_frame is not None:
            thumb = cv2.resize(best_frame, (120, 120), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(photo_path), thumb)

        # Create student record
        with Session(engine) as session:
            student = Student(
                id=student_id,
                name=name,
                grade=grade,
                balance=0,
                photo_path=f"data/faces/{photo_filename}" if best_frame is not None else "",
                embedding=avg_embedding.tobytes()  # Store as bytes
            )
            session.add(student)
            session.commit()

            # Add to HNSW index
            if index is not None:
                current_count = index.get_current_count()
                if current_count < MAX_ELEMENTS:
                    new_label = int(current_count)
                    index.add_items(avg_embedding.reshape(1, -1), [new_label])
                    index_labels[student_id] = new_label
                    save_index()
                else:
                    print(f"⚠️ HNSW index full ({current_count}/{MAX_ELEMENTS}). Skipping index update.")

        return {
            "id": student_id,
            "name": name,
            "grade": grade,
            "photo_path": f"data/faces/{photo_filename}" if best_frame is not None else "",
            "frames_processed": len(embeddings),
            "embedding_computed": True
        }

    except HTTPException as e:
        # Propagate intended HTTP errors (e.g., 400 problems with frames)
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enroll burst error: {str(e)}")


@app.post("/api/students/{student_id}/attach-face")
async def attach_face_to_student(student_id: str, frame: UploadFile = File(...)):
    """Attach current camera face to an existing student (update photo and embedding)."""
    # Read and decode image
    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Detect face
    bbox = face_engine.detect_face_bgr(img)
    if bbox is None:
        raise HTTPException(status_code=400, detail="No face detected")

    # Crop and align face
    face = face_engine.crop_align(img, bbox)
    if face is None:
        raise HTTPException(status_code=400, detail="Could not crop face")

    # Extract embedding
    embedding = face_engine.embed(face)
    if embedding is None:
        raise HTTPException(status_code=400, detail="Could not extract embedding")

    # Compute final embedding (average with existing one, if present)
    final_embedding = embedding

    with Session(engine) as session:
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        if student.embedding:
            try:
                old_emb = np.frombuffer(student.embedding, dtype=np.float32)

                # Solo combinamos si la nueva cara es razonablemente parecida
                sim = face_engine.cosine_similarity(old_emb, embedding)
                min_sim = SIM_THRESHOLD * 0.6  # umbral flexible, algo más bajo que el de reconocimiento

                if sim >= min_sim:
                    # Mezcla ponderada: mantenemos más peso del embedding histórico
                    alpha = 0.7  # peso del embedding antiguo
                    combined = old_emb * alpha + embedding * (1.0 - alpha)
                    combined = combined / np.linalg.norm(combined)
                    final_embedding = combined.astype(np.float32)
                else:
                    # Si la nueva cara es muy distinta (probablemente mala captura), conservamos el embedding anterior
                    final_embedding = old_emb
            except Exception:
                # Si algo falla en el promedio, seguimos con el embedding nuevo
                final_embedding = embedding

        # Save thumbnail
        photo_filename = f"{student_id}_thumb.jpg"
        photo_path = FACES_DIR / photo_filename

        thumb = cv2.resize(img, (120, 120), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(photo_path), thumb)

        # Update student record
        student.photo_path = f"data/faces/{photo_filename}"
        student.embedding = final_embedding.tobytes()
        session.add(student)
        session.commit()

    # Rebuild HNSW index to mantener consistentes los labels y embeddings
    # Esto evita que queden "vectores huérfanos" de alumnos antiguos al actualizar la cara.
    rebuild_index()

    return {
        "success": True,
        "student_id": student_id,
        "photo_url": f"/data/faces/{photo_filename}",
    }

@app.post("/api/students/{student_id}/attach-face-burst")
async def attach_face_burst_to_student(
    student_id: str,
    frames: List[UploadFile] = File(...)
):
    if len(frames) < 3 or len(frames) > 5:
        raise HTTPException(status_code=400, detail="Provide 3-5 frames")

    try:
        embeddings = []
        best_frame = None
        max_area = 0

        for frame in frames:
            contents = await frame.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            bbox = face_engine.detect_face_bgr(img)
            if bbox is None:
                continue

            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > max_area:
                max_area = area
                best_frame = img

            face = face_engine.crop_align(img, bbox)
            if face is None:
                continue

            embedding = face_engine.embed(face)
            if embedding is not None:
                embeddings.append(embedding)

        if len(embeddings) < 3:
            raise HTTPException(status_code=400, detail="Could not detect faces in enough frames")

        avg_embedding = face_engine.average_embeddings(embeddings)
        if avg_embedding is None:
            raise HTTPException(status_code=400, detail="Could not compute average embedding")

        final_embedding = avg_embedding

        with Session(engine) as session:
            student = session.get(Student, student_id)
            if not student:
                raise HTTPException(status_code=404, detail="Student not found")

            if student.embedding:
                try:
                    old_emb = np.frombuffer(student.embedding, dtype=np.float32)

                    sim = face_engine.cosine_similarity(old_emb, avg_embedding)
                    min_sim = SIM_THRESHOLD * 0.6

                    if sim >= min_sim:
                        alpha = 0.7
                        combined = old_emb * alpha + avg_embedding * (1.0 - alpha)
                        combined = combined / np.linalg.norm(combined)
                        final_embedding = combined.astype(np.float32)
                    else:
                        final_embedding = old_emb
                except Exception:
                    final_embedding = avg_embedding

            photo_filename = f"{student_id}_thumb.jpg"
            photo_path = FACES_DIR / photo_filename

            if best_frame is not None:
                thumb = cv2.resize(best_frame, (120, 120), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(photo_path), thumb)
                student.photo_path = f"data/faces/{photo_filename}"

            student.embedding = final_embedding.tobytes()
            session.add(student)
            session.commit()

        rebuild_index()

        return {
            "success": True,
            "student_id": student_id,
            "photo_url": f"/data/faces/{photo_filename}" if best_frame is not None else "",
            "frames_processed": len(embeddings),
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attach face burst error: {str(e)}")

@app.post("/api/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """Recognize face from uploaded image/frame with detection data"""
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        img_height, img_width = img.shape[:2]

        # Detect face
        bbox = face_engine.detect_face_bgr(img)
        
        # Prepare response with detection data
        detection_data = {
            "face_detected": bbox is not None,
            "bbox": None,
            "confidence": 0.0
        }
        
        if bbox is None:
            return {
                "match": False,
                "score": 0.0,
                "student": None,
                "detection": detection_data
            }

        # Normalize bbox coordinates to 0-1 range for frontend
        x1, y1, x2, y2 = bbox
        detection_data["bbox"] = {
            "x1": x1 / img_width,
            "y1": y1 / img_height,
            "x2": x2 / img_width,
            "y2": y2 / img_height
        }
        detection_data["confidence"] = 0.85  # Haar cascade doesn't provide confidence, use fixed value

        # Crop and align face
        face = face_engine.crop_align(img, bbox)
        if face is None:
            return {
                "match": False,
                "score": 0.0,
                "student": None,
                "detection": detection_data
            }

        # Extract embedding
        embedding = face_engine.embed(face)
        if embedding is None:
            return {
                "match": False,
                "score": 0.0,
                "student": None,
                "detection": detection_data
            }

        # Search in index
        if index is None or index.get_current_count() == 0:
            return {
                "match": False,
                "score": 0.0,
                "student": None,
                "detection": detection_data
            }

        labels, similarities = search_similar(embedding, k=1)

        if not labels:
            return {
                "match": False,
                "score": 0.0,
                "student": None,
                "detection": detection_data
            }

        # Get student info
        student_id = labels[0]
        with Session(engine) as session:
            student = session.get(Student, student_id)

        if not student:
            return {
                "match": False,
                "score": float(similarities[0]),
                "student": None,
                "detection": detection_data
            }

        student_response = {
            "id": student.id,
            "name": student.name,
            "grade": student.grade,
            "balance": student.balance,
            "photo_url": f"/data/faces/{Path(student.photo_path).name}" if student.photo_path else '/default-avatar.png',
            "photo_path": student.photo_path
        }

        return {
            "match": True,
            "student": student_response,
            "score": float(similarities[0]),
            "detection": detection_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition error: {str(e)}")

# Admin endpoints
@app.get("/api/students/{student_id}/transactions")
async def get_student_transactions(student_id: str):
    """Get transaction history for a specific student"""
    with Session(engine) as session:
        # Get student to verify exists
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Get transactions with product names
        transactions = session.exec(
            select(Transaction, Product)
            .join(Product, Transaction.product_id == Product.id)
            .where(Transaction.student_id == student_id)
            .order_by(Transaction.created_at.desc())
        ).all()
        
        return [
            {
                "id": t[0].id,
                "product_name": t[1].name,
                "amount": t[0].amount,
                "created_at": t[0].created_at.isoformat()
            }
            for t in transactions
        ]

@app.get("/api/students/{student_id}/suggestions")
async def get_student_suggestions(student_id: str):
    """Get product suggestions based on student's purchase history"""
    with Session(engine) as session:
        # Get student to verify exists
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Get student's purchase history
        transactions = session.exec(
            select(Transaction.product_id, Product.name, Product.price)
            .join(Product, Transaction.product_id == Product.id)
            .where(Transaction.student_id == student_id)
        ).all()
        
        if not transactions:
            # No history, suggest popular products
            popular_products = session.exec(
                select(Product.name, Product.price)
                .limit(3)
            ).all()
            return [
                {
                    "name": p.name,
                    "price": p.price,
                    "reason": "Producto popular"
                }
                for p in popular_products
            ]
        
        # Count purchases by product
        product_counts = {}
        for t in transactions:
            product_counts[t.product_id] = product_counts.get(t.product_id, 0) + 1
        
        # Get most purchased products
        most_purchased = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        
        suggestions = []
        for product_id, count in most_purchased:
            product = session.get(Product, product_id)
            if product:
                suggestions.append({
                    "name": product.name,
                    "price": product.price,
                    "reason": f"Comprado {count} veces"
                })
        
        # Add a random suggestion
        random_product = session.exec(
            select(Product)
            .where(Product.id.notin_([p[0] for p in most_purchased]))
            .limit(1)
        ).first()
        
        if random_product:
            suggestions.append({
                "name": random_product.name,
                "price": random_product.price,
                "reason": "Nuevo para probar"
            })
        
        return suggestions

@app.post("/api/students/{student_id}/add-credits")
async def add_student_credits(student_id: str, request: dict):
    """Add credits to student account"""
    amount = request.get('amount', 0)
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")
    
    with Session(engine) as session:
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        student.balance += amount
        session.add(student)
        session.commit()
        
        return {
            "success": True,
            "new_balance": student.balance,
            "added_amount": amount
        }

@app.put("/api/students/{student_id}")
async def update_student(
    student_id: str,
    name: str = Form(...),
    grade: str = Form(...),
    balance: int = Form(...),
    photo: UploadFile = File(None)
):
    """Update student information"""
    with Session(engine) as session:
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Update basic info
        student.name = name
        student.grade = grade
        student.balance = balance
        
        # Update photo if provided
        if photo and photo.filename:
            try:
                # Read and decode image
                contents = await photo.read()
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    raise HTTPException(status_code=400, detail="Invalid image data")
                
                # Save new photo
                photo_filename = f"{student_id}_thumb.jpg"
                photo_path = FACES_DIR / photo_filename
                
                # Resize to thumbnail
                thumb = cv2.resize(img, (120, 120), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(photo_path), thumb)
                
                # Update photo path
                student.photo_path = f"data/faces/{photo_filename}"
                
                # If we have a new photo, we should update the embedding too
                bbox = face_engine.detect_face_bgr(img)
                if bbox is not None:
                    face = face_engine.crop_align(img, bbox)
                    if face is not None:
                        embedding = face_engine.embed(face)
                        if embedding is not None:
                            student.embedding = embedding.tobytes()
                            
                            # Update HNSW index if student is in it
                            if student_id in index_labels:
                                label = index_labels[student_id]
                                # Remove old entry and add new one
                                # Note: HNSW doesn't support direct update, so we rebuild if needed
                                pass
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing photo: {str(e)}")
        
        session.add(student)
        session.commit()
        
        return {
            "success": True,
            "student": {
                "id": student.id,
                "name": student.name,
                "grade": student.grade,
                "balance": student.balance,
                "photo_url": f"/data/faces/{Path(student.photo_path).name}" if student.photo_path else '/default-avatar.png'
            }
        }

@app.delete("/api/students/{student_id}")
async def delete_student(student_id: str):
    """Delete a student"""
    with Session(engine) as session:
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Delete photo file if exists
        if student.photo_path:
            photo_file = Path(student.photo_path)
            if photo_file.exists():
                photo_file.unlink()
        
        # Remove from HNSW index
        if student_id in index_labels:
            del index_labels[student_id]
            # Note: HNSW doesn't support deletion, index will need rebuilding
            save_index()
        
        # Delete from database
        session.delete(student)
        session.commit()
        
        return {"success": True, "message": "Student deleted successfully"}

@app.post("/api/charge")
async def charge_student(charge: ChargeRequest):
    """Charge a product to a student's account"""
    with Session(engine) as session:
        # Get student
        student = session.get(Student, charge.student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        # Get product
        product = session.get(Product, charge.product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        # Check stock if managed
        if hasattr(product, "stock") and product.stock is not None and product.stock <= 0:
            raise HTTPException(status_code=400, detail="Product out of stock")

        # Check balance (convert to cents for comparison)
        if student.balance < product.price:
            raise HTTPException(status_code=400, detail="Insufficient balance")
        
        # Deduct balance
        student.balance -= product.price
        session.add(student)
        
        # Decrease stock
        if hasattr(product, "stock") and product.stock is not None:
            product.stock -= 1
            session.add(product)

        # Log transaction
        transaction = Transaction(
            student_id=charge.student_id,
            product_id=charge.product_id,
            amount=product.price
        )
        session.add(transaction)

        session.commit()

        return {
            "success": True,
            "student_name": student.name,
            "product_name": product.name,
            "amount": product.price,
            "new_balance": student.balance,
            "new_stock": product.stock,
        }

@app.get("/api/transactions")
async def list_transactions(
    student_id: Optional[str] = Query(None, description="Filter by student id"),
    date_from: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """List transactions with optional filters for student and date range"""
    with Session(engine) as session:
        query = (
            select(Transaction, Student, Product)
            .join(Student, Transaction.student_id == Student.id)
            .join(Product, Transaction.product_id == Product.id)
        )

        if student_id:
            query = query.where(Transaction.student_id == student_id)

        if date_from:
            try:
                dt_from = datetime.fromisoformat(date_from)
                query = query.where(Transaction.created_at >= dt_from)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format")

        if date_to:
            try:
                dt_to = datetime.fromisoformat(date_to)
                query = query.where(Transaction.created_at <= dt_to)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format")

        query = query.order_by(Transaction.created_at.desc())
        rows = session.exec(query).all()

        return [
            {
                "id": t.id,
                "student_id": s.id,
                "student_name": s.name,
                "product_id": p.id,
                "product_name": p.name,
                "amount": t.amount,
                "created_at": t.created_at.isoformat(),
            }
            for (t, s, p) in rows
        ]


@app.get("/api/analytics/summary")
async def analytics_summary(
    date_from: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Return basic analytics: top products, top students, and daily sales"""
    with Session(engine) as session:
        query = (
            select(Transaction, Student, Product)
            .join(Student, Transaction.student_id == Student.id)
            .join(Product, Transaction.product_id == Product.id)
        )

        if date_from:
            try:
                dt_from = datetime.fromisoformat(date_from)
                query = query.where(Transaction.created_at >= dt_from)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format")

        if date_to:
            try:
                dt_to = datetime.fromisoformat(date_to)
                query = query.where(Transaction.created_at <= dt_to)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format")

        rows = session.exec(query).all()

        if not rows:
            return {
                "top_products": [],
                "top_students": [],
                "daily_sales": [],
            }

        product_stats = {}
        student_stats = {}
        daily_stats = {}

        for t, s, p in rows:
            pid = p.id
            sid = s.id
            day = t.created_at.date().isoformat()

            if pid not in product_stats:
                product_stats[pid] = {
                    "product_id": pid,
                    "name": p.name,
                    "total_amount": 0,
                    "transaction_count": 0,
                }
            product_stats[pid]["total_amount"] += t.amount
            product_stats[pid]["transaction_count"] += 1

            if sid not in student_stats:
                student_stats[sid] = {
                    "student_id": sid,
                    "name": s.name,
                    "total_spent": 0,
                    "transaction_count": 0,
                }
            student_stats[sid]["total_spent"] += t.amount
            student_stats[sid]["transaction_count"] += 1

            if day not in daily_stats:
                daily_stats[day] = {
                    "date": day,
                    "total_amount": 0,
                    "transaction_count": 0,
                }
            daily_stats[day]["total_amount"] += t.amount
            daily_stats[day]["transaction_count"] += 1

        top_products = sorted(
            product_stats.values(),
            key=lambda x: x["total_amount"],
            reverse=True,
        )[:5]

        top_students = sorted(
            student_stats.values(),
            key=lambda x: x["total_spent"],
            reverse=True,
        )[:5]

        daily_sales = sorted(daily_stats.values(), key=lambda x: x["date"])

        return {
            "top_products": top_products,
            "top_students": top_students,
            "daily_sales": daily_sales,
        }

# Serve data directory for faces and other assets
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# Mount static files (absolute path to avoid CWD issues)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

import os
import json
import base64
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum

import cv2
import numpy as np
import hnswlib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, Session, create_engine, select
from PIL import Image
import uvicorn
from jose import JWTError, jwt
from passlib.context import CryptContext

from face_engine import FaceEngine
from config import SIM_THRESHOLD, FRAME_INTERVAL_MS, SEARCH_LIMIT, CURRENCY, MAX_ELEMENTS, EF_CONSTRUCTION, M_INDEX


class Role(str, Enum):
    ADMIN = "admin"
    CAJERA = "cajera"
    STOCK = "stock"
    PARENT = "parent"


class AllergyType(str, Enum):
    MANI = "mani"
    FRUTOS_SECOS = "frutos_secos"
    LECHE = "leche"
    LACTOSA = "lactosa"
    HUEVO = "huevo"
    TRIGO = "trigo"
    SOJA = "soja"
    PESCADO = "pescado"
    MARISCOS = "mariscos"
    SESAMO = "sesamo"
    GLUTEN = "gluten"
    FRUCTOSA = "fructosa"
    COLORANTES = "colorantes"
    DIABETES = "diabetes"
    VEGETARIANO = "vegetariano"
    VEGANO = "vegano"


class RestrictionMode(str, Enum):
    ALLOW_ONLY = "allow_only"
    BLOCK_LIST = "block_list"


# Database models
class PointOfSale(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    location: Optional[str] = None


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    full_name: Optional[str] = None
    role: Role = Field(default=Role.CAJERA)
    hashed_password: str
    point_of_sale_id: Optional[int] = Field(default=None, foreign_key="pointofsale.id")


class Student(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str
    grade: str
    balance: int = Field(default=0)
    photo_path: str = ""
    embedding: Optional[bytes] = None
    point_of_sale_id: Optional[int] = Field(default=None, foreign_key="pointofsale.id")


class Product(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    price: int
    stock: int = Field(default=0)
    default_min_stock: int = Field(default=20)
    allergens: Optional[str] = None


class ProductStock(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    product_id: int = Field(foreign_key="product.id")
    point_of_sale_id: int = Field(foreign_key="pointofsale.id")
    current_stock: int = Field(default=0)
    min_stock: int = Field(default=20)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Transaction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: str
    product_id: int
    amount: int
    created_at: datetime = Field(default_factory=datetime.now)
    payment_method: str = Field(default="balance")
    point_of_sale_id: Optional[int] = Field(default=None, foreign_key="pointofsale.id")
    cashier_id: Optional[int] = Field(default=None, foreign_key="user.id")
    quantity: int = Field(default=1)


class StudentGuardian(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: str = Field(foreign_key="student.id")
    parent_user_id: int = Field(foreign_key="user.id")


class StudentAllergy(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: str = Field(foreign_key="student.id")
    allergy: AllergyType
    notes: Optional[str] = None


class StudentProductRule(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: str = Field(foreign_key="student.id")
    product_id: int = Field(foreign_key="product.id")
    mode: RestrictionMode = Field(default=RestrictionMode.BLOCK_LIST)

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
    allergens: List[AllergyType] = []

class ProductUpdate(BaseModel):
    name: str
    price: int
    stock: int
    allergens: List[AllergyType] = []

class RecognitionResult(BaseModel):
    match: bool
    student: Optional[StudentResponse] = None
    score: float = 0.0

class ChargeRequest(BaseModel):
    student_id: str
    product_id: int
    quantity: int = 1
    payment_method: str = "balance"
    point_of_sale_id: Optional[int] = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[Role] = None


class UserCreate(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    role: Role = Role.CAJERA
    point_of_sale_id: Optional[int] = None


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    role: Role
    point_of_sale_id: Optional[int]


class PointOfSaleCreate(BaseModel):
    name: str
    location: Optional[str] = None


class PointOfSaleResponse(BaseModel):
    id: int
    name: str
    location: Optional[str]


class ProductStockCreate(BaseModel):
    product_id: int
    point_of_sale_id: int
    current_stock: int = 0
    min_stock: Optional[int] = None


class ProductStockUpdate(BaseModel):
    current_stock: Optional[int] = None
    min_stock: Optional[int] = None


class ProductStockResponse(BaseModel):
    id: int
    product_id: int
    product_name: str
    point_of_sale_id: int
    current_stock: int
    min_stock: int
    updated_at: datetime


class StudentAllergyRequest(BaseModel):
    allergies: List[AllergyType] = []
    notes: Optional[str] = None


class StudentProductRulesRequest(BaseModel):
    allow_product_ids: List[int] = []
    block_product_ids: List[int] = []

# Global variables
app = FastAPI(title="Cantina Face Recognition System")
face_engine = FaceEngine()

# Security / auth settings
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-super-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

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


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except ValueError:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_user_by_email(session: Session, email: str) -> Optional[User]:
    return session.exec(select(User).where(User.email == email)).first()


def authenticate_user(session: Session, email: str, password: str) -> Optional[User]:
    user = get_user_by_email(session, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        role: str = payload.get("role")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email, role=Role(role) if role else None)
    except (JWTError, ValueError):
        raise credentials_exception

    with Session(engine) as session:
        user = get_user_by_email(session, token_data.email)
        if user is None:
            raise credentials_exception
        return user


def require_roles(*roles: Role):
    async def _dependency(current_user: User = Depends(get_current_user)) -> User:
        if roles and current_user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user

    return _dependency


def ensure_default_admin():
    admin_email = os.getenv("ADMIN_EMAIL")
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_email or not admin_password:
        return

    with Session(engine) as session:
        if get_user_by_email(session, admin_email):
            return

        admin_user = User(
            email=admin_email,
            full_name="Administrador",
            role=Role.ADMIN,
            hashed_password=get_password_hash(admin_password),
        )
        session.add(admin_user)
        session.commit()


def ensure_column(table_name: str, column_name: str, column_type: str):
    table_quoted = f'"{table_name}"'
    with engine.connect() as conn:
        result = conn.exec_driver_sql(f"PRAGMA table_info({table_quoted})").all()
        if not any(row[1] == column_name for row in result):
            conn.exec_driver_sql(
                f"ALTER TABLE {table_quoted} ADD COLUMN {column_name} {column_type}"
            )


def ensure_product_stock_column():
    """Ensure legacy columns exist on Product table (SQLite migrations)."""
    ensure_column("product", "stock", "INTEGER NOT NULL DEFAULT 0")
    ensure_column("product", "default_min_stock", "INTEGER NOT NULL DEFAULT 20")
    ensure_column("product", "allergens", "TEXT")


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
    ensure_column("student", "point_of_sale_id", "INTEGER")
    ensure_column("transaction", "payment_method", "TEXT DEFAULT 'balance'")
    ensure_column("transaction", "point_of_sale_id", "INTEGER")
    ensure_column("transaction", "cashier_id", "INTEGER")
    ensure_column("transaction", "quantity", "INTEGER NOT NULL DEFAULT 1")
    init_hnsw_index()
    ensure_default_admin()

    # Rebuild index if empty but we have students in DB
    with Session(engine) as session:
        student_count = session.exec(select(Student)).all()

    if index.get_current_count() == 0 and len(student_count) > 0:
        rebuild_index()


def get_or_create_product_stock(session: Session, product_id: int, point_of_sale_id: int) -> ProductStock:
    stock = session.exec(
        select(ProductStock).where(
            ProductStock.product_id == product_id,
            ProductStock.point_of_sale_id == point_of_sale_id,
        )
    ).first()

    if stock:
        return stock

    product = session.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    stock = ProductStock(
        product_id=product_id,
        point_of_sale_id=point_of_sale_id,
        current_stock=product.stock,
        min_stock=product.default_min_stock,
    )
    session.add(stock)
    session.commit()
    session.refresh(stock)
    return stock


def serialize_product_stock(stock: ProductStock, product_name: str) -> ProductStockResponse:
    return ProductStockResponse(
        id=stock.id,
        product_id=stock.product_id,
        product_name=product_name,
        point_of_sale_id=stock.point_of_sale_id,
        current_stock=stock.current_stock,
        min_stock=stock.min_stock,
        updated_at=stock.updated_at,
    )


def parse_product_allergens(product: Product) -> List[AllergyType]:
    if not product.allergens:
        return []
    try:
        data = json.loads(product.allergens)
        normalized = []
        for item in data:
            try:
                normalized.append(AllergyType(item))
            except ValueError:
                continue
        return normalized
    except Exception:
        return []


def serialize_product(product: Product) -> Dict[str, Optional[str]]:
    allergens = [a.value if isinstance(a, AllergyType) else a for a in parse_product_allergens(product)]
    return {
        "id": product.id,
        "name": product.name,
        "price": product.price,
        "stock": product.stock,
        "allergens": allergens,
    }


def is_student_guardian(session: Session, parent_id: int, student_id: str) -> bool:
    return session.exec(
        select(StudentGuardian).where(
            StudentGuardian.student_id == student_id,
            StudentGuardian.parent_user_id == parent_id,
        )
    ).first() is not None


def ensure_student_access(session: Session, current_user: User, student_id: str):
    if current_user.role == Role.ADMIN:
        return
    if current_user.role == Role.PARENT and is_student_guardian(session, current_user.id, student_id):
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for this student")


def get_student_restrictions(session: Session, student_id: str) -> Tuple[Set[AllergyType], Set[int], Set[int]]:
    allergies = {
        entry.allergy
        for entry in session.exec(select(StudentAllergy).where(StudentAllergy.student_id == student_id)).all()
    }

    rules = session.exec(select(StudentProductRule).where(StudentProductRule.student_id == student_id)).all()
    allow = {rule.product_id for rule in rules if rule.mode == RestrictionMode.ALLOW_ONLY}
    block = {rule.product_id for rule in rules if rule.mode == RestrictionMode.BLOCK_LIST}
    return allergies, allow, block


def check_product_restrictions(
    product: Product,
    allergies: Set[AllergyType],
    allow: Set[int],
    block: Set[int]
) -> Tuple[bool, Optional[str]]:
    product_allergens = parse_product_allergens(product)
    for allergen in product_allergens:
        if allergen in allergies:
            return False, f"Producto restringido por alergia: {allergen.value}"

    if allow and product.id not in allow:
        return False, "Producto no autorizado para este alumno"

    if product.id in block:
        return False, "Producto bloqueado por los padres"

    return True, None


@app.post("/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as session:
        user = authenticate_user(session, form_data.username, form_data.password)
        if not user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect email or password")

        access_token = create_access_token({"sub": user.email, "role": user.role.value})
        return {"access_token": access_token, "token_type": "bearer"}


@app.post("/auth/users", response_model=UserResponse)
async def create_user(user_in: UserCreate, current_user: User = Depends(require_roles(Role.ADMIN))):
    with Session(engine) as session:
        if get_user_by_email(session, user_in.email):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

        user = User(
            email=user_in.email,
            full_name=user_in.full_name,
            role=user_in.role,
            point_of_sale_id=user_in.point_of_sale_id,
            hashed_password=get_password_hash(user_in.password),
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        return UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            point_of_sale_id=user.point_of_sale_id,
        )


@app.get("/auth/me", response_model=UserResponse)
async def read_current_user(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        point_of_sale_id=current_user.point_of_sale_id,
    )


@app.post("/api/points-of-sale", response_model=PointOfSaleResponse)
async def create_point_of_sale(
    payload: PointOfSaleCreate,
    current_user: User = Depends(require_roles(Role.ADMIN))
):
    with Session(engine) as session:
        pos = PointOfSale(name=payload.name, location=payload.location)
        session.add(pos)
        session.commit()
        session.refresh(pos)
        return PointOfSaleResponse(id=pos.id, name=pos.name, location=pos.location)


@app.get("/api/points-of-sale", response_model=List[PointOfSaleResponse])
async def list_points_of_sale(current_user: User = Depends(require_roles(Role.ADMIN, Role.STOCK))):
    with Session(engine) as session:
        points = session.exec(select(PointOfSale)).all()
        return [PointOfSaleResponse(id=p.id, name=p.name, location=p.location) for p in points]


@app.post("/api/stock", response_model=ProductStockResponse)
async def create_product_stock(
    payload: ProductStockCreate,
    current_user: User = Depends(require_roles(Role.ADMIN, Role.STOCK))
):
    with Session(engine) as session:
        product = session.get(Product, payload.product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        stock = ProductStock(
            product_id=payload.product_id,
            point_of_sale_id=payload.point_of_sale_id,
            current_stock=payload.current_stock,
            min_stock=payload.min_stock or product.default_min_stock,
        )
        session.add(stock)
        session.commit()
        session.refresh(stock)
        return serialize_product_stock(stock, product.name)


@app.get("/api/stock/{point_of_sale_id}", response_model=List[ProductStockResponse])
async def list_stock_by_pos(point_of_sale_id: int, current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        query = (
            select(ProductStock, Product)
            .join(Product, Product.id == ProductStock.product_id)
            .where(ProductStock.point_of_sale_id == point_of_sale_id)
        )
        rows = session.exec(query).all()
        return [serialize_product_stock(stock, prod.name) for stock, prod in rows]


@app.patch("/api/stock/{stock_id}", response_model=ProductStockResponse)
async def update_product_stock(
    stock_id: int,
    payload: ProductStockUpdate,
    current_user: User = Depends(require_roles(Role.ADMIN, Role.STOCK))
):
    with Session(engine) as session:
        stock = session.get(ProductStock, stock_id)
        if not stock:
            raise HTTPException(status_code=404, detail="Stock record not found")

        if payload.current_stock is not None:
            stock.current_stock = payload.current_stock
        if payload.min_stock is not None:
            stock.min_stock = payload.min_stock
        stock.updated_at = datetime.utcnow()
        session.add(stock)
        session.commit()
        session.refresh(stock)

        product = session.get(Product, stock.product_id)
        product_name = product.name if product else "Producto"
        return serialize_product_stock(stock, product_name)

@app.post("/api/products")
async def create_product(product: ProductCreate, current_user: User = Depends(require_roles(Role.ADMIN, Role.STOCK))):
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
            allergens=json.dumps([a.value for a in product.allergens]) if product.allergens else None,
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


@app.get("/api/parents/students", response_model=List[StudentResponse])
async def list_parent_students(current_user: User = Depends(require_roles(Role.PARENT))):
    with Session(engine) as session:
        guardian_links = session.exec(
            select(StudentGuardian).where(StudentGuardian.parent_user_id == current_user.id)
        ).all()

        student_ids = [link.student_id for link in guardian_links]
        if not student_ids:
            return []

        students = session.exec(select(Student).where(Student.id.in_(student_ids))).all()
        return [
            StudentResponse(
                id=s.id,
                name=s.name,
                grade=s.grade,
                balance=s.balance,
                photo_path=f"/data/faces/{Path(s.photo_path).name}" if s.photo_path else '/default-avatar.png'
            )
            for s in students
        ]


@app.get("/api/students/{student_id}/allergies")
async def get_student_allergies(student_id: str, current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        ensure_student_access(session, current_user, student_id)

        entries = session.exec(select(StudentAllergy).where(StudentAllergy.student_id == student_id)).all()
        return {
            "student_id": student_id,
            "allergies": [entry.allergy.value for entry in entries],
            "notes": entries[0].notes if entries else None,
        }


@app.put("/api/students/{student_id}/allergies")
async def update_student_allergies(
    student_id: str,
    payload: StudentAllergyRequest,
    current_user: User = Depends(get_current_user)
):
    with Session(engine) as session:
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        ensure_student_access(session, current_user, student_id)

        existing = session.exec(select(StudentAllergy).where(StudentAllergy.student_id == student_id)).all()
        for entry in existing:
            session.delete(entry)

        for allergy in payload.allergies:
            session.add(StudentAllergy(student_id=student_id, allergy=allergy, notes=payload.notes))

        session.commit()

        return {
            "student_id": student_id,
            "allergies": [a.value for a in payload.allergies],
            "notes": payload.notes,
        }


@app.put("/api/students/{student_id}/product-rules")
async def update_student_product_rules(
    student_id: str,
    payload: StudentProductRulesRequest,
    current_user: User = Depends(get_current_user)
):
    with Session(engine) as session:
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        ensure_student_access(session, current_user, student_id)

        existing = session.exec(select(StudentProductRule).where(StudentProductRule.student_id == student_id)).all()
        for entry in existing:
            session.delete(entry)

        product_ids = set(payload.allow_product_ids + payload.block_product_ids)
        if product_ids:
            valid_ids = {
                prod.id
                for prod in session.exec(select(Product).where(Product.id.in_(product_ids))).all()
            }
        else:
            valid_ids = set()

        for product_id in payload.allow_product_ids:
            if product_id in valid_ids or not valid_ids:
                session.add(
                    StudentProductRule(
                        student_id=student_id,
                        product_id=product_id,
                        mode=RestrictionMode.ALLOW_ONLY,
                    )
                )

        for product_id in payload.block_product_ids:
            if product_id in valid_ids or not valid_ids:
                session.add(
                    StudentProductRule(
                        student_id=student_id,
                        product_id=product_id,
                        mode=RestrictionMode.BLOCK_LIST,
                    )
                )

        session.commit()

        return {
            "student_id": student_id,
            "allow_product_ids": payload.allow_product_ids,
            "block_product_ids": payload.block_product_ids,
        }


@app.get("/api/students/{student_id}/catalog")
async def get_student_catalog(
    student_id: str,
    point_of_sale_id: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):
    with Session(engine) as session:
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        ensure_student_access(session, current_user, student_id)

        allergies, allow, block = get_student_restrictions(session, student_id)

        products_query = select(Product)
        products = session.exec(products_query).all()

        stock_map: Dict[int, int] = {}
        if point_of_sale_id:
            stock_rows = session.exec(
                select(ProductStock).where(ProductStock.point_of_sale_id == point_of_sale_id)
            ).all()
            stock_map = {row.product_id: row.current_stock for row in stock_rows}

        catalog = []
        for product in products:
            allowed, reason = check_product_restrictions(product, allergies, allow, block)
            item = serialize_product(product)
            item.update(
                {
                    "allowed": allowed,
                    "restriction_reason": reason,
                    "stock_at_pos": stock_map.get(product.id) if point_of_sale_id else None,
                }
            )
            catalog.append(item)

        return {
            "student_id": student_id,
            "catalog": catalog,
        }


@app.get("/api/students/{student_id}")
async def get_student_detail(student_id: str):
    with Session(engine) as session:
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        return {
            "id": student.id,
            "name": student.name,
            "grade": student.grade,
            "balance": student.balance,
            "photo_url": f"/data/faces/{Path(student.photo_path).name}" if student.photo_path else '/default-avatar.png'
        }

@app.put("/api/products/{product_id}")
async def update_product(product_id: int, product: ProductUpdate, current_user: User = Depends(require_roles(Role.ADMIN, Role.STOCK))):
    """Update an existing product (name, price, stock)."""
    with Session(engine) as session:
        db_product = session.get(Product, product_id)
        if not db_product:
            raise HTTPException(status_code=404, detail="Product not found")

        db_product.name = product.name
        db_product.price = product.price
        db_product.stock = product.stock
        db_product.allergens = json.dumps([a.value for a in product.allergens]) if product.allergens else None
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
async def seed_products(current_user: User = Depends(require_roles(Role.ADMIN))):
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
async def charge_student(
    charge: ChargeRequest,
    current_user: User = Depends(require_roles(Role.ADMIN, Role.CAJERA))
):
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

        quantity = max(1, charge.quantity)
        total_amount = product.price * quantity

        pos_id = (
            charge.point_of_sale_id
            or current_user.point_of_sale_id
            or student.point_of_sale_id
        )
        if pos_id is None:
            raise HTTPException(status_code=400, detail="Point of sale must be specified")

        stock_record = get_or_create_product_stock(session, product.id, pos_id)
        if stock_record.current_stock < quantity:
            raise HTTPException(status_code=400, detail="Insufficient stock at this point of sale")

        if charge.payment_method == "balance":
            if student.balance < total_amount:
                raise HTTPException(status_code=400, detail="Insufficient balance")
            student.balance -= total_amount
        elif charge.payment_method == "cash":
            # cash doesn't change student balance
            pass
        else:
            raise HTTPException(status_code=400, detail="Unsupported payment method")

        session.add(student)

        # Decrease stock for point of sale
        stock_record.current_stock -= quantity
        stock_record.updated_at = datetime.utcnow()
        session.add(stock_record)

        # Log transaction
        transaction = Transaction(
            student_id=charge.student_id,
            product_id=charge.product_id,
            amount=total_amount,
            payment_method=charge.payment_method,
            point_of_sale_id=pos_id,
            cashier_id=current_user.id,
            quantity=quantity,
        )
        session.add(transaction)

        session.commit()

        low_stock = stock_record.current_stock <= stock_record.min_stock

        return {
            "success": True,
            "student_name": student.name,
            "product_name": product.name,
            "amount": total_amount,
            "quantity": quantity,
            "new_balance": student.balance,
            "point_of_sale_id": pos_id,
            "stock_remaining": stock_record.current_stock,
            "low_stock": low_stock,
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
                "quantity": t.quantity,
                "payment_method": t.payment_method,
                "point_of_sale_id": t.point_of_sale_id,
                "cashier_id": t.cashier_id,
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


@app.get("/sales")
async def sales_page():
    """Dedicated sales page route"""
    return FileResponse(STATIC_DIR / "sales.html")


# Mount static files (absolute path to avoid CWD issues)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

import os
import json
import base64
import shutil
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple, Any
from datetime import datetime, timedelta, date, timezone
from enum import Enum
from uuid import uuid4
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import hnswlib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, Session, create_engine, select, delete
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


class ProductCategory(str, Enum):
    GASEOSA = "gaseosa"
    CHOCOLATE = "chocolate"
    DULCE = "dulce"
    HELADO = "helado"
    SNACK = "snack"
    OTRO = "otro"


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


class BalanceAllocationMode(str, Enum):
    EQUAL = "equal"
    CUSTOM = "custom"


class BalanceTopUpStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ScheduledOrderStatus(str, Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    CANCELLED = "cancelled"


class ParentStudentLinkStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# Database models
DEFAULT_POS_ID = 1


class PointOfSale(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    location: Optional[str] = None


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dni: Optional[str] = Field(default=None, index=True)
    phone: Optional[str] = None
    role: Role = Field(default=Role.CAJERA)
    hashed_password: str
    point_of_sale_id: Optional[int] = Field(default=None, foreign_key="pointofsale.id")
    is_active: bool = Field(default=True)


class Student(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str
    grade: str
    balance: int = Field(default=0)
    photo_path: str = ""
    embedding: Optional[bytes] = None
    point_of_sale_id: Optional[int] = Field(default=None, foreign_key="pointofsale.id")


class BalanceAdjustment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: str = Field(foreign_key="student.id")
    amount: int
    pay_from_balance: bool = Field(default=True)
    point_of_sale_id: Optional[int] = Field(default=None, foreign_key="pointofsale.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Product(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    price: int
    stock: int = Field(default=0)
    default_min_stock: int = Field(default=20)
    allergens: Optional[str] = None
    category: Optional[ProductCategory] = None


class ProductStock(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    product_id: int = Field(foreign_key="product.id")
    point_of_sale_id: int = Field(foreign_key="pointofsale.id")
    current_stock: int = Field(default=0)
    min_stock: int = Field(default=10)
    reserved_stock: int = Field(default=0)
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


class ParentStudentLinkRequest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    parent_id: int = Field(foreign_key="user.id")
    student_identifier: Optional[str] = None
    student_name: str
    student_grade: Optional[str] = None
    notes: Optional[str] = None
    status: ParentStudentLinkStatus = Field(default=ParentStudentLinkStatus.PENDING)
    admin_notes: Optional[str] = None
    student_id: Optional[str] = Field(default=None, foreign_key="student.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None


class ProductMinStockUpdate(BaseModel):
    min_stock: int = Field(ge=0)


class StockAlertResponse(BaseModel):
    product_id: int
    product_name: str
    current_stock: int
    min_stock: int
    status: str
    point_of_sale_id: Optional[int] = None


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


class BalanceTopUpRequest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    parent_id: int = Field(foreign_key="user.id")
    total_amount: int
    allocation_mode: BalanceAllocationMode = Field(default=BalanceAllocationMode.EQUAL)
    allocations: Optional[str] = None  # JSON payload {student_id: amount}
    payment_reference: Optional[str] = None
    status: BalanceTopUpStatus = Field(default=BalanceTopUpStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None


class ScheduledOrder(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: str = Field(foreign_key="student.id")
    parent_id: int = Field(foreign_key="user.id")
    scheduled_for: date
    status: ScheduledOrderStatus = Field(default=ScheduledOrderStatus.PENDING)
    notes: Optional[str] = None
    pay_from_balance: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ScheduledOrderItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    order_id: int = Field(foreign_key="scheduledorder.id")
    product_id: int = Field(foreign_key="product.id")
    quantity: int = Field(default=1)


class DailyMenu(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    menu_date: date = Field(index=True, unique=True)
    title: Optional[str] = None
    description: Optional[str] = None
    created_by: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DailyMenuItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    menu_id: int = Field(foreign_key="dailymenu.id")
    product_id: Optional[int] = Field(default=None, foreign_key="product.id")
    name: str
    meal_type: Optional[str] = None


class StudentMenuSelection(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    parent_id: int = Field(foreign_key="user.id")
    student_id: str = Field(foreign_key="student.id")
    menu_item_id: int = Field(foreign_key="dailymenuitem.id")
    menu_date: date
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

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
    default_min_stock: int = 20
    allergens: List[AllergyType] = []


class ProductUpdate(BaseModel):
    name: str
    price: int
    stock: int
    default_min_stock: int = 20
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
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dni: Optional[str] = None
    phone: Optional[str] = None
    role: Role = Role.CAJERA
    point_of_sale_id: Optional[int] = None
    is_active: bool = True


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    dni: Optional[str]
    phone: Optional[str]
    role: Role
    point_of_sale_id: Optional[int]
    is_active: bool


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dni: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[Role] = None
    point_of_sale_id: Optional[int] = None
    is_active: Optional[bool] = None


class UserPasswordReset(BaseModel):
    new_password: str


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


class ParentRegisterRequest(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str
    dni: Optional[str] = None
    phone: Optional[str] = None


class ParentStudentAssignRequest(BaseModel):
    student_ids: List[str]


class ParentStudentLinkRequestCreate(BaseModel):
    student_identifier: Optional[str] = None
    student_name: str
    student_grade: Optional[str] = None
    notes: Optional[str] = None


class ParentStudentLinkRequestResponse(BaseModel):
    id: int
    parent_id: int
    student_identifier: Optional[str]
    student_name: str
    student_grade: Optional[str]
    notes: Optional[str]
    status: ParentStudentLinkStatus
    admin_notes: Optional[str]
    student_id: Optional[str]
    created_at: datetime
    processed_at: Optional[datetime]


class ParentStudentLinkDecision(BaseModel):
    student_id: Optional[str] = None
    admin_notes: Optional[str] = None


class BalanceTopUpCreate(BaseModel):
    total_amount: int
    allocation_mode: BalanceAllocationMode = BalanceAllocationMode.EQUAL
    per_student_amounts: Optional[Dict[str, int]] = None
    payment_reference: Optional[str] = None


class BalanceTopUpResponse(BaseModel):
    id: int
    parent_id: int
    total_amount: int
    allocation_mode: BalanceAllocationMode
    allocations: Dict[str, int]
    payment_reference: Optional[str]
    status: BalanceTopUpStatus
    created_at: datetime
    processed_at: Optional[datetime]


class BalanceTopUpDecision(BaseModel):
    payment_reference: Optional[str] = None


class ScheduledOrderItemPayload(BaseModel):
    product_id: int
    quantity: int = 1


class ScheduledOrderCreate(BaseModel):
    student_id: str
    scheduled_for: date
    items: List[ScheduledOrderItemPayload]
    notes: Optional[str] = None
    pay_from_balance: bool = True


class ScheduledOrderItemResponse(BaseModel):
    id: int
    product_id: int
    product_name: Optional[str]
    quantity: int


class ScheduledOrderResponse(BaseModel):
    id: int
    student_id: str
    parent_id: int
    scheduled_for: date
    status: ScheduledOrderStatus
    notes: Optional[str]
    pay_from_balance: bool
    point_of_sale_id: Optional[int] = None
    created_at: datetime
    items: List[ScheduledOrderItemResponse]


class DailyMenuItemInput(BaseModel):
    product_id: Optional[int] = None
    name: str
    meal_type: Optional[str] = None


class DailyMenuCreate(BaseModel):
    menu_date: date
    title: Optional[str] = None
    description: Optional[str] = None
    items: List[DailyMenuItemInput] = []


class DailyMenuItemResponse(BaseModel):
    id: int
    product_id: Optional[int]
    name: str
    meal_type: Optional[str]


class DailyMenuResponse(BaseModel):
    id: int
    menu_date: date
    title: Optional[str]
    description: Optional[str]
    items: List[DailyMenuItemResponse]


class StudentMenuSelectionCreate(BaseModel):
    menu_item_id: int
    student_id: str
    notes: Optional[str] = None


class StudentMenuSelectionResponse(BaseModel):
    id: int
    menu_item_id: int
    student_id: str
    parent_id: int
    menu_date: date
    notes: Optional[str]
    created_at: datetime

# Global variables
LOCAL_TIMEZONE = os.getenv("LOCAL_TIMEZONE", "America/Asuncion")
LOCAL_TZ = ZoneInfo(LOCAL_TIMEZONE)


def localize_datetime(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(LOCAL_TZ)


def localize_iso(value: Optional[datetime]) -> Optional[str]:
    localized = localize_datetime(value)
    return localized.isoformat() if localized else None


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
PARENTS_DIR = STATIC_DIR / "parents"

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


def ensure_column_exists(table: str, column: str, definition: str) -> None:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            if column not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    except sqlite3.OperationalError as exc:
        print(f"[ensure_column_exists] warning: {exc}")


def ensure_default_admin(email: str, password: str) -> None:
    try:
        with Session(engine) as session:
            if session.exec(select(User).where(User.email == email)).first():
                return
            hashed = get_password_hash(password)
            admin_user = User(
                email=email,
                full_name="Administrador",
                role=Role.ADMIN,
                hashed_password=hashed,
                is_active=True,
            )
            session.add(admin_user)
            session.commit()
            print("[ensure_default_admin] admin@siloe.com.py creado")
    except Exception as exc:
        print(f"[ensure_default_admin] warning: {exc}")


def initialize_legacy_data():
    ensure_column_exists("user", "is_active", "INTEGER DEFAULT 1")
    ensure_default_admin("admin@siloe.com.py", "admin321")


initialize_legacy_data()


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
    ensure_column("productstock", "reserved_stock", "INTEGER NOT NULL DEFAULT 0")
    ensure_column("scheduledorder", "pay_from_balance", "INTEGER NOT NULL DEFAULT 1")
    ensure_column("scheduledorder", "point_of_sale_id", "INTEGER")
    ensure_column("balanceadjustment", "pay_from_balance", "INTEGER NOT NULL DEFAULT 1")
    ensure_column("balanceadjustment", "point_of_sale_id", "INTEGER")
    ensure_column("user", "first_name", "TEXT")
    ensure_column("user", "last_name", "TEXT")
    ensure_column("user", "dni", "TEXT")
    ensure_column("user", "phone", "TEXT")
    ensure_column("product", "category", "TEXT")
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


def sync_default_pos_stock(session: Session, product: Product):
    """Ensure the default POS stock mirrors the product's stock."""
    if not product or not product.id:
        return

    pos_id = DEFAULT_POS_ID
    if not pos_id:
        return

    stock = session.exec(
        select(ProductStock).where(
            ProductStock.product_id == product.id,
            ProductStock.point_of_sale_id == pos_id,
        )
    ).first()

    if stock:
        stock.current_stock = product.stock
        stock.min_stock = product.default_min_stock
        stock.updated_at = datetime.utcnow()
        session.add(stock)
    else:
        new_stock = ProductStock(
            product_id=product.id,
            point_of_sale_id=pos_id,
            current_stock=product.stock,
            min_stock=product.default_min_stock,
        )
        session.add(new_stock)


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
        "default_min_stock": product.default_min_stock,
        "allergens": allergens,
        "category": product.category.value if isinstance(product.category, ProductCategory) else product.category,
    }


def serialize_user(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        first_name=user.first_name,
        last_name=user.last_name,
        dni=user.dni,
        phone=user.phone,
        role=user.role,
        point_of_sale_id=user.point_of_sale_id,
        is_active=user.is_active,
    )


def serialize_link_request(record: ParentStudentLinkRequest) -> ParentStudentLinkRequestResponse:
    return ParentStudentLinkRequestResponse(
        id=record.id,
        parent_id=record.parent_id,
        student_identifier=record.student_identifier,
        student_name=record.student_name,
        student_grade=record.student_grade,
        notes=record.notes,
        status=record.status,
        admin_notes=record.admin_notes,
        student_id=record.student_id,
        created_at=localize_datetime(record.created_at),
        processed_at=localize_datetime(record.processed_at),
    )


def get_stock_status(current: int, minimum: int) -> str:
    if minimum <= 0:
        return "ok"
    if current <= max(0, int(minimum * 0.25)):
        return "critical"
    if current < minimum:
        return "low"
    return "ok"


def is_student_guardian(session: Session, parent_id: int, student_id: str) -> bool:
    return session.exec(
        select(StudentGuardian).where(
            StudentGuardian.student_id == student_id,
            StudentGuardian.parent_user_id == parent_id,
        )
    ).first() is not None


def link_parent_to_student(session: Session, parent_id: int, student_id: str):
    if is_student_guardian(session, parent_id, student_id):
        return
    session.add(StudentGuardian(student_id=student_id, parent_user_id=parent_id))


def get_parent_student_ids(session: Session, parent_id: int) -> List[str]:
    links = session.exec(
        select(StudentGuardian.student_id).where(StudentGuardian.parent_user_id == parent_id)
    ).all()
    return [link[0] if isinstance(link, tuple) else link for link in links]


def ensure_parent_student(session: Session, parent_id: int, student_id: str):
    if not is_student_guardian(session, parent_id, student_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Student not linked to parent")


def ensure_student_access(session: Session, current_user: User, student_id: str):
    if current_user.role in (Role.ADMIN, Role.CAJERA, Role.STOCK):
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


def serialize_topup_request(record: BalanceTopUpRequest) -> BalanceTopUpResponse:
    allocations = {}
    if record.allocations:
        try:
            allocations = json.loads(record.allocations)
        except json.JSONDecodeError:
            allocations = {}
    return BalanceTopUpResponse(
        id=record.id,
        parent_id=record.parent_id,
        total_amount=record.total_amount,
        allocation_mode=record.allocation_mode,
        allocations=allocations,
        payment_reference=record.payment_reference,
        status=record.status,
        created_at=record.created_at,
        processed_at=record.processed_at,
    )


def serialize_scheduled_order(order: ScheduledOrder, items: List[ScheduledOrderItem], products: Dict[int, Product]) -> ScheduledOrderResponse:
    item_payloads = []
    for item in items:
        product = products.get(item.product_id)
        item_payloads.append(
            ScheduledOrderItemResponse(
                id=item.id,
                product_id=item.product_id,
                product_name=product.name if product else None,
                quantity=item.quantity,
            )
        )
    return ScheduledOrderResponse(
        id=order.id,
        student_id=order.student_id,
        parent_id=order.parent_id,
        scheduled_for=order.scheduled_for,
        status=order.status,
        notes=order.notes,
        pay_from_balance=order.pay_from_balance,
        created_at=localize_datetime(order.created_at),
        items=item_payloads,
    )


def serialize_daily_menu(menu: DailyMenu, items: List[DailyMenuItem]) -> DailyMenuResponse:
    return DailyMenuResponse(
        id=menu.id,
        menu_date=menu.menu_date,
        title=menu.title,
        description=menu.description,
        items=[
            DailyMenuItemResponse(
                id=item.id,
                product_id=item.product_id,
                name=item.name,
                meal_type=item.meal_type,
            )
            for item in items
        ],
    )


def serialize_menu_selection(selection: StudentMenuSelection) -> StudentMenuSelectionResponse:
    return StudentMenuSelectionResponse(
        id=selection.id,
        menu_item_id=selection.menu_item_id,
        student_id=selection.student_id,
        parent_id=selection.parent_id,
        menu_date=selection.menu_date,
        notes=selection.notes,
        created_at=localize_datetime(selection.created_at),
    )


def compute_topup_allocations(
    session: Session,
    parent_id: int,
    total_amount: int,
    mode: BalanceAllocationMode,
    per_student_amounts: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    student_ids = get_parent_student_ids(session, parent_id)
    if not student_ids:
        raise HTTPException(status_code=400, detail="Parent has no students assigned")
    if total_amount <= 0:
        raise HTTPException(status_code=400, detail="Total amount must be positive")

    allocations: Dict[str, int] = {}

    if mode == BalanceAllocationMode.EQUAL:
        base_amount = total_amount // len(student_ids)
        remainder = total_amount % len(student_ids)
        for idx, student_id in enumerate(student_ids):
            allocations[student_id] = base_amount + (1 if idx < remainder else 0)
    else:
        if not per_student_amounts:
            raise HTTPException(status_code=400, detail="Custom allocation requires per_student_amounts")
        invalid_ids = [sid for sid in per_student_amounts.keys() if sid not in student_ids]
        if invalid_ids:
            raise HTTPException(status_code=400, detail=f"Invalid student ids: {', '.join(invalid_ids)}")
        allocations = {sid: per_student_amounts.get(sid, 0) for sid in student_ids}
        if sum(allocations.values()) != total_amount:
            raise HTTPException(status_code=400, detail="Sum of custom allocations must equal total amount")

    return allocations


def process_link_request(
    session: Session,
    request: ParentStudentLinkRequest,
    status: ParentStudentLinkStatus,
    student_id: Optional[str] = None,
    admin_notes: Optional[str] = None,
):
    if request.status != ParentStudentLinkStatus.PENDING:
        raise HTTPException(status_code=400, detail="Request already processed")

    if status == ParentStudentLinkStatus.APPROVED:
        if not student_id:
            raise HTTPException(status_code=400, detail="student_id is required to approve a request")
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        link_parent_to_student(session, request.parent_id, student_id)
        request.student_id = student_id

    request.status = status
    request.admin_notes = admin_notes
    request.processed_at = datetime.utcnow()
    session.add(request)


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
            first_name=user_in.first_name,
            last_name=user_in.last_name,
            dni=user_in.dni,
            phone=user_in.phone,
            role=user_in.role,
            point_of_sale_id=user_in.point_of_sale_id,
            is_active=user_in.is_active,
            hashed_password=get_password_hash(user_in.password),
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        return serialize_user(user)


@app.get("/auth/users", response_model=List[UserResponse])
async def list_users(current_user: User = Depends(require_roles(Role.ADMIN))):
    with Session(engine) as session:
        users = session.exec(select(User)).all()
        return [serialize_user(u) for u in users]


@app.put("/auth/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, payload: UserUpdate, current_user: User = Depends(require_roles(Role.ADMIN))):
    with Session(engine) as session:
        user = session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if payload.full_name is not None:
            user.full_name = payload.full_name
        if payload.first_name is not None:
            user.first_name = payload.first_name
        if payload.last_name is not None:
            user.last_name = payload.last_name
        if payload.dni is not None:
            user.dni = payload.dni
        if payload.phone is not None:
            user.phone = payload.phone
        if payload.role is not None:
            user.role = payload.role
        if payload.point_of_sale_id is not None:
            user.point_of_sale_id = payload.point_of_sale_id
        if payload.is_active is not None:
            user.is_active = payload.is_active

        session.add(user)
        session.commit()
        session.refresh(user)

        return serialize_user(user)


@app.post("/auth/users/{user_id}/reset-password")
async def reset_user_password(user_id: int, payload: UserPasswordReset, current_user: User = Depends(require_roles(Role.ADMIN))):
    with Session(engine) as session:
        user = session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.hashed_password = get_password_hash(payload.new_password)
        session.add(user)
        session.commit()

        return {"message": "Password reset successfully", "user_id": user_id}


@app.get("/auth/me", response_model=UserResponse)
async def read_current_user(current_user: User = Depends(get_current_user)):
    return serialize_user(current_user)


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


@app.get("/api/stock/alerts", response_model=List[StockAlertResponse])
async def stock_alerts(current_user: User = Depends(require_roles(Role.ADMIN, Role.STOCK))):
    with Session(engine) as session:
        query = select(ProductStock, Product).join(Product, Product.id == ProductStock.product_id)
        rows = session.exec(query).all()

        alerts: List[StockAlertResponse] = []
        for stock, product in rows:
            status = get_stock_status(stock.current_stock, stock.min_stock or product.default_min_stock)
            alerts.append(
                StockAlertResponse(
                    product_id=product.id,
                    product_name=product.name,
                    current_stock=stock.current_stock,
                    min_stock=stock.min_stock,
                    status=status,
                    point_of_sale_id=stock.point_of_sale_id,
                )
            )

        return alerts


@app.put("/api/products/{product_id}/min-stock", response_model=Dict[str, Any])
async def update_product_min_stock(
    product_id: int,
    payload: ProductMinStockUpdate,
    current_user: User = Depends(require_roles(Role.ADMIN, Role.STOCK))
):
    with Session(engine) as session:
        product = session.get(Product, product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        product.default_min_stock = payload.min_stock
        session.add(product)
        session.commit()
        session.refresh(product)

        return {
            "product_id": product.id,
            "default_min_stock": product.default_min_stock,
            "message": "Minimum stock updated",
        }


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
            default_min_stock=product.default_min_stock,
            allergens=json.dumps([a.value for a in product.allergens]) if product.allergens else None,
        )
        
        session.add(new_product)
        session.commit()
        session.refresh(new_product)

        sync_default_pos_stock(session, new_product)
        session.commit()

        return {
            "id": new_product.id,
            "name": new_product.name,
            "price": new_product.price,
            "stock": new_product.stock,
            "default_min_stock": new_product.default_min_stock,
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
                "default_min_stock": p.default_min_stock,
            }
            for p in products
        ]


@app.post("/api/parents/topups", response_model=BalanceTopUpResponse)
async def create_parent_topup(
    payload: BalanceTopUpCreate,
    current_user: User = Depends(require_roles(Role.PARENT))
):
    with Session(engine) as session:
        allocations = compute_topup_allocations(
            session,
            current_user.id,
            payload.total_amount,
            payload.allocation_mode,
            payload.per_student_amounts,
        )

        record = BalanceTopUpRequest(
            parent_id=current_user.id,
            total_amount=payload.total_amount,
            allocation_mode=payload.allocation_mode,
            allocations=json.dumps(allocations),
            payment_reference=payload.payment_reference,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return serialize_topup_request(record)


@app.get("/api/parents/topups", response_model=List[BalanceTopUpResponse])
async def list_parent_topups(current_user: User = Depends(require_roles(Role.PARENT))):
    with Session(engine) as session:
        records = session.exec(
            select(BalanceTopUpRequest)
            .where(BalanceTopUpRequest.parent_id == current_user.id)
            .order_by(BalanceTopUpRequest.created_at.desc())
        ).all()
        return [serialize_topup_request(record) for record in records]


@app.get("/api/topups", response_model=List[BalanceTopUpResponse])
async def list_all_topups(current_user: User = Depends(require_roles(Role.ADMIN, Role.STOCK))):
    with Session(engine) as session:
        records = session.exec(
            select(BalanceTopUpRequest).order_by(BalanceTopUpRequest.created_at.desc())
        ).all()
        return [serialize_topup_request(record) for record in records]


def _process_topup(
    session: Session,
    topup: BalanceTopUpRequest,
    status: BalanceTopUpStatus,
    payment_reference: Optional[str] = None,
):
    if topup.status != BalanceTopUpStatus.PENDING:
        raise HTTPException(status_code=400, detail="Top-up already processed")

    if payment_reference:
        topup.payment_reference = payment_reference

    if status == BalanceTopUpStatus.APPROVED:
        allocations_raw = topup.allocations or "{}"
        try:
            allocations = json.loads(allocations_raw)
            if not isinstance(allocations, dict):
                allocations = {}
        except json.JSONDecodeError:
            allocations = {}
        for student_id, amount in allocations.items():
            student = session.get(Student, student_id)
            if not student:
                continue
            student.balance += int(amount)
            session.add(student)
            session.add(BalanceAdjustment(student_id=student_id, amount=int(amount)))

    topup.status = status
    topup.processed_at = datetime.utcnow()
    session.add(topup)


@app.post("/api/topups/{topup_id}/approve", response_model=BalanceTopUpResponse)
async def approve_topup(
    topup_id: int,
    payload: BalanceTopUpDecision,
    current_user: User = Depends(require_roles(Role.ADMIN, Role.STOCK))
):
    with Session(engine) as session:
        topup = session.get(BalanceTopUpRequest, topup_id)
        if not topup:
            raise HTTPException(status_code=404, detail="Top-up request not found")

        _process_topup(session, topup, BalanceTopUpStatus.APPROVED, payload.payment_reference)
        session.commit()
        session.refresh(topup)
        return serialize_topup_request(topup)


@app.post("/api/topups/{topup_id}/reject", response_model=BalanceTopUpResponse)
async def reject_topup(
    topup_id: int,
    payload: BalanceTopUpDecision,
    current_user: User = Depends(require_roles(Role.ADMIN, Role.STOCK))
):
    with Session(engine) as session:
        topup = session.get(BalanceTopUpRequest, topup_id)
        if not topup:
            raise HTTPException(status_code=404, detail="Top-up request not found")

        _process_topup(session, topup, BalanceTopUpStatus.REJECTED, payload.payment_reference)
        session.commit()
        session.refresh(topup)
        return serialize_topup_request(topup)


@app.post("/api/parents/scheduled-orders", response_model=ScheduledOrderResponse)
async def create_scheduled_order(
    payload: ScheduledOrderCreate,
    current_user: User = Depends(require_roles(Role.PARENT))
):
    if payload.scheduled_for < date.today():
        raise HTTPException(status_code=400, detail="Scheduled date must be today or later")

    with Session(engine) as session:
        ensure_parent_student(session, current_user.id, payload.student_id)

        product_ids = {item.product_id for item in payload.items}
        products = session.exec(select(Product).where(Product.id.in_(product_ids))).all()
        product_map = {product.id: product for product in products}
        missing = [pid for pid in product_ids if pid not in product_map]
        if missing:
            raise HTTPException(status_code=404, detail=f"Products not found: {', '.join(map(str, missing))}")

        order = ScheduledOrder(
            student_id=payload.student_id,
            parent_id=current_user.id,
            scheduled_for=payload.scheduled_for,
            notes=payload.notes,
            pay_from_balance=payload.pay_from_balance,
        )
        session.add(order)
        session.commit()
        session.refresh(order)

        items: List[ScheduledOrderItem] = []
        for item in payload.items:
            order_item = ScheduledOrderItem(
                order_id=order.id,
                product_id=item.product_id,
                quantity=max(1, item.quantity),
            )
            session.add(order_item)
            items.append(order_item)
        session.commit()

        products_dict = {product.id: product for product in products}
        return serialize_scheduled_order(order, items, products_dict)


@app.get("/api/parents/scheduled-orders", response_model=List[ScheduledOrderResponse])
async def list_parent_scheduled_orders(current_user: User = Depends(require_roles(Role.PARENT))):
    with Session(engine) as session:
        orders = session.exec(
            select(ScheduledOrder)
            .where(ScheduledOrder.parent_id == current_user.id)
            .order_by(ScheduledOrder.scheduled_for.asc())
        ).all()

        order_ids = [order.id for order in orders]
        items = (
            session.exec(select(ScheduledOrderItem).where(ScheduledOrderItem.order_id.in_(order_ids))).all()
            if order_ids
            else []
        )
        product_ids = {item.product_id for item in items}
        products = {
            product.id: product
            for product in session.exec(select(Product).where(Product.id.in_(product_ids))).all()
        } if product_ids else {}

        items_by_order: Dict[int, List[ScheduledOrderItem]] = {}
        for item in items:
            items_by_order.setdefault(item.order_id, []).append(item)

        return [serialize_scheduled_order(order, items_by_order.get(order.id, []), products) for order in orders]


@app.get("/api/students/{student_id}/scheduled-orders", response_model=List[ScheduledOrderResponse])
async def list_student_scheduled_orders(
    student_id: str,
    status_filter: Optional[ScheduledOrderStatus] = Query(None),
    current_user: User = Depends(get_current_user)
):
    with Session(engine) as session:
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        ensure_student_access(session, current_user, student_id)

        query = select(ScheduledOrder).where(ScheduledOrder.student_id == student_id)
        if status_filter:
            query = query.where(ScheduledOrder.status == status_filter)
        orders = session.exec(query.order_by(ScheduledOrder.scheduled_for.asc())).all()

        order_ids = [order.id for order in orders]
        items = (
            session.exec(select(ScheduledOrderItem).where(ScheduledOrderItem.order_id.in_(order_ids))).all()
            if order_ids
            else []
        )
        product_ids = {item.product_id for item in items}
        products = {
            product.id: product
            for product in session.exec(select(Product).where(Product.id.in_(product_ids))).all()
        } if product_ids else {}

        items_by_order: Dict[int, List[ScheduledOrderItem]] = {}
        for item in items:
            items_by_order.setdefault(item.order_id, []).append(item)

        return [serialize_scheduled_order(order, items_by_order.get(order.id, []), products) for order in orders]


@app.post("/api/scheduled-orders/{order_id}/dispatch", response_model=ScheduledOrderResponse)
async def dispatch_scheduled_order(
    order_id: int,
    current_user: User = Depends(require_roles(Role.ADMIN, Role.CAJERA))
):
    with Session(engine) as session:
        order = session.get(ScheduledOrder, order_id)
        if not order:
            raise HTTPException(status_code=404, detail="Scheduled order not found")

        if order.status != ScheduledOrderStatus.PENDING:
            raise HTTPException(status_code=400, detail="Scheduled order already processed")

        items = session.exec(select(ScheduledOrderItem).where(ScheduledOrderItem.order_id == order.id)).all()
        product_ids = {item.product_id for item in items}
        products = {
            product.id: product
            for product in session.exec(select(Product).where(Product.id.in_(product_ids))).all()
        } if product_ids else {}

        if order.pay_from_balance:
            student = session.get(Student, order.student_id)
            if not student:
                raise HTTPException(status_code=404, detail="Student not found")

            total_amount = 0
            for item in items:
                product = products.get(item.product_id)
                if not product:
                    raise HTTPException(status_code=404, detail="Product not found for scheduled order item")
                total_amount += product.price * item.quantity

            if student.balance < total_amount:
                raise HTTPException(status_code=400, detail="Saldo insuficiente para este pedido programado")

            student.balance -= total_amount
            session.add(student)
            session.add(BalanceAdjustment(student_id=student.id, amount=-total_amount))

        order.status = ScheduledOrderStatus.DISPATCHED
        session.add(order)
        session.commit()

        return serialize_scheduled_order(order, items, products)


@app.post("/api/daily-menu", response_model=DailyMenuResponse)
async def upsert_daily_menu(
    payload: DailyMenuCreate,
    current_user: User = Depends(require_roles(Role.ADMIN, Role.STOCK))
):
    with Session(engine) as session:
        menu = session.exec(select(DailyMenu).where(DailyMenu.menu_date == payload.menu_date)).first()
        if menu:
            menu.title = payload.title
            menu.description = payload.description
        else:
            menu = DailyMenu(
                menu_date=payload.menu_date,
                title=payload.title,
                description=payload.description,
                created_by=current_user.id,
            )
            session.add(menu)
            session.commit()
            session.refresh(menu)

        session.exec(delete(DailyMenuItem).where(DailyMenuItem.menu_id == menu.id))
        session.commit()

        for item in payload.items:
            menu_item = DailyMenuItem(
                menu_id=menu.id,
                product_id=item.product_id,
                name=item.name,
                meal_type=item.meal_type,
            )
            session.add(menu_item)
        session.commit()

        items = session.exec(select(DailyMenuItem).where(DailyMenuItem.menu_id == menu.id)).all()
        return serialize_daily_menu(menu, items)


@app.get("/api/daily-menu", response_model=List[DailyMenuResponse])
async def list_daily_menu(
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    current_user: User = Depends(get_current_user)
):
    if not start:
        start = date.today()
    if not end:
        end = start + timedelta(days=14)

    with Session(engine) as session:
        menus = session.exec(
            select(DailyMenu)
            .where(DailyMenu.menu_date >= start, DailyMenu.menu_date <= end)
            .order_by(DailyMenu.menu_date.asc())
        ).all()
        menu_ids = [menu.id for menu in menus]
        items = (
            session.exec(select(DailyMenuItem).where(DailyMenuItem.menu_id.in_(menu_ids))).all()
            if menu_ids
            else []
        )
        items_by_menu: Dict[int, List[DailyMenuItem]] = {}
        for item in items:
            items_by_menu.setdefault(item.menu_id, []).append(item)

        return [serialize_daily_menu(menu, items_by_menu.get(menu.id, [])) for menu in menus]


@app.post("/api/daily-menu/{menu_id}/selections", response_model=StudentMenuSelectionResponse)
async def create_menu_selection(
    menu_id: int,
    payload: StudentMenuSelectionCreate,
    current_user: User = Depends(require_roles(Role.PARENT))
):
    with Session(engine) as session:
        menu = session.get(DailyMenu, menu_id)
        if not menu:
            raise HTTPException(status_code=404, detail="Menu not found")

        ensure_parent_student(session, current_user.id, payload.student_id)

        menu_item = session.get(DailyMenuItem, payload.menu_item_id)
        if not menu_item or menu_item.menu_id != menu_id:
            raise HTTPException(status_code=400, detail="Menu item does not belong to this menu")

        selection = StudentMenuSelection(
            parent_id=current_user.id,
            student_id=payload.student_id,
            menu_item_id=payload.menu_item_id,
            menu_date=menu.menu_date,
            notes=payload.notes,
        )
        session.add(selection)
        session.commit()
        session.refresh(selection)
        return serialize_menu_selection(selection)


@app.get("/api/parents/menu-selections", response_model=List[StudentMenuSelectionResponse])
async def list_parent_menu_selections(current_user: User = Depends(require_roles(Role.PARENT))):
    with Session(engine) as session:
        selections = session.exec(
            select(StudentMenuSelection)
            .where(StudentMenuSelection.parent_id == current_user.id)
            .order_by(StudentMenuSelection.menu_date.desc())
        ).all()
        return [serialize_menu_selection(selection) for selection in selections]
@app.post("/api/parents/register", response_model=UserResponse)
async def parent_register(payload: ParentRegisterRequest):
    with Session(engine) as session:
        if get_user_by_email(session, payload.email):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

        user = User(
            email=payload.email,
            full_name=f"{payload.first_name} {payload.last_name}".strip(),
            first_name=payload.first_name,
            last_name=payload.last_name,
            dni=payload.dni,
            phone=payload.phone,
            role=Role.PARENT,
            hashed_password=get_password_hash(payload.password),
            is_active=True,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return serialize_user(user)


@app.post("/api/parents/{parent_id}/students")
async def assign_students_to_parent(
    parent_id: int,
    payload: ParentStudentAssignRequest,
    current_user: User = Depends(require_roles(Role.ADMIN))
):
    with Session(engine) as session:
        parent = session.get(User, parent_id)
        if not parent or parent.role != Role.PARENT:
            raise HTTPException(status_code=404, detail="Parent not found")

        students = session.exec(select(Student).where(Student.id.in_(payload.student_ids))).all()
        found_ids = {s.id for s in students}
        missing = [sid for sid in payload.student_ids if sid not in found_ids]
        if missing:
            raise HTTPException(status_code=404, detail=f"Students not found: {', '.join(missing)}")

        existing_links = session.exec(
            select(StudentGuardian).where(
                StudentGuardian.parent_user_id == parent_id,
                StudentGuardian.student_id.in_(payload.student_ids)
            )
        ).all()
        existing_pairs = {(link.student_id, link.parent_user_id) for link in existing_links}

        for student_id in payload.student_ids:
            if (student_id, parent_id) in existing_pairs:
                continue
            session.add(StudentGuardian(student_id=student_id, parent_user_id=parent_id))

        session.commit()

        return {
            "parent_id": parent_id,
            "student_ids": payload.student_ids,
            "assigned": len(payload.student_ids) - len(existing_pairs),
        }


@app.post("/api/parents/link-requests", response_model=ParentStudentLinkRequestResponse)
async def create_link_request(
    payload: ParentStudentLinkRequestCreate,
    current_user: User = Depends(require_roles(Role.PARENT))
):
    with Session(engine) as session:
        request = ParentStudentLinkRequest(
            parent_id=current_user.id,
            student_identifier=payload.student_identifier,
            student_name=payload.student_name,
            student_grade=payload.student_grade,
            notes=payload.notes,
        )
        session.add(request)
        session.commit()
        session.refresh(request)
        return serialize_link_request(request)


@app.get("/api/parents/link-requests", response_model=List[ParentStudentLinkRequestResponse])
async def list_parent_link_requests(current_user: User = Depends(require_roles(Role.PARENT))):
    with Session(engine) as session:
        requests = session.exec(
            select(ParentStudentLinkRequest)
            .where(ParentStudentLinkRequest.parent_id == current_user.id)
            .order_by(ParentStudentLinkRequest.created_at.desc())
        ).all()
        return [serialize_link_request(r) for r in requests]


@app.get("/api/link-requests", response_model=List[ParentStudentLinkRequestResponse])
async def list_all_link_requests(current_user: User = Depends(require_roles(Role.ADMIN))):
    with Session(engine) as session:
        requests = session.exec(
            select(ParentStudentLinkRequest).order_by(ParentStudentLinkRequest.created_at.desc())
        ).all()
        return [serialize_link_request(r) for r in requests]


@app.post("/api/link-requests/{request_id}/approve", response_model=ParentStudentLinkRequestResponse)
async def approve_link_request(
    request_id: int,
    payload: ParentStudentLinkDecision,
    current_user: User = Depends(require_roles(Role.ADMIN))
):
    with Session(engine) as session:
        request = session.get(ParentStudentLinkRequest, request_id)
        if not request:
            raise HTTPException(status_code=404, detail="Request not found")

        process_link_request(session, request, ParentStudentLinkStatus.APPROVED, payload.student_id, payload.admin_notes)
        session.commit()
        session.refresh(request)
        return serialize_link_request(request)


@app.post("/api/link-requests/{request_id}/reject", response_model=ParentStudentLinkRequestResponse)
async def reject_link_request(
    request_id: int,
    payload: ParentStudentLinkDecision,
    current_user: User = Depends(require_roles(Role.ADMIN))
):
    with Session(engine) as session:
        request = session.get(ParentStudentLinkRequest, request_id)
        if not request:
            raise HTTPException(status_code=404, detail="Request not found")

        process_link_request(session, request, ParentStudentLinkStatus.REJECTED, admin_notes=payload.admin_notes)
        session.commit()
        session.refresh(request)
        return serialize_link_request(request)


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
async def update_product(
    product_id: int,
    product: ProductUpdate,
    current_user: User = Depends(require_roles(Role.ADMIN, Role.STOCK, Role.CAJERA))
):
    """Update an existing product (name, price, stock)."""
    with Session(engine) as session:
        db_product = session.get(Product, product_id)
        if not db_product:
            raise HTTPException(status_code=404, detail="Product not found")

        if current_user.role in (Role.ADMIN, Role.STOCK):
            db_product.name = product.name
            db_product.price = product.price
            db_product.allergens = json.dumps([a.value for a in product.allergens]) if product.allergens else None
        else:
            # Cajeras solo pueden ajustar stock y mínimo; ignorar cambios en otros campos
            product.stock = product.stock  # keep for clarity

        db_product.stock = product.stock
        db_product.default_min_stock = product.default_min_stock
        session.add(db_product)
        session.commit()
        session.refresh(db_product)

        sync_default_pos_stock(session, db_product)
        session.commit()

        return {
            "id": db_product.id,
            "name": db_product.name,
            "price": db_product.price,
            "stock": db_product.stock,
            "default_min_stock": db_product.default_min_stock,
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
                "created_at": localize_iso(t[0].created_at)
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

        adjustment = BalanceAdjustment(student_id=student_id, amount=amount)
        session.add(adjustment)
        session.commit()
        session.refresh(adjustment)
        
        return {
            "success": True,
            "new_balance": student.balance,
            "added_amount": amount,
            "logged_at": localize_iso(adjustment.created_at),
        }


@app.get("/api/students/{student_id}/credit-history")
async def get_credit_history(student_id: str, limit: int = 25):
    """Return the latest balance adjustments for a student"""
    with Session(engine) as session:
        student = session.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        history = session.exec(
            select(BalanceAdjustment)
            .where(BalanceAdjustment.student_id == student_id)
            .order_by(BalanceAdjustment.created_at.desc())
            .limit(limit)
        ).all()

        return [
            {
                "id": entry.id,
                "amount": entry.amount,
                "created_at": localize_iso(entry.created_at),
            }
            for entry in history
        ]

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
        if stock_record.current_stock < quantity and pos_id == DEFAULT_POS_ID and product.stock >= quantity:
            # Re-sync default POS stock if product stock was updated but POS stock lagged behind
            stock_record.current_stock = product.stock
            stock_record.updated_at = datetime.utcnow()
            session.add(stock_record)

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

        # Keep product table stock in sync for admin view / reporting
        if product.stock is not None:
            product.stock = max(0, product.stock - quantity)
            session.add(product)

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
                "created_at": localize_iso(t.created_at),
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
            empty_summary = {
                "top_products": [],
                "top_students": [],
                "daily_sales": [],
                "summary": {
                    "total_sales": 0,
                    "total_transactions": 0,
                    "average_ticket": 0,
                    "unique_students": 0,
                    "best_day": None,
                },
            }
            return empty_summary

        product_stats = {}
        student_stats = {}
        daily_stats = {}
        total_sales_amount = 0

        for t, s, p in rows:
            pid = p.id
            sid = s.id
            day = t.created_at.date().isoformat()
            total_sales_amount += t.amount

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

        total_transactions = len(rows)
        unique_students = len(student_stats)
        average_ticket = (
            total_sales_amount / total_transactions if total_transactions else 0
        )
        best_day = (
            max(daily_sales, key=lambda x: x["total_amount"]) if daily_sales else None
        )

        return {
            "top_products": top_products,
            "top_students": top_students,
            "daily_sales": daily_sales,
            "summary": {
                "total_sales": total_sales_amount,
                "total_transactions": total_transactions,
                "average_ticket": average_ticket,
                "unique_students": unique_students,
                "best_day": best_day,
            },
        }

# Serve data directory for faces and other assets
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.get("/sales")
async def sales_page():
    """Dedicated sales page route"""
    return FileResponse(STATIC_DIR / "sales.html")


if PARENTS_DIR.exists():
    app.mount("/parents", StaticFiles(directory=str(PARENTS_DIR), html=True), name="parents")


# Mount static files (absolute path to avoid CWD issues)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

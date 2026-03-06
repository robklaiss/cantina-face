#!/usr/bin/env python3
"""
Script de sincronización de estudiantes desde el sistema de caja hacia el backend en la nube.
Se ejecuta periódicamente (ej: cada 5 minutos) para mantener sincronizados los datos.
"""

import os
import sys
import sqlite3
import requests
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configuración
SCRIPT_DIR = Path(__file__).parent
# Try different possible database locations
possible_paths = [
    Path("/opt/cantina-face/data/db.sqlite"),  # Production installation
    SCRIPT_DIR / "data" / "db.sqlite",          # Development (script in project dir)
    Path.home() / "data" / "db.sqlite",         # Alternative home installation
]
DB_PATH = None
for path in possible_paths:
    if path.exists():
        DB_PATH = path
        break
if DB_PATH is None:
    DB_PATH = SCRIPT_DIR / "data" / "db.sqlite"  # fallback

CONFIG_PATH = SCRIPT_DIR / "config.py"

# Cargar configuración
sys.path.insert(0, str(SCRIPT_DIR))
def _load_env_sync():
    env_path = SCRIPT_DIR / ".env-sync"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k, v)


_load_env_sync()

try:
    from config import CLOUD_BACKEND_URL, CLOUD_SYNC_TOKEN
except ImportError:
    CLOUD_BACKEND_URL = os.getenv("CLOUD_BACKEND_URL", "https://sistema.siloe.com.py")
    CLOUD_SYNC_TOKEN = os.getenv("CLOUD_SYNC_TOKEN", "change-this-sync-token")

SYNC_ENDPOINT = f"{CLOUD_BACKEND_URL}/api/sync.php?action=students"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_students_from_local_db() -> List[Dict[str, Any]]:
    """Obtiene todos los estudiantes de la base de datos local."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, grade, balance, photo_path, point_of_sale_id
        FROM student
        ORDER BY name
    """)
    
    students = []
    for row in cursor.fetchall():
        students.append({
            'id': row['id'],
            'name': row['name'],
            'grade': row['grade'],
            'balance': row['balance'],
            'photo_path': row['photo_path'],
            'point_of_sale_id': row['point_of_sale_id']
        })
    
    conn.close()
    return students


def sync_students_to_cloud(students: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Envía los estudiantes al backend en la nube."""
    headers = {
        'Content-Type': 'application/json',
        'X-Sync-Token': CLOUD_SYNC_TOKEN,
        'User-Agent': 'CantinaPOS/1.0 (Student Sync)'
    }
    
    payload = {
        'students': students
    }
    
    try:
        response = requests.post(
            SYNC_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        # Log response for debugging
        logger.debug(f"Response status: {response.status_code}")
        logger.debug(f"Response headers: {response.headers}")
        
        if response.status_code == 406:
            logger.error("Error 406: El servidor rechazó la petición (posible bloqueo de ModSecurity)")
            logger.error(f"Response body: {response.text[:500]}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al sincronizar con el backend: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response body: {e.response.text[:500]}")
        raise


def pull_students_from_cloud() -> List[Dict[str, Any]]:
    headers = {
        "X-Sync-Token": CLOUD_SYNC_TOKEN,
        "User-Agent": "CantinaPOS/1.0 (Student Sync)"
    }
    url = f"{CLOUD_BACKEND_URL}/api/sync.php?action=students"
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("ok") is False:
        raise RuntimeError(payload.get("error", "Error desconocido al obtener estudiantes"))
    return payload.get("data", []) or payload


def apply_cloud_balances_to_local(students: List[Dict[str, Any]]):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for stu in students:
        sid = stu.get("id")
        balance = stu.get("balance")
        if sid is None or balance is None:
            continue
        cursor.execute(
            "UPDATE student SET balance = ? WHERE id = ?",
            (balance, sid),
        )
    conn.commit()
    conn.close()


def main():
    """Función principal de sincronización."""
    logger.info("Iniciando sincronización de estudiantes...")
    
    try:
        # 1) Pull primero: traer balances de la nube y aplicarlos localmente
        cloud_students = pull_students_from_cloud()
        apply_cloud_balances_to_local(cloud_students)
        logger.info("Balances locales actualizados desde nube")

        # 2) Solo pull (evitamos sobrescribir la nube). Si se requiere push, habilitar manualmente.
        logger.info("Sincronización pull-only completada")

    except Exception as e:
        logger.error(f"Error durante la sincronización: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

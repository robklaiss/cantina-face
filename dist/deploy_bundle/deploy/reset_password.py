#!/usr/bin/env python3
"""Reset or create Cantina Face user passwords in the local SQLite DB."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from getpass import getpass
from pathlib import Path

from passlib.context import CryptContext

REPO_DIR = Path(__file__).resolve().parents[1]
DB_PATH = REPO_DIR / "data" / "db.sqlite"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset Cantina Face user password")
    parser.add_argument("--email", default=os.getenv("RESET_EMAIL"), help="Email address of the user (or env RESET_EMAIL)")
    parser.add_argument("--password", default=os.getenv("RESET_PASSWORD"), help="New password value (or env RESET_PASSWORD)")
    parser.add_argument(
        "--prompt-password",
        action="store_true",
        help="Prompt for password interactively if --password is omitted",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the user if it does not exist (role=admin, active)",
    )
    args = parser.parse_args()
    if not args.email:
        parser.error("Se requiere --email o la variable de entorno RESET_EMAIL")
    return args


def get_password_value(args: argparse.Namespace) -> str:
    if args.password:
        return args.password
    if args.prompt_password:
        pwd = getpass("Nuevo password: ")
        if not pwd:
            sys.exit("[reset_password] Password vacío")
        confirm = getpass("Confirmar password: ")
        if pwd != confirm:
            sys.exit("[reset_password] Los passwords no coinciden")
        return pwd
    sys.exit("[reset_password] Debes usar --password o --prompt-password")


def ensure_db_exists() -> None:
    if not DB_PATH.exists():
        sys.exit(f"[reset_password] No se encontró la base en {DB_PATH}")


def upsert_user(email: str, password: str, create: bool) -> int:
    hashed = pwd_context.hash(password)

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT id FROM user WHERE email = ?", (email,))
        row = cur.fetchone()

        if row:
            cur = conn.execute(
                """
                UPDATE user
                   SET hashed_password = ?,
                       plain_password = ?,
                       is_active = 1
                 WHERE email = ?
                """,
                (hashed, password, email),
            )
            conn.commit()
            return cur.rowcount

        if not create:
            sys.exit("[reset_password] Usuario no existe. Usa --create para crearlo.")

        cur = conn.execute(
            """
            INSERT INTO user (email, full_name, role, hashed_password, plain_password, is_active)
            VALUES (?, ?, ?, ?, ?, 1)
            """,
            (email, "Administrador", "admin", hashed, password),
        )
        conn.commit()
        return cur.rowcount


def main() -> None:
    args = parse_args()
    ensure_db_exists()
    password = get_password_value(args)
    updated = upsert_user(args.email, password, args.create)
    print(f"OK actualizado — {args.email} ({updated} fila(s) afectada(s))")
    print(f"Base de datos: {DB_PATH}")


if __name__ == "__main__":
    main()

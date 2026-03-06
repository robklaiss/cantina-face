#!/usr/bin/env python3
"""
Smoke test de cámara para Cantina Face.
Captura frames headless durante N segundos, intenta detección Haar y reporta FPS.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cantina Face camera smoke test")
    parser.add_argument("--device", default="/dev/video0", help="Dispositivo de video (default: %(default)s)")
    parser.add_argument("--seconds", type=float, default=3.0, help="Duración del test en segundos (default: %(default)s)")
    parser.add_argument("--width", type=int, default=640, help="Ancho deseado (default: %(default)s)")
    parser.add_argument("--height", type=int, default=480, help="Alto deseado (default: %(default)s)")
    parser.add_argument("--fps", type=int, default=15, help="FPS objetivo (default: %(default)s)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print(f"❌ No se pudo abrir la cámara {args.device}", file=sys.stderr)
        return 1

    start = time.time()
    deadline = start + max(0.5, args.seconds)
    frames = 0
    detect_hits = 0
    last_face = None

    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        if len(faces) > 0:
            detect_hits += 1
            last_face = faces[0].tolist()

    cap.release()

    duration = max(1e-6, time.time() - start)
    if frames == 0:
        print("❌ No se pudieron capturar frames. Verificá que la cámara esté libre y accesible.", file=sys.stderr)
        return 1

    fps = frames / duration
    print(f"✅ Camera smoketest OK — device={args.device} frames={frames} duration={duration:.2f}s fps={fps:.1f} hits={detect_hits}")
    if last_face is not None:
        print(f"   Último bounding box Haar: {last_face}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

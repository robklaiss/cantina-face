# Cantina Face - Sistema de Caja Rápida

Sistema completo de cantina escolar offline con reconocimiento facial y respaldo manual, optimizado para kioscos de autoservicio.

## Características

- **100% Offline**: Sin conexión a internet requerida, todo procesamiento local
- **Reconocimiento Facial**: Modelo ArcFace con embeddings de 512-D para identificación precisa
- **Búsqueda Rápida**: Búsqueda vectorial HNSW para consultas <10ms con hasta 10k estudiantes
- **Respaldo Manual**: Búsqueda por nombre cuando falla el reconocimiento facial
- **Procesamiento en Tiempo Real**: Feed de video en vivo con detección continua de rostros
- **Gestión de Productos**: Compra rápida con atajos de teclado numérico (1-9)
- **Gestión de Estudiantes**: Registro ágil de estudiantes con captura rápida o ráfaga
- **Base de Datos SQLite**: Almacenamiento local con SQLModel para persistencia de datos
- **UI Moderna**: Interfaz web de alto contraste optimizada para kioscos táctiles y teclado

## Inicio Rápido

### Prerrequisitos

- Python 3.10+
- Webcam/cámara con acceso
- ~2GB de espacio libre para modelos y datos

### Instalación

1. **Clona o descarga** el proyecto
2. **Prepara y ejecuta los scripts**:
   ```bash
   chmod +x setup.sh run.sh
   ./setup.sh   # crea/activa venv e instala dependencias
   ./run.sh     # activa el venv y arranca uvicorn

   # Windows
   run.bat
   ```

3. **Abre tu navegador** y ve a: `http://localhost:8000/login.html` para iniciar sesión antes de usar `index.html`

## Quick Start

### Prerequisites

- Python 3.10+
- Webcam/camera access
- ~2GB free disk space for models and data

### Installation

1. **Clone or download** the project
2. **Prepare and run the helper scripts**:
   ```bash
   chmod +x setup.sh run.sh
   ./setup.sh   # creates the venv and installs deps
   ./run.sh     # starts uvicorn inside the venv

   # Windows
   run.bat
   ```

3. **Open your browser** and head to `http://localhost:8000/login.html` first, then continue in `index.html`/`admin.html` once authenticated.

## Deploy 1-click (Chromebook / Debian/Ubuntu)

Estos pasos funcionan en un contenedor Debian/Ubuntu limpio (incluido Chromebook Crostini) sin instalaciones manuales adicionales:

1. Copia `project.zip` a tu contenedor y descomprímelo:
   ```bash
   unzip project.zip
   cd cantina-face
   chmod +x deploy/*.sh
   ```
2. Corre el instalador automatizado. Detecta `apt`, usa `sudo` si está disponible e instala `python3`, `python3-venv`, `python3-dev`, `python3-pip`, `build-essential`, `g++`, `rsync` y `lsof`. Luego crea `venv/` dentro del repo y ejecuta `python3 -m pip install -r requirements.txt`:
   ```bash
   ./deploy/install.sh
   ```
3. **Lanza la app (usa uvicorn directamente). Por defecto se publica en `127.0.0.1:8000`.** Puedes ajustar la vinculación con `HOST=0.0.0.0` (o `BIND=0.0.0.0` para compatibilidad), cambiar puerto con `PORT=9000` y activar recarga en vivo con `RELOAD=1`. Si el puerto ya está ocupado y tienes `lsof`, verás una advertencia:
   ```bash
   ./deploy/run.sh
   # o
   PORT=9001 RELOAD=1 ./deploy/run.sh
   ```
4. Abre `http://localhost:8000/login.html` (o el puerto elegido) para iniciar sesión antes de usar `index.html`.

### Actualizaciones rápidas del zip

`deploy/update.sh` automatiza la rotación de versiones desde un zip de despliegue:

```bash
./deploy/update.sh                # usa deploy/project.zip por defecto
./deploy/update.sh /ruta/otra.zip # opcional
```

- Descomprime en una carpeta temporal, ignore archivos basura de macOS y detecta si el zip trae `cantina-face/` o solo los archivos.
- Respalda la versión actual como `deploy/backups/cantina-face-YYYYmmdd-HHMMSS.tgz`, manteniendo por defecto los últimos 5 o 7 días (configurable con `BACKUP_MAX_COUNT` y `BACKUP_MAX_AGE_DAYS`).
- Mantiene `data/` intacto y lo reinserta después del `rsync` (`--delete --exclude venv deploy/backups deploy/project.zip .DS_Store ._*`).
- Recrea/actualiza `venv/` con `python3` (o `PYTHON_BIN`) y reinstala dependencias automáticamente.
- Si `rsync`, `unzip` o `python3` faltan, muestra un mensaje claro indicando qué instalar.

> Con esto basta ejecutar `./deploy/update.sh && ./deploy/run.sh` para dejar la instancia al día sin mover carpetas ni correr comandos manuales. No necesitas rutas absolutas: todos los scripts calculan `REPO_DIR` automáticamente.

### Autoarranque y kiosco listo para usar

El modo kiosco ahora se configura directamente con los scripts estándar. Instala dependencias y crea el servicio/autostart/shortcut ejecutando:

```bash
chmod +x deploy/install.sh deploy/update.sh
./deploy/install.sh --setup-autostart
# o si ya tenías la app instalada:
./deploy/update.sh --setup-autostart
```

Las banderas `--setup-autostart` generan:

1. **Servicio systemd** `cantina-face.service` (personalizable) que llama a `deploy/run.sh` en cada arranque.
2. **Autostart gráfico** en `~/.config/autostart/` que abre `http://localhost:8000/login.html` tras iniciar sesión.
3. **Acceso directo** `~/Desktop/CantinaFace.desktop` para iniciar manualmente y abrir el login.

> Tip: tanto `deploy/install.sh` como `deploy/update.sh` ejecutan automáticamente `deploy/backup_faces.sh` antes de tocar el código, dejando las copias `db-backup-01..03` listas por si necesitas restaurar la base facial.

Variables opcionales (exporta antes de ejecutar el comando):

- `SERVICE_NAME`: nombre del servicio systemd (default `cantina-face`).
- `LOGIN_URL`: URL a abrir (default `http://localhost:8000/login.html`).
- `TARGET_USER`: dueño del servicio/autostart (default usuario que corre el script).
- `AUTOSTART_DISPLAY`: display X11 (default `:0`).
- `ICON_PATH_OVERRIDE`: ruta alternativa para el ícono del acceso directo.

Administración rápida del servicio:

```bash
sudo systemctl status cantina-face
sudo systemctl restart cantina-face
sudo systemctl disable cantina-face  # si quieres desactivarlo
```

> Requisitos: distro con systemd, `sudo`, sesión gráfica y el repo alojado en el home del usuario configurado.

## Acceso y Autenticación

1. Dirígete a `http://localhost:8000/login.html`.
2. Ingresa el usuario y contraseña del personal (el sistema crea `admin@siloe.com.py / admin123` si no existe).
3. Tras iniciar sesión se guarda un token JWT en `localStorage` y se redirige a la última página solicitada.
4. Usa los botones "Cerrar sesión" en `index.html` o `admin.html` para limpiar el token y volver a la pantalla de login.

> Si el token expira o se borra, cualquier visita a `index.html`/`admin.html` redireccionará automáticamente a la pantalla de acceso.

### Reset oficial de contraseñas (DB local correcta)

El único script soportado para resetear/crear usuarios administra siempre `data/db.sqlite` dentro del repo:

```bash
cd cantina-face
source venv/bin/activate
python deploy/reset_password.py \
  --email admin@siloe.com.py \
  --prompt-password \
  --create      # opcional, crea el usuario si no existe
```

## Chromebook / ChromeOS Flex (Instalador Local)

### Instalación inicial

1. **Activa Linux (Beta)** en *Configuración → Avanzado → Desarrolladores → Linux* y establece al menos 10 GB de espacio.
2. **Copia la carpeta del proyecto** (por USB, Drive o Git) dentro de `Archivos → Linux`.
3. **Abre la terminal de Linux** y ejecuta la instalación completa:
   ```bash
   cd cantina-face
   chmod +x deploy/install.sh deploy/run.sh
   bash deploy/install.sh --setup-autostart
   ```
   Esto instala dependencias, crea el entorno virtual, configura el servicio para que arranque automáticamente y crea un acceso directo en el launcher de ChromeOS.

### Inicio automático al encender

Para que el sistema arranque **sin intervención** al prender la máquina:

1. Abre **Chrome** en la barra de direcciones y ve a `chrome://flags`
2. Busca **"Start Linux on login"** (o `#crostini-use-lxd-5`)
3. Cámbialo a **Enabled** y reinicia ChromeOS

Con esto, al encender la máquina → ChromeOS inicia → Linux arranca automáticamente → el servicio de Cantina Face se levanta solo.

### Acceso directo (alternativa manual)

Si prefieres iniciar manualmente con un clic:

1. Abre el **launcher** de ChromeOS (el cajón de apps)
2. Busca **"Cantina Face"** — aparece como app de Linux
3. Haz clic derecho → **Anclar al estante** para tenerlo siempre visible en la barra inferior
4. Al hacer clic, inicia el servidor y abre el navegador automáticamente

### Inicio manual desde terminal

Si necesitas iniciar manualmente desde la terminal de Linux:
```bash
cd cantina-face
./deploy/run.sh
```
Luego abre `http://localhost:8000/login.html` en el navegador de ChromeOS.

### Comandos útiles (ChromeOS)

| Acción | Comando |
|--------|---------|
| Ver estado del servicio | `systemctl --user status cantina-face` |
| Reiniciar servicio | `systemctl --user restart cantina-face` |
| Ver logs | `journalctl --user -u cantina-face -f` |
| Detener servicio | `systemctl --user stop cantina-face` |

## Guía de Uso

### Primera Configuración

1. **Productos Demo**: El sistema incluye productos de demostración. Ejecuta:
   ```bash
   curl -X POST http://localhost:8000/api/seed
   ```

2. **Registro de Estudiantes**: Presiona **Ctrl+E** o haz clic en "Registro Ágil"
   - **Captura Rápida**: Una foto para registro inmediato
   - **Captura Ráfaga**: 3-5 fotos para mejor precisión

### Operación Diaria

#### Modo Reconocimiento Facial (Automático)
1. El estudiante se para frente a la cámara
2. El sistema detecta y reconoce automáticamente el rostro
3. Aparece información del estudiante (foto, nombre, grado, saldo)
4. Selecciona producto con teclas numéricas (1-9) o botones
5. Presiona **Enter** para completar la transacción

#### Modo Manual (Respaldo)
1. Presiona **F2** para enfocar la caja de búsqueda
2. Escribe nombre del estudiante (2+ letras)
3. Navega con ↑↓ y presiona **Enter** para seleccionar
4. Selecciona producto y cobra normalmente

### Atajos de Teclado

- **1-9**: Seleccionar productos (corresponde a botones)
- **Enter**: Cobrar producto al estudiante actual
- **F2**: Enfocar caja de búsqueda para búsqueda manual
- **Escape**: Limpiar selección actual de estudiante
- **Ctrl+E**: Ir a registro ágil de estudiantes
- **Space**: Capturar foto (en modo registro)
- **Shift+Space**: Captura ráfaga (en modo registro)
- **Ctrl+S**: Guardar registro (en modo registro)
- **Ctrl+←**: Volver a caja (desde registro)

### Proceso de Registro Ágil

1. Presiona **Ctrl+E** desde la caja principal
2. Completa nombre y grado del estudiante
3. Configura saldo inicial (opcional)
4. **Captura Rápida**: Una foto para registro inmediato
5. **Captura Ráfaga**: 5 fotos en 2 segundos para mejor precisión
6. El sistema calcula embedding automáticamente
7. Presiona **Ctrl+S** para guardar

## Arquitectura del Sistema

```
cantina-face/
├── app.py              # Backend FastAPI con reconocimiento facial
├── face_engine.py      # Modelo ArcFace y procesamiento facial
├── config.py           # Configuración del sistema
├── models/             # Modelos IA (ArcFace ONNX)
├── data/               # Base de datos SQLite e índice HNSW
│   ├── db.sqlite       # Datos de estudiantes, productos, transacciones
│   ├── index.bin       # Índice de búsqueda vectorial HNSW
│   └── faces/          # Almacenamiento de fotos de estudiantes
└── static/             # Frontend web
    ├── index.html      # UI principal con layout de 4 zonas
    ├── enroll.html     # Página de registro ágil
    ├── app.js          # Lógica frontend y llamadas API
    ├── enroll.js       # Manejo de cámara para registro
    └── style.css       # Estilos optimizados para kioscos
```

## Technical Details

### Face Recognition Pipeline
1. **Detection**: OpenCV Haar cascades for face detection
2. **Alignment**: Center crop for rough face alignment
3. **Embedding**: ArcFace R50 model extracts 512-D features
4. **Search**: HNSW index for fast similarity search
5. **Threshold**: Configurable similarity threshold (default 0.38)

### Liveness Detection Pipeline
The system includes basic spoof protection using facial movement analysis:

1. **MediaPipe Processing**: Uses MediaPipe Face Mesh for precise facial landmark detection
2. **Blink Detection**: Monitors Eye Aspect Ratio (EAR) for eye closure detection
3. **Mouth Movement**: Tracks Mouth Aspect Ratio (MAR) for mouth opening
4. **Frame History**: Maintains rolling history of last 10 frames for consecutive detection
5. **Threshold Logic**: Requires 2+ consecutive frames of either blink OR mouth movement

#### Thresholds
- **EAR Blink Threshold**: <0.21 (eye closed when ratio below this value)
- **MAR Movement Threshold**: >0.6 (mouth open when ratio above this value)
- **Consecutive Frames**: 2+ frames required for liveness confirmation

#### UI Indicators
- **Green Border**: Match + Liveness confirmed ✅
- **Amber Border**: Match but liveness not confirmed ⚠️
- **Red Border**: No match
- **Warning Message**: "Live not confirmed - please blink or move your mouth"

### Database Schema
- **Students**: ID, name, grade, balance, photo_path, embedding
- **Products**: ID, name, price, description
- **Transactions**: Student ID, product ID, amount, timestamp

### Performance
- **Recognition Speed**: <10ms per frame en escenarios normales
- **Storage**: ~5KB por estudiante (embedding + mini photo)
- **Memory**: ~200MB RAM usage
- **CPU**: Optimized for CPU-only inference gracias a throttling y cacheo de embeddings

Tuning clave vía variables de entorno:

| Variable | Default | Descripción |
| --- | --- | --- |
| `FACE_MAX_EMB_PER_SEC` | `2` | Límite de embeddings por segundo por cámara (cooldown automático). |
| `FACE_CACHE_MS` | `500` | Ventana de cache en ms para reusar coincidencias cuando el rostro no cambia. |
| `FACE_CACHE_IOU` | `0.7` | IoU mínima entre bounding boxes para considerar la cara la misma. |
| `FACE_DETECT_WIDTH` | `640` | Ancho al que se normaliza el frame para detección (reduce CPU). |
| `ORT_INTRA_THREADS` / `ORT_INTER_THREADS` | `1` | Controlan los hilos de onnxruntime para evitar saturar la CPU. |
| `CV2_NUM_THREADS` | `1` | Limita threads internos de OpenCV. |

Además, `/api/health/timing` (requiere autenticación) devuelve un resumen JSON de los últimos `PERF_WINDOW_SECONDS` segundos, agregando métricas de detección/embedding y la cola de requests recientes capturada por el middleware de FastAPI. Ejemplo abreviado:

```json
{
  "window_seconds": 10,
  "face": {
    "detect": { "count": 8, "avg_ms": 7.9, "p95_ms": 9.1, "max_ms": 11.2 },
    "embed": { "count": 4, "avg_ms": 53.3, "p95_ms": 58.4, "max_ms": 60.0 }
  },
  "requests": { "count": 25, "avg_ms": 12.4, "p95_ms": 19.7, "max_ms": 22.1 }
}
```

Usa esto para verificar que los embeddings bajen drásticamente (<= ~20/100s) en hardware limitado o para detectar endpoints lentos en la caja.

#### Diagnóstico y benchmark del checkout

El script `scripts/bench_checkout.py` automatiza una sesión mínima del flujo de caja para medir latencias de `/api/products`, `/api/students/{id}` y `/api/students/{id}/scheduled-orders`. Ejecútalo dentro del repo (venv opcional):

```bash
python scripts/bench_checkout.py \
  --username admin@siloe.com.py \
  --password "admin123" \
  --iterations 5
```

Notas:

1. El script obtiene un token vía `/auth/token` y reusa el primer alumno disponible, salvo que pases `--student-id`.
2. Reporta `avg/p95/max` por endpoint y, al finalizar, imprime el payload crudo de `/api/health/timing` para correlacionar los resultados.
3. Puedes apuntarlo a otra máquina con `--base-url http://IP:PUERTO` (por defecto `http://127.0.0.1:8000`).

## Configuration

### Similarity Threshold
Adjust face recognition sensitivity in `app.py`:
```python
SIMILARITY_THRESHOLD = 0.38  # Lower = more strict, Higher = more lenient
```

### Liveness Detection Thresholds
Configure spoof protection sensitivity in `config.py`:
```python
LIVENESS_CONFIG = {
    'ear_blink_threshold': 0.21,        # Lower = more sensitive to blinks
    'mar_movement_threshold': 0.6,      # Higher = more sensitive to mouth movement
    'consecutive_frames_required': 2,   # Frames needed for confirmation
    'max_frame_history': 10,           # Frames to keep in memory
}
```

### Camera Settings
Modify camera resolution in `static/app.js`:
```javascript
video: {
    width: { ideal: 640 },
    height: { ideal: 480 },
    facingMode: 'user'
}
```

## Privacy & Security

### Data Storage
- All data stored locally on your device
- No internet connection required or used
- Face embeddings and photos never leave your system

### Privacy Considerations
- **Consent**: Obtain parental/guardian consent before enrolling students
- **Data Retention**: Students can be removed from the system at any time
- **Access Control**: Physical access to the device controls system access
- **Transparency**: Parents can request to see stored photos and data

### Security Best Practices
- Keep the device in a secure location
- Use strong passwords for device access
- Regularly backup the `data/` directory
- Monitor transaction logs for irregularities

### Respaldos de la base facial

Para no perder embeddings ni miniaturas críticas, se añadió `deploy/backup_faces.sh`, que crea hasta 3 copias rotativas (`db-backup-01`, `-02`, `-03`) dentro de `data/backups/`:

```bash
chmod +x deploy/backup_faces.sh
./deploy/backup_faces.sh            # usa data/ y 3 copias por defecto
BACKUP_COUNT=5 BACKUP_DIR=/mnt/usb/backups \
    ./deploy/backup_faces.sh        # ejemplo con destino externo
```

Qué incluye cada backup:

1. `db.sqlite` (usando `sqlite3 .backup` cuando está disponible para un snapshot consistente)
2. `index.bin` + `index_labels.json` del índice HNSW
3. Carpeta `data/faces/` con las miniaturas

Ejemplo de cron para ejecutarlo cada 6 horas y guardar registros:

```cron
0 */6 * * * /ruta/a/cantina-face/deploy/backup_faces.sh \
    >> /var/log/cantina-face-backup.log 2>&1
```

Recomendaciones:

- Verificar periódicamente que `data/backups/db-backup-01` existe y contiene `backup-info.txt`
- Sincronizar manualmente estas carpetas a un medio externo o a otra máquina segura
- Ejecutar el script antes de actualizaciones mayores o reinstalaciones

## Troubleshooting

### Common Issues

**"Camera access denied"**
- Grant camera permissions in browser settings
- Try refreshing the page
- Check if camera is being used by another application

**"Model download failed"**
- Check internet connection for first run
- Manually download from: https://storage.googleapis.com/insightface/models/arcface_r50/model.onnx
- Place in `models/arcface_r50.onnx`

**"Liveness not confirmed" warnings**
- Ask student to blink naturally or open mouth slightly
- Ensure good lighting for facial landmark detection
- Check camera angle - face should be clearly visible
- Adjust thresholds in `config.py` if too strict
- Verify MediaPipe is properly installed (check console for errors)

**"Recognition not working"**
- Ensure good lighting on faces
- Clean camera lens
- Re-enroll student with better quality photos
- Adjust similarity threshold if too strict/lenient

**"Low balance warnings"**
- Add funds to student account via enrollment
- Check transaction history for errors

### Performance Tuning
- Close other applications to free up CPU
- Use wired camera if available (better quality)
- Reduce video resolution for slower devices
- Clear browser cache if UI is slow

## API Reference

### Endpoints

- `GET /api/products` - List all products
- `POST /api/seed` - Seed demo products
- `POST /api/students` - Create student manually
- `GET /api/students?query=name` - Search students
- `POST /api/enroll` - Enroll student with photos
- `POST /api/recognize` - Recognize face from image
  - **Response**: `{ match: bool, student: {...}, score: float, bbox: [x,y,w,h], liveness: bool }`
  - **Liveness Field**: `true` = live person detected, `false` = potential spoof or no movement
- `POST /api/charge` - Charge product to student

### WebSocket (Future)
Real-time updates for multiple terminals planned.

## Development

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Adding New Features
- Backend features in `app.py`
- Face processing in `face_engine.py`
- UI changes in `static/` files
- Database changes require migration planning

## License

This project is provided as-is for educational and non-commercial use. Ensure compliance with local privacy laws and regulations.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Test with the demo products first
4. Ensure all prerequisites are met

---

**Remember**: This system processes sensitive student data. Always obtain proper consent and follow your school's privacy policies.

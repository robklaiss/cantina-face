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
2. **Ejecuta el script de configuración**:
   ```bash
   # Linux/Mac
   ./run.sh

   # Windows
   run.bat
   ```

   Esto hará:
   - Crear entorno virtual
   - Instalar todas las dependencias
   - Descargar modelo ArcFace (primera ejecución)
   - Iniciar el servidor

3. **Abre tu navegador** y ve a: `http://localhost:8000/static/index.html`

## Quick Start

### Prerequisites

- Python 3.10+
- Webcam/camera access
- ~2GB free disk space for models and data

### Installation

1. **Clone or download** the project
2. **Run the setup script**:
   ```bash
   # Linux/Mac
   ./run.sh

   # Windows
   run.bat
   ```

   This will:
   - Create a virtual environment
   - Install all dependencies
   - Download the ArcFace model (first run only)
   - Start the server

3. **Open your browser** and go to: `http://localhost:8000/static/index.html`

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
- **Recognition Speed**: <10ms per frame on modern hardware
- **Storage**: ~5KB per student (embedding + mini photo)
- **Memory**: ~200MB RAM usage
- **CPU**: Optimized for CPU-only inference

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

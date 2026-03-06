# Deploy Bundle — Cantina Face

Bundle de despliegue autocontenido y robusto para instalación offline desde USB o local.

## Estructura

```
deploy_bundle/
├── run_install.sh          # Runner principal para instalación
├── run_update.sh           # Runner principal para actualizaciones
├── project.zip             # Código de la aplicación empaquetado
├── README_DEPLOY.md        # Este archivo
├── models/                 # Modelos de IA (obligatorio)
│   └── arcface_r50.onnx    # Modelo de reconocimiento facial (~166MB)
└── deploy/
    ├── install.sh          # Script de instalación (llamado por run_install.sh)
    ├── update.sh           # Script de actualización (llamado por run_update.sh)
    ├── run.sh              # Lanzador de la aplicación
    ├── backup_faces.sh     # Respaldo de datos faciales
    ├── guardrails/
    │   └── check_python_hardcode.sh  # Validación de versiones Python
    └── ubuntu/             # Scripts específicos para Ubuntu offline
        ├── install.sh
        ├── preflight.sh
        └── ...
```

## Características

### 🔒 Noexec-Safe
Los runners `run_install.sh` y `run_update.sh` copian siempre el bundle completo a `/opt/cantina-face-deploy` antes de ejecutar, evitando errores de permisos con USB montado en noexec.

### 📍 Rutas Determinísticas
- `ROOT_DIR`: raíz del bundle copiado (`/opt/cantina-face-deploy`)
- `DEPLOY_DIR`: `$ROOT_DIR/deploy`
- `TARGET_APP_DIR`: directorio real de la app (`/opt/cantina-face`)
- Los modelos se instalan en `$TARGET_APP_DIR/models/`

### 🛡️ Guardrails Consistentes
El script `deploy/guardrails/check_python_hardcode.sh` valida que no haya referencias hardcodeadas a versiones específicas de Python (como `python3.11` o `pip3.12`). Se ejecuta automáticamente antes de instalar/actualizar.

### 📦 Requirements desde ZIP
`update.sh` instala dependencias desde `requirements.txt` dentro del `project.zip` descomprimido, no desde el filesystem del bundle.

## Uso

### Instalación Inicial

```bash
# Desde USB
bash /media/$USER/OS-FLEX/deploy_bundle/run_install.sh
```

El script:
1. Copia el bundle a `/opt/cantina-face-deploy`
2. Ejecuta `deploy/install.sh` con `TARGET_APP_DIR=/opt/cantina-face`
3. Instala modelos en `/opt/cantina-face/models/`
4. Crea el entorno virtual e instala dependencias

### Actualización

```bash
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
```

El script:
1. Copia el bundle a `/opt/cantina-face-deploy`
2. Ejecuta guardrails
3. Genera backup
4. Descomprime el ZIP y sincroniza código
5. Instala modelos en `/opt/cantina-face/models/`
6. Preserva `data/` existente
7. Reinstala dependencias

### Autostart (opcional)

```bash
bash run_install.sh --setup-autostart
```

Configura:
- Servicio systemd (`cantina-face.service`)
- Autostart gráfico (`~/.config/autostart/`)
- Acceso directo en escritorio

Variables opcionales:
- `SERVICE_NAME`: nombre del servicio (default: `cantina-face`)
- `LOGIN_URL`: URL a abrir (default: `http://localhost:8000/login.html`)
- `TARGET_USER`: usuario propietario (default: `$SUDO_USER`)
- `AUTOSTART_DISPLAY`: display X11 (default: `:0`)
- `ICON_PATH_OVERRIDE`: ruta al ícono

### Ejecución Manual

```bash
bash deploy/run.sh              # Puerto 8000 por defecto
PORT=9000 bash deploy/run.sh    # Puerto custom
```

## Despliegue desde USB

### Preparación del USB

1. Copia el `deploy_bundle/` completo al USB
2. Asegúrate de que `project.zip` esté presente
3. Verifica que `models/arcface_r50.onnx` esté presente (obligatorio)

### En la máquina objetivo

```bash
# Actualización (recomendado):
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh

# Instalación inicial (primera vez):
bash /media/$USER/OS-FLEX/deploy_bundle/run_install.sh
```

**Eso es todo.** El script copia el bundle a disco local y ejecuta automáticamente.

## Variables de Entorno

### Instalación
- `PYTHON_BIN`: binario de Python (default: `python3`)
- `FORCE_VENV`: forzar recreación del venv (default: `0`)
- `SKIP_FACE_BACKUP`: omitir respaldo facial (default: `0`)

### Actualización
- `BACKUP_MAX_COUNT`: máximo de backups a mantener (default: `5`)
- `BACKUP_MAX_AGE_DAYS`: días de retención de backups (default: `7`)
- `TARGET_APP_DIR`: directorio real de la app (default: `/opt/cantina-face`)

## Troubleshooting

### Error: "Guardrail obligatorio no encontrado"
Verifica que `deploy/guardrails/check_python_hardcode.sh` exista y sea ejecutable:
```bash
chmod +x deploy/guardrails/check_python_hardcode.sh
```

### Error: "No se encontró project.zip"
Asegúrate de que `project.zip` esté en la raíz del `deploy_bundle/`:
```bash
ls -lh project.zip
```

### Error: "requirements.txt: No such file"
El `update.sh` busca `requirements.txt` dentro del ZIP descomprimido. Verifica que el ZIP contenga este archivo:
```bash
unzip -l project.zip | grep requirements.txt
```

### USB montado con noexec
No es necesario hacer nada especial. Los runners detectan esto automáticamente y copian el bundle a `/tmp` antes de ejecutar.

## Logs

- Instalación: salida estándar
- Servicio systemd: `sudo journalctl -u cantina-face -f`
- Aplicación: `deploy/logs/` (si configurado)

## Backups

Los backups se generan automáticamente en `deploy/backups/` con formato:
```
cantina-face-YYYYmmdd-HHMMSS.tgz
```

Excluyen: `venv/`, `data/`, archivos temporales de macOS.

## ChromeOS Flex / Crostini

El instalador detecta automáticamente entornos Crostini y configura:
- Servicio systemd a nivel usuario (`~/.config/systemd/user/`)
- App launcher en ChromeOS (`~/.local/share/applications/`)
- Autostart al abrir Linux

Para inicio automático al encender:
1. Abre `chrome://flags`
2. Busca "Crostini"
3. Activa "Start Linux on login"

## Soporte

Para más detalles, consulta:
- `deploy/README.md`: documentación del paquete deploy
- `deploy/ubuntu/README.md`: instalación específica Ubuntu
- `README.md`: documentación principal del proyecto

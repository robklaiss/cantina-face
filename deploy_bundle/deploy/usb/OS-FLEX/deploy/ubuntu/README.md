# Instalación de Cantina Face — Ubuntu (Intel mini-PC)

Instalación reproducible, sin pasos manuales, con auto-start post-reboot para Ubuntu 24.04+ en mini-PC Intel (KAMRUI).

## Instalación en 1 comando

```bash
sudo bash deploy/ubuntu/install.sh
```

El script se auto-eleva a root si no se ejecuta con `sudo`.

Tras la instalación el equipo **reinicia automáticamente** y el servicio `cantina-face` queda `active (running)`.

### Variables configurables

| Variable | Default | Descripción |
|---|---|---|
| `SILOE_USER` | usuario que invocó sudo | Usuario que ejecuta la app (no root) |
| `CANTINA_PORT` | `8000` | Puerto HTTP de la app |
| `CANTINA_DIR` | `/opt/cantina-face` | Ruta de instalación |
| `NO_REBOOT` | `0` | `1` para no reiniciar al final (debugging) |

Ejemplo con variables personalizadas:

```bash
sudo SILOE_USER=siloe-caja CANTINA_PORT=8080 bash deploy/ubuntu/install.sh
```

### Sin reiniciar al final

```bash
sudo bash deploy/ubuntu/install.sh --no-reboot
# o equivalente:
NO_REBOOT=1 sudo bash deploy/ubuntu/install.sh
```

## Qué hace el instalador

El script `install.sh` ejecuta en orden:

1. **`00_bootstrap_system.sh`** — Paquetes del sistema (Python, build tools, libGL), Google Chrome (repo oficial + fix sandbox), configuración de energía (sin suspensión, sin blank screen).
2. **`10_install_app.sh`** — Copia la app a `/opt/cantina-face`, crea virtualenv `.venv`, instala dependencias Python.
3. **`15_camera_check.sh`** — Verifica cámara, permisos y grupo `video`.
4. **`16_opencv_smoketest.sh`** — Smoke test de captura con OpenCV/Haar.
5. **`20_models.sh`** — Verifica que `models/arcface_r50.onnx` existe y crea symlink `models/mobile_face.onnx → arcface_r50.onnx` (siempre idempotente).
6. **`30_systemd.sh`** — Genera el unit file systemd con `ExecStartPre` (preflight), crea `/etc/default/cantina-face`, habilita e inicia el servicio.
7. **`99_reboot.sh`** — Reinicia el equipo. Tras el reboot, la app sube sola.

### Preflight automático (antes de cada arranque)

El servicio systemd ejecuta `preflight.sh` como `ExecStartPre` antes de iniciar uvicorn. Este script:

- Verifica cámara y modelos (auto-recrea symlink si falta).
- **Migra la base de datos** si detecta la tabla `"transaction"` (palabra reservada SQLite):
  - Hace backup con timestamp: `data/db.sqlite.bak_YYYYMMDD_HHMMSS`
  - Renombra `"transaction"` → `transactions`
  - Crea índices sobre `transactions(student_id)`, `transactions(created_at)`, `transactions(point_of_sale_id)`
- Es **idempotente**: si ya fue migrada, no hace nada.

## Provisionar el modelo

El modelo **NO se descarga de internet** (las URLs son frágiles y fallan con 404). Debe provisionarse offline:

```bash
# Copiar desde un pendrive USB:
sudo cp /media/<usuario>/<USB>/arcface_r50.onnx /opt/cantina-face/models/
sudo chown siloe-caja:siloe-caja /opt/cantina-face/models/arcface_r50.onnx

# Luego ejecutar (o re-ejecutar install.sh):
sudo bash deploy/ubuntu/20_models.sh
```

## Cómo verificar

### Estado del servicio

```bash
systemctl status cantina-face
```

### Ver logs en tiempo real

```bash
journalctl -u cantina-face -f
```

### Probar que la app responde

```bash
curl http://localhost:8000/docs
```

### Ver log de instalación

```bash
cat /var/log/cantina-face-install.log
```

## Cómo desinstalar

```bash
# Detener y deshabilitar servicio
sudo systemctl disable --now cantina-face

# Eliminar archivos
sudo rm -f /etc/systemd/system/cantina-face.service
sudo rm -f /etc/default/cantina-face
sudo rm -rf /opt/cantina-face

# Recargar systemd
sudo systemctl daemon-reload
```

## Cómo re-instalar / actualizar

El instalador es **idempotente**: re-ejecutar no rompe nada. Solo sincroniza archivos nuevos, recrea el venv si falta, y reinicia el servicio.

```bash
cd /ruta/al/repo
git pull
sudo bash deploy/ubuntu/install.sh
```

## Estructura de archivos

```
deploy/ubuntu/
├── _common.sh              # Variables y funciones compartidas
├── 00_bootstrap_system.sh  # Paquetes, Chrome, energía
├── 10_install_app.sh       # Copiar app, venv, deps
├── 15_camera_check.sh      # Verificar cámara y permisos
├── 16_opencv_smoketest.sh  # Smoke test OpenCV
├── 20_models.sh            # Symlink del modelo ONNX
├── 30_systemd.sh           # Servicio systemd
├── 99_reboot.sh            # Reiniciar equipo
├── preflight.sh            # Preflight: validaciones + migración DB
├── cantina-face.service    # Template del unit file (referencia)
├── smoketest_face.py       # Script Python de smoke test
├── install.sh              # Entrypoint único
└── README.md               # Esta documentación
```

## Notas

- **Auto-elevación**: si ejecutás `bash deploy/ubuntu/install.sh` sin `sudo`, el script se re-ejecuta automáticamente con `sudo`.
- **Modo Live**: el instalador detecta si estás en modo Live (USB) y aborta con un mensaje claro. Primero instalá Ubuntu en el disco.
- **Chrome sandbox**: se aplica automáticamente `chown root:root` + `chmod 4755` al sandbox de Chrome.
- **Energía**: se desactiva suspensión tanto en GNOME (gsettings) como en logind (IdleAction=ignore). Se enmascaran los targets de sleep/suspend/hibernate.
- **Migración DB**: la tabla `transaction` (palabra reservada SQLite) se renombra automáticamente a `transactions` en el preflight. Esto es idempotente y seguro.
- **Uvicorn**: en producción (systemd) corre **sin `--reload`**. Para desarrollo, usá `deploy/run.sh` con `RELOAD=1`.

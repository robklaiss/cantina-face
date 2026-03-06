# Deploy Bundle - Cantina Face

Este bundle contiene todo lo necesario para instalar o actualizar Cantina Face en la máquina caja.

**NUEVO:** Ahora incluye instalación automática de Cloudflare Tunnel para acceso remoto seguro. Funciona **incluso con USB montado en noexec** o sistemas de archivos exFAT/FAT sin soporte de symlinks.

## 🚀 Quick Start (Comando Único)

### En la caja Ubuntu:

```bash
# Actualización (recomendado):
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh

# Instalación inicial (primera vez):
bash /media/$USER/OS-FLEX/deploy_bundle/run_install.sh
```

**Eso es todo.** El script copia el bundle a disco local y ejecuta automáticamente.

---

## 📁 Estructura del Bundle

```
deploy_bundle/
├── run_update.sh          # Runner principal para actualización
├── run_install.sh         # Runner principal para instalación inicial
├── project.zip            # Código fuente de la aplicación (app.py, static/, etc.)
├── README_DEPLOY.md       # Este archivo
├── models/                # Modelos de IA (obligatorio)
│   └── arcface_r50.onnx   # Modelo de reconocimiento facial (~166MB)
└── deploy/                # Scripts de deploy
    ├── update.sh          # Script de actualización (llamado por run_update.sh)
    ├── install.sh         # Script de instalación (llamado por run_install.sh)
    ├── run.sh             # Script para iniciar servidor
    ├── backup_faces.sh    # Backup de datos faciales
    ├── guardrails/        # Validaciones pre-deploy
    │   └── check_python_hardcode.sh
    └── ubuntu/            # Scripts específicos de Ubuntu
        └── *.sh, *.service
```

**IMPORTANTE:** El bundle NO contiene `app.py`, `venv/`, ni `static/` en la raíz. Esos archivos están dentro de `project.zip`. Los modelos van SOLO en `models/`.

---

## 🔧 Cómo Funciona (Noexec-Safe)

Los runners (`run_update.sh` y `run_install.sh`) son **noexec-safe**:

1. **Copian TODO el bundle** desde el USB a `/opt/cantina-face-deploy`
2. **Ejecutan los scripts** desde el disco local (evitando noexec)
3. **Crean el venv** en disco local (evitando problemas de symlinks en exFAT)
4. **Permiten desconectar** el USB de forma segura después

**Nunca** intentan ejecutar Python o crear venvs directamente en el USB.

---

## 📋 Preparación del USB (en Mac/Linux de desarrollo)

### Opción 1: Makefile (recomendado)

```bash
# En el repo base:
make deploy-bundle-usb USB=/Volumes/OS-FLEX
```

Esto construye, valida y copia el bundle a `OS-FLEX/deploy_bundle/`.

### Opción 2: Manual

```bash
# Construir bundle
make deploy-bundle

# Validar
make deploy-bundle-validate

# Copiar a USB
rsync -av --delete \
  --exclude '.DS_Store' \
  --exclude '._*' \
  dist/deploy_bundle/ /Volumes/OS-FLEX/deploy_bundle/
```

---

## 🖥️ En la Máquina Ubuntu

### Instalación Inicial (primera vez)

```bash
bash /media/$USER/OS-FLEX/deploy_bundle/run_install.sh
```

Esto:
- Copia el bundle a `/opt/cantina-face-deploy`
- Descomprime `project.zip`
- Crea el venv
- Instala dependencias
- Configura el sistema

### Actualización (versiones posteriores)

```bash
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
```

Esto:
- Copia el bundle a `/opt/cantina-face-deploy`
- Hace backup de la versión actual
- Descomprime el nuevo `project.zip`
- Instala modelos en `/opt/cantina-face/models/`
- Preserva `data/` (DB, índices faciales)
- Reinstala dependencias

---

## ✅ Requisitos

- **Ubuntu 24.04** (o compatible)
- **Python 3** (se instala automáticamente si falta)
- **rsync** (se instala automáticamente si falta)
- **Conexión a internet** (para descargar dependencias Python)

---

## 🐛 Solución de Problemas

### "Permission denied" al ejecutar

```bash
# Ejecutar con bash explícitamente (no requiere +x):
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
```

### USB montado con noexec

✅ **No hay problema.** Los runners detectan esto automáticamente y copian el bundle a `/opt/cantina-face-deploy` antes de ejecutar.

### Sistema de archivos exFAT/FAT

✅ **No hay problema.** El venv se crea en disco local, no en el USB.

### "run_update.sh no existe"

Verifica la estructura en el USB:
```bash
ls -la /media/$USER/OS-FLEX/deploy_bundle/
```

Debe mostrar:
- `run_update.sh`
- `run_install.sh`
- `project.zip`
- `deploy/`

**NO** debe estar anidado como `OS-FLEX/deploy_bundle/deploy_bundle/`.

### "No se encontró project.zip"

```bash
# Verificar que el ZIP existe y no está vacío:
ls -lh /media/$USER/OS-FLEX/deploy_bundle/project.zip
```

Debe tener al menos 100KB.

---

## 🎯 Después de la Instalación

### Iniciar el servidor manualmente:

```bash
bash /opt/cantina-face-deploy/deploy/run.sh
```

### Configurar autostart (opcional):

```bash
bash /opt/cantina-face-deploy/deploy/update.sh --setup-autostart
```

### Configurar acceso remoto (Cloudflare Tunnel):

**NUEVO:** El sistema ahora instala automáticamente `cloudflared` para permitir acceso remoto seguro.

```bash
# Configurar túnel (solo primera vez)
bash /opt/cantina-face-deploy/deploy/setup_cloudflare_tunnel.sh
```

Esto permite que el backend en HostGator se comunique con la máquina caja sin necesidad de:
- Abrir puertos en el router
- Configurar IP pública
- Configurar certificados SSL

Ver documentación completa en: `README_CLOUDFLARE_TUNNEL.md`

### Ver logs del servicio:

```bash
sudo journalctl -u cantina-face -f
```

---

## 💾 Backups

Los backups se generan automáticamente en `/opt/cantina-face-deploy/deploy/backups/` con formato:

```
cantina-face-20260210-143022.tgz
```

Excluyen: `venv/`, `data/`, archivos temporales de macOS.

---

## 📝 Notas Importantes

- ✅ El bundle local queda en `/opt/cantina-face-deploy`
- ✅ Los datos faciales (DB, índices) se preservan entre actualizaciones
- ✅ El USB puede desconectarse después de la instalación/actualización
- ✅ Los runners siempre usan `bash` explícitamente (no dependen de shebang)
- ✅ No se crea basura de macOS (`.DS_Store`, `._*`, `__MACOSX`)

---

## 🆘 Soporte

Para más detalles, consulta:
- `deploy/README.md`: documentación del paquete deploy
- `deploy/ubuntu/README.md`: instalación específica Ubuntu
- `README.md`: documentación principal del proyecto

---

**Generado por:** `tools/build_deploy_bundle.sh`  
**Validado por:** `tools/validate_deploy_bundle.sh`

# Deploy Bundle - Solución Completa USB/noexec

## ✅ Problema Resuelto

Este bundle elimina TODOS los problemas históricos de deploy en USB con noexec/exFAT:

- ❌ "run_update.sh no existe" → ✅ Validación automática en build
- ❌ Rutas REPO_DIR mal calculadas → ✅ Rutas determinísticas con SCRIPT_DIR
- ❌ Guardrail "no encontrado" por `-x` en USB → ✅ Usa `bash` en lugar de `-x`
- ❌ venv fallando en USB (symlink lib → lib64) → ✅ venv siempre en disco local
- ❌ Carpetas anidadas raras → ✅ Estructura plana validada
- ❌ Basura macOS `._*` y `.DS_Store` → ✅ Limpieza automática

## 📁 Estructura del Bundle

```
OS-FLEX/deploy_bundle/          ← Copiar esto al USB
├── run_update.sh               ← Runner para actualización
├── run_install.sh              ← Runner para instalación
├── project.zip                 ← Código fuente (112KB)
├── README_DEPLOY.md            ← Documentación
└── deploy/                     ← Scripts internos
    ├── update.sh               ← Script de actualización
    ├── install.sh              ← Script de instalación
    ├── run.sh                  ← Iniciar servidor
    ├── backup_faces.sh         ← Backup de datos
    ├── guardrails/             ← Validaciones
    │   └── check_python_hardcode.sh
    └── ubuntu/                 ← Scripts Ubuntu
        └── *.sh, *.service
```

## 🛠️ Componentes Creados

### 1. Builder (`tools/build_deploy_bundle.sh`)

**Función:** Construye el bundle completo con validación automática.

**Características:**
- Copia estructura `deploy/` desde `deploy_bundle/deploy`
- Crea `project.zip` desde `project/` (sin venv, data, basura)
- Genera runners `run_update.sh` y `run_install.sh`
- Elimina archivos macOS (`.DS_Store`, `._*`, `__MACOSX`)
- Valida que existen todos los archivos críticos
- Falla con `exit 1` si falta algo

**Uso:**
```bash
make deploy-bundle
# o
bash tools/build_deploy_bundle.sh
```

### 2. Runners (noexec-safe)

#### `run_update.sh`
- Detecta ubicación en USB
- Copia TODO a `/opt/cantina-face-deploy`
- Ejecuta `deploy/update.sh` desde disco local
- Permite desconectar USB después

#### `run_install.sh`
- Detecta ubicación en USB
- Copia TODO a `/opt/cantina-face-deploy`
- Descomprime `project.zip` en disco local
- Ejecuta `deploy/install.sh` desde disco local
- Permite desconectar USB después

### 3. Scripts Refactorizados

#### `deploy/update.sh`
**Cambios:**
- ✅ Rutas determinísticas: `SCRIPT_DIR`, `ROOT_DIR`, `DEPLOY_DIR`
- ✅ Guardrail con `bash` en lugar de `-x`
- ✅ `requirements.txt` desde ZIP descomprimido con fallback
- ✅ Elimina dependencia de `REPO_DIR/..`

#### `deploy/install.sh`
**Cambios:**
- ✅ Rutas determinísticas: `SCRIPT_DIR`, `ROOT_DIR`, `DEPLOY_DIR`
- ✅ Guardrail con `bash` en lugar de `-x`
- ✅ `requirements.txt` extraído desde `project.zip` si es necesario
- ✅ venv siempre en `$ROOT_DIR/venv` (disco local)

#### `deploy/guardrails/check_python_hardcode.sh`
**Cambios:**
- ✅ Excluye `guardrails/**` completo (no solo el archivo)
- ✅ Funciona con `bash` (no requiere `-x`)

## 🚀 Quick Start

### Construir Bundle
```bash
make deploy-bundle
```

### Copiar al USB
```bash
# Copiar dist/deploy_bundle/ al USB en:
OS-FLEX/deploy_bundle/
```

### En Ubuntu 24.04

**Primera instalación:**
```bash
bash /media/$USER/OS-FLEX/deploy_bundle/run_install.sh
```

**Actualizaciones:**
```bash
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
```

## 🔍 Validación Automática

El builder valida:
```bash
✅ run_update.sh
✅ run_install.sh
✅ project.zip
✅ README_DEPLOY.md
✅ deploy/update.sh
✅ deploy/install.sh
✅ deploy/run.sh
✅ deploy/backup_faces.sh
✅ deploy/guardrails/check_python_hardcode.sh
```

Si falta algo → `exit 1`

## 🎯 Criterio de Aceptación

En Ubuntu 24.04 con USB noexec/exFAT:

```bash
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
```

**Debe funcionar sin:**
- Hacks manuales
- Crear venv en USB
- Problemas de permisos
- Errores de "archivo no encontrado"
- Problemas de symlinks

## 📝 Archivos Adicionales

### `Makefile`
```makefile
make deploy-bundle  # Construir bundle
make clean-bundle   # Limpiar bundle
make help           # Ayuda
```

### `.windsurf/workflows/deploy-usb.md`
Workflow completo documentado para uso futuro.

### `README_DEPLOY.md` (generado automáticamente)
Documentación incluida en el bundle para el usuario final.

## 🔧 Cómo Funciona (Internamente)

### Flujo de Actualización

```
1. Usuario ejecuta: bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
   ↓
2. run_update.sh detecta: SCRIPT_DIR=/media/$USER/OS-FLEX/deploy_bundle
   ↓
3. Copia TODO a: /opt/cantina-face-deploy (con rsync)
   ↓
4. cd /opt/cantina-face-deploy
   ↓
5. bash ./deploy/update.sh ./project.zip
   ↓
6. update.sh:
   - Calcula rutas: SCRIPT_DIR=deploy/, ROOT_DIR=/opt/cantina-face-deploy
   - Ejecuta guardrail: bash ./deploy/guardrails/check_python_hardcode.sh
   - Descomprime project.zip a tmp
   - Hace backup de data/
   - Sincroniza archivos con rsync
   - Restaura data/
   - Crea/actualiza venv en /opt/cantina-face-deploy/venv
   - Instala requirements.txt desde ZIP descomprimido
   ↓
7. ✅ Actualización completa
   ↓
8. Usuario puede desconectar USB
```

### Rutas Determinísticas

**Antes (problemático):**
```bash
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)}"
DEPLOY_DIR="${DEPLOY_DIR:-$ROOT_DIR/deploy}"
```
❌ Si ejecutas desde `deploy/`, ROOT_DIR apunta a `deploy/` (mal)

**Después (correcto):**
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DEPLOY_DIR="$SCRIPT_DIR"
```
✅ SCRIPT_DIR siempre es donde está el script
✅ ROOT_DIR siempre es el padre de SCRIPT_DIR
✅ DEPLOY_DIR siempre es SCRIPT_DIR (para deploy/*.sh)

## 🧪 Testing

### Validar sintaxis de todos los scripts:
```bash
bash -n dist/deploy_bundle/run_update.sh
bash -n dist/deploy_bundle/run_install.sh
bash -n dist/deploy_bundle/deploy/update.sh
bash -n dist/deploy_bundle/deploy/install.sh
```

### Validar guardrail:
```bash
bash dist/deploy_bundle/deploy/guardrails/check_python_hardcode.sh
```

### Validar estructura:
```bash
ls -la dist/deploy_bundle/
```

## 📦 Entregables

1. ✅ `tools/build_deploy_bundle.sh` - Builder con validación
2. ✅ `run_update.sh` - Runner para actualización (generado)
3. ✅ `run_install.sh` - Runner para instalación (generado)
4. ✅ `deploy/update.sh` - Refactorizado con rutas determinísticas
5. ✅ `deploy/install.sh` - Refactorizado con rutas determinísticas
6. ✅ `deploy/guardrails/check_python_hardcode.sh` - Actualizado
7. ✅ `README_DEPLOY.md` - Generado automáticamente
8. ✅ `Makefile` - Target `make deploy-bundle`
9. ✅ `.windsurf/workflows/deploy-usb.md` - Workflow documentado

## 🎉 Resultado Final

**Bundle generado en:** `dist/deploy_bundle/`

**Tamaño:** ~112KB (project.zip) + scripts

**Compatible con:**
- ✅ Ubuntu 24.04
- ✅ USB montado con noexec
- ✅ Sistemas de archivos exFAT/FAT (sin symlinks)
- ✅ ChromeOS Flex (Crostini)

**No requiere:**
- ❌ Permisos de ejecución en USB
- ❌ Soporte de symlinks en USB
- ❌ Hacks manuales
- ❌ Edición de scripts

**Funciona siempre:**
```bash
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
```

## 🔄 Mantenimiento Futuro

### Actualizar el bundle:
1. Modificar código en `project/`
2. Ejecutar: `make deploy-bundle`
3. Copiar `dist/deploy_bundle/` al USB
4. Listo

### Agregar nuevos scripts:
1. Agregar en `deploy_bundle/deploy/`
2. Actualizar `REQUIRED_FILES` en `tools/build_deploy_bundle.sh` si es crítico
3. Ejecutar: `make deploy-bundle`

### Debugging:
- Logs en: `/opt/cantina-face-deploy/deploy/backups/`
- Venv en: `/opt/cantina-face-deploy/venv/`
- Datos en: `/opt/cantina-face-deploy/data/`

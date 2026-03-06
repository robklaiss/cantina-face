# Deploy Bundle - Solución Implementada ✅

## Problema Original

El USB contenía una carpeta `deploy_bundle/` con estructura incorrecta:
- ❌ Contenía `app.py`, `static/`, `venv/` en la raíz (contaminación)
- ❌ NO contenía `run_update.sh` / `run_install.sh` funcionales
- ❌ La caja no podía hacer "one-command" deploy
- ❌ Basura de macOS (`.DS_Store`, `._*`, `__MACOSX`)

## Solución Implementada

### ✅ Bundle Real con Estructura Correcta

```
dist/deploy_bundle/
├── run_update.sh          # ⭐ Runner noexec-safe para actualización
├── run_install.sh         # ⭐ Runner noexec-safe para instalación
├── project.zip            # Código de la app (112KB)
├── README_DEPLOY.md       # Documentación para la caja
└── deploy/                # Scripts de deploy
    ├── update.sh
    ├── install.sh
    ├── run.sh
    ├── backup_faces.sh
    ├── guardrails/check_python_hardcode.sh
    └── ubuntu/*.sh
```

**Validado:** Sin `app.py`, `venv/`, `static/`, `.DS_Store` en la raíz ✅

---

## Herramientas Creadas

### 1. `tools/build_deploy_bundle.sh`
Construye el bundle completo en `dist/deploy_bundle/`:
- Copia estructura `deploy/`
- Crea `project.zip` desde `project/` (excluyendo venv, data, __pycache__)
- Genera runners noexec-safe
- Limpia basura de macOS
- Valida archivos críticos

### 2. `tools/validate_deploy_bundle.sh`
Valida la estructura del bundle (exit 1 si falla):
- ✅ Existen archivos requeridos (runners, ZIP, scripts)
- ✅ NO existen archivos prohibidos (app.py, venv en raíz)
- ✅ NO existe basura de macOS
- ✅ project.zip > 10KB (no vacío)
- ✅ Runners son ejecutables

### 3. `tools/test_deploy_bundle.sh`
Test completo del workflow:
- Build → Validate → Simulate USB copy
- Verifica estructura final
- Detecta contaminación y basura

### 4. `Makefile` - Targets

```bash
# Construir bundle
make deploy-bundle

# Validar bundle
make deploy-bundle-validate

# Test completo (build + validate + simulate)
make test-deploy-bundle

# ⭐ TODO-EN-UNO: Construir + validar + copiar a USB
make deploy-bundle-usb USB=/Volumes/OS-FLEX

# Limpiar bundle
make clean-bundle

# Ayuda
make help
```

---

## Runners Noexec-Safe

### `run_update.sh` y `run_install.sh`

**Características:**
- ✅ Copian TODO el bundle desde USB a `/opt/cantina-face-deploy`
- ✅ Ejecutan desde disco local (evitan noexec)
- ✅ Usan `bash` explícitamente (no dependen de shebang)
- ✅ Nunca crean venv en USB
- ✅ Permiten desconectar USB después

**Flujo:**
1. Verifican que `project.zip` y `deploy/` existen
2. Copian bundle con `rsync -a --delete` a `/opt/cantina-face-deploy`
3. Ejecutan `deploy/update.sh` o `deploy/install.sh` desde disco local
4. Descomprimen `project.zip` en disco local
5. Crean venv en disco local
6. Preservan `data/` entre actualizaciones

---

## Workflow Completo

### En Mac (Desarrollo)

```bash
# Comando único para preparar USB:
make deploy-bundle-usb USB=/Volumes/OS-FLEX

# Verificar en USB:
ls -la /Volumes/OS-FLEX/deploy_bundle/
# Debe mostrar: run_update.sh, run_install.sh, project.zip, deploy/
```

### En Ubuntu (Caja)

```bash
# Actualización (comando único):
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh

# Instalación inicial (primera vez):
bash /media/$USER/OS-FLEX/deploy_bundle/run_install.sh
```

**Eso es todo.** No más hacks, no más pasos manuales.

---

## Validaciones Implementadas

### Build-time
- ✅ Verifica que `deploy_bundle/deploy/` existe
- ✅ Crea ZIP excluyendo venv, data, __pycache__
- ✅ Limpia basura de macOS
- ✅ Valida archivos críticos antes de terminar

### Validation-time
- ✅ 9 archivos requeridos verificados
- ✅ 8 archivos prohibidos verificados (NO deben existir)
- ✅ 3 patrones de basura macOS verificados
- ✅ Tamaño de ZIP verificado (> 10KB)
- ✅ Permisos de runners verificados

### Runtime (en runners)
- ✅ Verifican `project.zip` existe
- ✅ Verifican `deploy/update.sh` o `deploy/install.sh` existen
- ✅ Copian bundle completo antes de ejecutar
- ✅ Ejecutan guardrails antes de instalar/actualizar

---

## Acceptance Test

### Test Automatizado

```bash
# En Mac:
make test-deploy-bundle
# Resultado: ✅ ALL TESTS PASSED
```

### Test Manual en Mac

```bash
# 1. Limpiar y construir
make clean-bundle
make deploy-bundle

# 2. Validar
make deploy-bundle-validate
# Resultado: ✅ Validation PASSED

# 3. Verificar estructura
ls -la dist/deploy_bundle/
# Debe mostrar: run_update.sh, run_install.sh, project.zip, deploy/
# NO debe mostrar: app.py, venv/, static/, .DS_Store

# 4. Copiar a USB
make deploy-bundle-usb USB=/Volumes/OS-FLEX

# 5. Verificar en USB
ls -la /Volumes/OS-FLEX/deploy_bundle/
# Debe mostrar la misma estructura limpia
```

### Test en Ubuntu (Caja)

```bash
# 1. Verificar bundle en USB
ls -la /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
# Debe existir

# 2. Ejecutar actualización
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh

# 3. Verificar copia local
ls -la /opt/cantina-face-deploy/
# Debe mostrar: run_update.sh, project.zip, deploy/, app.py, static/, venv/

# 4. Verificar servidor
bash /opt/cantina-face-deploy/deploy/run.sh
# Debe iniciar en http://localhost:8000
```

---

## Resultados de Validación Real

```
============================================
Validating deploy_bundle structure
============================================

[1/3] Checking required files...
  ✅ Found: run_update.sh
  ✅ Found: run_install.sh
  ✅ Found: project.zip
  ✅ Found: README_DEPLOY.md
  ✅ Found: deploy/update.sh
  ✅ Found: deploy/install.sh
  ✅ Found: deploy/run.sh
  ✅ Found: deploy/backup_faces.sh
  ✅ Found: deploy/guardrails/check_python_hardcode.sh

[2/3] Checking for forbidden items (contamination)...
  ✅ Clean: app.py not found
  ✅ Clean: config.py not found
  ✅ Clean: face_engine.py not found
  ✅ Clean: venv not found
  ✅ Clean: static not found
  ✅ Clean: scripts not found
  ✅ Clean: data not found
  ✅ Clean: __pycache__ not found

[3/3] Checking for macOS junk files...
  ✅ No macOS junk files found

[Bonus] Checking project.zip size...
  ✅ project.zip size: 112418 bytes

[Bonus] Checking runner permissions...
  ✅ run_update.sh is executable
  ✅ run_install.sh is executable

============================================
✅ Validation PASSED
============================================
```

---

## Documentación Generada

### Para Desarrollo
- `DEPLOY_BUNDLE_GUIDE.md` - Guía completa del sistema (444 líneas)
- `tools/build_deploy_bundle.sh` - Script de construcción (472 líneas)
- `tools/validate_deploy_bundle.sh` - Script de validación (143 líneas)
- `tools/test_deploy_bundle.sh` - Script de testing (nuevo)
- `Makefile` - Targets actualizados con `test-deploy-bundle`

### Para la Caja
- `README_DEPLOY.md` - Generado automáticamente en el bundle
- Comandos exactos para Ubuntu
- Troubleshooting completo

---

## Comandos de Referencia Rápida

```bash
# Desarrollo (Mac):
make deploy-bundle-usb USB=/Volumes/OS-FLEX

# Caja (Ubuntu):
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
```

---

## Estado Final

✅ **Bundle real generado** en `dist/deploy_bundle/`  
✅ **Validación automática** implementada  
✅ **Runners noexec-safe** funcionando  
✅ **Comando único** para USB export  
✅ **Sin basura de macOS**  
✅ **Sin contaminación** (app.py, venv fuera de raíz)  
✅ **Tests pasando** (build + validate + simulate)  
✅ **Documentación completa**  

**El bundle está listo para producción.** 🚀

---

**Fecha:** 2026-02-10  
**Versión:** 1.0.0  
**Status:** ✅ COMPLETADO

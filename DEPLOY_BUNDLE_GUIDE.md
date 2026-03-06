# Deploy Bundle - Guía Completa

Esta guía documenta el sistema completo de generación, validación y despliegue del bundle USB para Cantina Face.

## 🎯 Problema Resuelto

**Antes:** El USB contenía una carpeta `deploy_bundle/` con `app.py`, `venv/`, y archivos del proyecto mezclados con los runners, causando confusión y fallos en el despliegue.

**Ahora:** El bundle tiene una estructura limpia y validada:
- ✅ Runners (`run_update.sh`, `run_install.sh`) en la raíz
- ✅ Código de la app empaquetado en `project.zip`
- ✅ Scripts de deploy en `deploy/`
- ✅ Sin basura de macOS (`.DS_Store`, `._*`, `__MACOSX`)
- ✅ Sin contaminación (`app.py`, `venv/`, etc. NO están en la raíz)

---

## 📁 Estructura del Bundle

```
dist/deploy_bundle/                    # Bundle generado (listo para USB)
├── run_update.sh                      # ⭐ Runner principal para actualización
├── run_install.sh                     # ⭐ Runner principal para instalación
├── project.zip                        # Código de la app (app.py, static/, etc.)
├── README_DEPLOY.md                   # Documentación para la caja
└── deploy/                            # Scripts de deploy
    ├── update.sh                      # Script de actualización
    ├── install.sh                     # Script de instalación
    ├── run.sh                         # Lanzador del servidor
    ├── backup_faces.sh                # Backup de datos faciales
    ├── guardrails/                    # Validaciones pre-deploy
    │   └── check_python_hardcode.sh
    └── ubuntu/                        # Scripts específicos Ubuntu
        └── *.sh, *.service
```

**IMPORTANTE:** El bundle NO contiene `app.py`, `venv/`, ni `static/` en la raíz. Esos archivos están dentro de `project.zip`.

---

## 🛠️ Herramientas Disponibles

### 1. `tools/build_deploy_bundle.sh`

**Propósito:** Construir el bundle completo en `dist/deploy_bundle/`.

**Qué hace:**
1. Limpia el bundle anterior
2. Copia `deploy_bundle/deploy/` a `dist/deploy_bundle/deploy/`
3. Crea `project.zip` desde `project/` (excluyendo `venv/`, `data/`, `__pycache__/`)
4. Genera runners noexec-safe (`run_update.sh`, `run_install.sh`)
5. Limpia basura de macOS (`.DS_Store`, `._*`, `__MACOSX`)
6. Genera `README_DEPLOY.md` con instrucciones para la caja
7. Valida que existan todos los archivos críticos

**Uso:**
```bash
bash tools/build_deploy_bundle.sh
# o
make deploy-bundle
```

### 2. `tools/validate_deploy_bundle.sh`

**Propósito:** Validar la estructura del bundle antes de copiarlo al USB.

**Validaciones:**
- ✅ Existen todos los archivos requeridos (runners, ZIP, scripts)
- ✅ NO existen archivos prohibidos (`app.py`, `venv/`, etc. en raíz)
- ✅ NO existe basura de macOS
- ✅ `project.zip` no está vacío (> 10KB)
- ✅ Los runners son ejecutables

**Uso:**
```bash
bash tools/validate_deploy_bundle.sh
# o
make deploy-bundle-validate
```

**Exit codes:**
- `0`: Validación exitosa
- `1`: Validación fallida (muestra errores específicos)

### 3. `Makefile` - Targets Disponibles

#### `make deploy-bundle`
Construye el bundle en `dist/deploy_bundle/`.

#### `make deploy-bundle-validate`
Valida la estructura del bundle.

#### `make deploy-bundle-usb USB=/path/to/usb`
**Comando todo-en-uno:** Construye + valida + copia al USB.

**Ejemplos:**
```bash
# En Mac:
make deploy-bundle-usb USB=/Volumes/OS-FLEX

# En Linux:
make deploy-bundle-usb USB=/media/$USER/OS-FLEX
```

**Qué hace:**
1. Ejecuta `tools/build_deploy_bundle.sh`
2. Ejecuta `tools/validate_deploy_bundle.sh`
3. Verifica que el USB existe
4. Copia el bundle a `USB/deploy_bundle/` con `rsync`
5. Excluye basura de macOS durante la copia
6. Muestra comandos para ejecutar en Ubuntu

#### `make clean-bundle`
Elimina `dist/deploy_bundle/`.

#### `make help`
Muestra todos los targets disponibles.

---

## 🚀 Workflow Completo

### En Mac/Linux (Desarrollo)

```bash
# 1. Construir, validar y copiar al USB (un solo comando):
make deploy-bundle-usb USB=/Volumes/OS-FLEX

# 2. Verificar en el USB:
ls -la /Volumes/OS-FLEX/deploy_bundle/
# Debe mostrar: run_update.sh, run_install.sh, project.zip, deploy/
```

### En Ubuntu (Caja)

```bash
# Actualización (recomendado):
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh

# Instalación inicial (primera vez):
bash /media/$USER/OS-FLEX/deploy_bundle/run_install.sh
```

**Eso es todo.** Los runners copian el bundle a `/opt/cantina-face-deploy` y ejecutan automáticamente.

---

## 🔧 Cómo Funcionan los Runners (Noexec-Safe)

### `run_update.sh` y `run_install.sh`

Ambos runners siguen el mismo patrón:

1. **Verifican** que estén en el bundle correcto (existen `project.zip` y `deploy/`)
2. **Copian TODO el bundle** desde el USB a `/opt/cantina-face-deploy`
3. **Ejecutan** los scripts desde el disco local (evitando noexec)
4. **Permiten desconectar** el USB de forma segura después

**Características clave:**
- ✅ Usan `bash` explícitamente (no dependen de shebang o permisos +x)
- ✅ Copian con `rsync -a --delete` (preserva permisos, elimina archivos viejos)
- ✅ Usan `sudo` solo para crear `/opt/cantina-face-deploy`
- ✅ Cambian ownership a `$USER` después de copiar
- ✅ Nunca crean venvs ni ejecutan Python en el USB

### Diferencias entre `run_install.sh` y `run_update.sh`

**`run_install.sh`:**
- Descomprime `project.zip` en `/opt/cantina-face-deploy`
- Crea el venv desde cero
- Instala todas las dependencias
- Configura el sistema (opcional: `--setup-autostart`)

**`run_update.sh`:**
- Hace backup de la versión actual
- Descomprime el nuevo `project.zip`
- Preserva `data/` (DB, índices faciales)
- Reinstala dependencias (por si cambiaron)

---

## ✅ Validaciones Implementadas

### Build-time (en `build_deploy_bundle.sh`)

1. Verifica que `deploy_bundle/deploy/` existe
2. Crea `project.zip` excluyendo `venv/`, `data/`, `__pycache__/`
3. Limpia basura de macOS después de crear el bundle
4. Valida que existan todos los archivos críticos antes de terminar

### Validation-time (en `validate_deploy_bundle.sh`)

1. **Archivos requeridos:**
   - `run_update.sh`
   - `run_install.sh`
   - `project.zip`
   - `README_DEPLOY.md`
   - `deploy/update.sh`
   - `deploy/install.sh`
   - `deploy/run.sh`
   - `deploy/backup_faces.sh`
   - `deploy/guardrails/check_python_hardcode.sh`

2. **Archivos prohibidos (NO deben existir en raíz):**
   - `app.py`
   - `config.py`
   - `face_engine.py`
   - `venv/`
   - `static/`
   - `scripts/`
   - `data/`
   - `__pycache__/`

3. **Basura de macOS (NO debe existir):**
   - `.DS_Store`
   - `._*`
   - `__MACOSX/`

4. **Validaciones adicionales:**
   - `project.zip` > 10KB (no vacío)
   - Runners son ejecutables

### Runtime (en los runners)

1. Verifican que `project.zip` existe
2. Verifican que `deploy/update.sh` o `deploy/install.sh` existen
3. Copian el bundle completo antes de ejecutar
4. Ejecutan guardrails antes de instalar/actualizar

---

## 🐛 Troubleshooting

### "Bundle directory does not exist"

```bash
# Construir el bundle primero:
make deploy-bundle
```

### "Validation FAILED"

El script muestra exactamente qué archivos faltan o están mal. Ejemplo:

```
❌ Missing: deploy/guardrails/check_python_hardcode.sh
❌ Found forbidden item: app.py (should NOT be in bundle root)
❌ Found macOS junk: .DS_Store
```

**Solución:** Corregir el problema y ejecutar `make deploy-bundle` de nuevo.

### "USB path does not exist"

```bash
# Verificar que el USB está montado:
ls -la /Volumes/           # Mac
ls -la /media/$USER/       # Linux

# Usar la ruta correcta:
make deploy-bundle-usb USB=/Volumes/NOMBRE-CORRECTO
```

### "project.zip is too small"

El ZIP está vacío o corrupto. Verificar que `project/` tiene contenido:

```bash
ls -la project/
# Debe mostrar: app.py, static/, scripts/, requirements.txt, etc.
```

### Bundle contiene `app.py` en la raíz

Esto NO debería pasar si usas `make deploy-bundle`. Si pasa:

1. Verificar que no estás copiando manualmente archivos incorrectos
2. Ejecutar `make clean-bundle && make deploy-bundle`
3. Validar con `make deploy-bundle-validate`

---

## 📋 Acceptance Test

### En Mac:

```bash
# 1. Limpiar y construir desde cero:
make clean-bundle
make deploy-bundle

# 2. Validar:
make deploy-bundle-validate
# Debe mostrar: ✅ Validation PASSED

# 3. Verificar estructura manualmente:
ls -la dist/deploy_bundle/
# Debe mostrar: run_update.sh, run_install.sh, project.zip, deploy/

# NO debe mostrar: app.py, venv/, static/, .DS_Store

# 4. Copiar al USB:
make deploy-bundle-usb USB=/Volumes/OS-FLEX

# 5. Verificar en el USB:
ls -la /Volumes/OS-FLEX/deploy_bundle/
# Debe mostrar la misma estructura
```

### En Ubuntu (Caja):

```bash
# 1. Verificar que el bundle existe en el USB:
ls -la /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
# Debe existir

# 2. Ejecutar actualización:
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh

# 3. Verificar que se copió a disco local:
ls -la /opt/cantina-face-deploy/
# Debe mostrar: run_update.sh, project.zip, deploy/, app.py, static/, venv/

# 4. Verificar que el servidor arranca:
bash /opt/cantina-face-deploy/deploy/run.sh
# Debe iniciar en http://localhost:8000
```

---

## 🔐 Seguridad y Limpieza

### Archivos excluidos del `project.zip`

- `venv/` - Entorno virtual (se crea en destino)
- `data/` - Datos faciales (se preservan en actualizaciones)
- `__pycache__/` - Bytecode Python
- `.git/` - Historial Git
- `.DS_Store`, `._*` - Basura de macOS

### Limpieza automática

El script `build_deploy_bundle.sh` ejecuta:

```bash
find "$BUNDLE_DIR" -name '.DS_Store' -delete
find "$BUNDLE_DIR" -name '._*' -delete
find "$BUNDLE_DIR" -name '__MACOSX' -type d -prune -exec rm -rf {} +
```

El Makefile también excluye estos archivos al copiar al USB:

```bash
rsync -av --delete \
  --exclude '.DS_Store' \
  --exclude '._*' \
  --exclude '__MACOSX' \
  dist/deploy_bundle/ "$USB/deploy_bundle/"
```

---

## 📝 Notas Importantes

1. **El bundle es autocontenido:** Todo lo necesario está en `dist/deploy_bundle/`
2. **El ZIP es parte del bundle:** `project.zip` se genera automáticamente
3. **Los runners son noexec-safe:** Funcionan incluso con USB en noexec
4. **La validación es obligatoria:** `make deploy-bundle-usb` valida antes de copiar
5. **La estructura es determinística:** Siempre la misma, sin sorpresas

---

## 🔄 Flujo de Datos

```
Repo base (cantina-face/)
  ├── project/              → Se empaqueta en project.zip
  │   ├── app.py
  │   ├── static/
  │   ├── scripts/
  │   └── requirements.txt
  │
  └── deploy_bundle/deploy/ → Se copia a dist/deploy_bundle/deploy/
      ├── update.sh
      ├── install.sh
      └── ...

↓ make deploy-bundle

dist/deploy_bundle/         → Bundle listo para USB
  ├── run_update.sh         (generado)
  ├── run_install.sh        (generado)
  ├── project.zip           (generado desde project/)
  ├── README_DEPLOY.md      (generado)
  └── deploy/               (copiado desde deploy_bundle/deploy/)

↓ make deploy-bundle-usb USB=/Volumes/OS-FLEX

/Volumes/OS-FLEX/deploy_bundle/  → USB
  ├── run_update.sh
  ├── run_install.sh
  ├── project.zip
  ├── README_DEPLOY.md
  └── deploy/

↓ bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh

/opt/cantina-face-deploy/   → Disco local en Ubuntu
  ├── run_update.sh
  ├── run_install.sh
  ├── project.zip
  ├── deploy/
  ├── app.py               (descomprimido desde project.zip)
  ├── static/              (descomprimido desde project.zip)
  ├── venv/                (creado en disco local)
  └── data/                (preservado entre actualizaciones)
```

---

## 🎓 Mejores Prácticas

1. **Siempre usar `make deploy-bundle-usb`:** Es el comando todo-en-uno que garantiza calidad
2. **Validar antes de llevar el USB a la caja:** Evita viajes innecesarios
3. **No copiar manualmente:** Usar el Makefile para evitar errores
4. **Verificar la estructura en el USB:** Antes de desconectar
5. **Mantener backups:** Los runners crean backups automáticamente en `/opt/cantina-face-deploy/deploy/backups/`

---

## 📚 Referencias

- **Build script:** `tools/build_deploy_bundle.sh`
- **Validation script:** `tools/validate_deploy_bundle.sh`
- **Makefile:** `Makefile`
- **Runners:** `deploy_bundle/run_update.sh`, `deploy_bundle/run_install.sh`
- **Deploy scripts:** `deploy_bundle/deploy/update.sh`, `deploy_bundle/deploy/install.sh`
- **Documentación para la caja:** `README_DEPLOY.md` (generado en el bundle)

---

**Última actualización:** 2026-02-10  
**Versión:** 1.0.0

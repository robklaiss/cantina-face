---
description: Construir y desplegar bundle USB para Ubuntu 24.04 (noexec-safe)
---

# Deploy USB Bundle - Workflow Completo

Este workflow cubre la construcción y despliegue del bundle USB que funciona incluso con `noexec` y sistemas de archivos exFAT/FAT.

## 1. Construir el Bundle

```bash
make deploy-bundle
```

O manualmente:

```bash
bash tools/build_deploy_bundle.sh
```

Esto genera `dist/deploy_bundle/` con:
- `run_update.sh` - Runner para actualización
- `run_install.sh` - Runner para instalación inicial
- `project.zip` - Código fuente empaquetado
- `models/arcface_r50.onnx` - Modelo de reconocimiento facial (~166MB)
- `deploy/` - Scripts de deploy
- `README_DEPLOY.md` - Documentación

## 2. Copiar al USB

Copiar `dist/deploy_bundle/` al USB en la ruta:

```
OS-FLEX/deploy_bundle/
```

**Estructura final en el USB:**
```
OS-FLEX/
└── deploy_bundle/
    ├── run_update.sh
    ├── run_install.sh
    ├── project.zip
    ├── README_DEPLOY.md
    ├── models/
    │   └── arcface_r50.onnx
    └── deploy/
```

## 3. En la Máquina Ubuntu

### Instalación Inicial (primera vez)

```bash
bash /media/$USER/OS-FLEX/deploy_bundle/run_install.sh
```

### Actualización (versiones posteriores)

```bash
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
```

## Cómo Funciona (Internamente)

1. **Runner detecta su ubicación** en el USB
2. **Copia TODO** a `/opt/cantina-face-deploy` (disco local)
3. **Ejecuta scripts** desde disco local (evita noexec)
4. **Instala modelos** en `/opt/cantina-face/models/` (TARGET_APP_DIR)
5. **Crea venv** en disco local (evita problemas de symlinks)
6. **Permite desconectar** el USB después

## Validación del Bundle

El builder valida automáticamente:
- ✅ `run_update.sh` existe
- ✅ `run_install.sh` existe
- ✅ `project.zip` existe
- ✅ `models/arcface_r50.onnx` existe y > 1MB
- ✅ `deploy/update.sh` existe
- ✅ `deploy/install.sh` existe
- ✅ `deploy/guardrails/check_python_hardcode.sh` existe
- ❌ NO existe `deploy/usb/` (legacy prohibido)
- ❌ NO existe `project/` (legacy prohibido)
- ❌ NO existe `venv/`, `app.py` (contaminación)

Si falta algo o hay contaminación, el build falla con `exit 1`.

## Limpieza

```bash
make clean-bundle
```

## Troubleshooting

### "run_update.sh no existe"
- Verifica que el bundle esté en `OS-FLEX/deploy_bundle/`
- NO debe estar anidado: `OS-FLEX/deploy_bundle/deploy_bundle/`

### "Permission denied"
```bash
chmod +x /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
```

### USB montado con noexec
No hay problema. Los runners copian todo a disco local antes de ejecutar.

### Archivos ._* o .DS_Store
El builder los elimina automáticamente del bundle.

# Deploy Package

Scripts portables para instalar, ejecutar y actualizar `cantina-face` en cualquier máquina sin rutas absolutas.

## Contenido

1. **install.sh** – crea/usa el entorno virtual en `venv/` dentro del repo y resuelve dependencias desde `requirements.txt`.
2. **run.sh** – activa el venv y lanza FastAPI con `uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}`.
3. **update.sh** – aplica un `project.zip` al repo actual, preserva `data/`, reinstala dependencias y rota backups en `deploy/backups/`.
4. **project.zip** – snapshot del código (sin `venv/`).

## Instalación inicial

```bash
unzip project.zip
cd cantina-face
chmod +x deploy/*.sh
./deploy/install.sh
```

## Ejecución

```bash
./deploy/run.sh              # PORT=8000 por defecto
PORT=9000 ./deploy/run.sh    # puerto custom
UVICORN_RELOAD=1 ./deploy/run.sh  # habilitar --reload
```

## Actualizaciones

```bash
# usa deploy/project.zip por defecto
./deploy/update.sh

# o provee otro zip
./deploy/update.sh /ruta/a/project.zip
```

Durante la actualización se generan backups comprimidos (excluyendo `venv/`) en `deploy/backups/` con retención configurable vía `BACKUP_RETENTION` (default 5). Se preserva la carpeta `data/` moviéndola antes del `rsync` y restaurándola al final, por lo que los datos locales no se pierden.

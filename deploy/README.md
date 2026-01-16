# Deploy Package

Esta carpeta contiene lo necesario para mover todo el proyecto a otra máquina:

1. **install.sh** – crea el entorno virtual, instala dependencias y deja todo listo.
2. **run.sh** – inicia el servidor usando el entorno virtual.
3. **project.zip** – snapshot del código fuente (sin `venv/`).

## Pasos rápidos

```bash
# En la máquina destino
unzip project.zip
cd cantina-face
chmod +x deploy/install.sh deploy/run.sh
./deploy/install.sh
./deploy/run.sh
```


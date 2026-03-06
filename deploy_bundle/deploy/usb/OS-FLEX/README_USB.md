# Cantina Face — USB OFFLINE (OS-FLEX)

1. Montá el USB. Debe quedar en `/media/$USER/OS-FLEX`.
2. Confirmá que existen estas rutas:
   - `project/`
   - `models/arcface_r50.onnx`
   - `deploy/project.zip`
   - `deploy/ubuntu/install.sh`
3. Ejecutá el instalador en la caja objetivo:

```bash
sudo bash /media/$USER/OS-FLEX/deploy/ubuntu/install.sh
```

4. Diagnóstico del servicio luego de instalar:

```bash
sudo journalctl -u cantina-face -n 200 --no-pager
```

> Re-sellá el USB corriendo `bash deploy/usb/make_usb_offline.sh` dentro del repo.

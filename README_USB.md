# README_USB
1. Conectá el USB y verificá que quede montado como /media/$USER/OS-FLEX.
2. Confirmá que existan deploy/, project/, models/ y este README_USB.md.
3. Ejecutá: sudo bash /media/$USER/OS-FLEX/deploy/ubuntu/install.sh
4. El instalador valida estructura y copia la app sellada a /opt/cantina-face.
5. Crea el entorno virtual, instala dependencias y vincula el modelo local.
6. Corre preflight para chequear cámara/modelo sin tocar GDM.
7. Instala el servicio systemd cantina-face y sólo lo habilita si todo está OK.
8. Revisá el estado con systemctl status cantina-face y los logs en journalctl.
9. Ante cualquier error, seguí el mensaje, corregí y reejecutá el comando del paso 3.

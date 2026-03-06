# Deploy por SSH - Guía Completa

## 🎯 ¿Qué es esto?

Ahora puedes actualizar la máquina caja **remotamente por SSH** sin necesidad de USB. Ideal para:
- Actualizaciones rápidas sin ir físicamente a la caja
- Deploy desde tu máquina de desarrollo
- Automatización de actualizaciones

## 🚀 Quick Start

### Opción 1: Makefile (Recomendado)

```bash
# Construir bundle + subir por SSH (sin actualizar)
make deploy-ssh HOST=cantina@192.168.1.100

# Construir bundle + subir + actualizar automáticamente
make deploy-ssh-auto HOST=cantina@192.168.1.100
```

### Opción 2: Script directo

```bash
# Solo subir (sin actualizar)
./tools/deploy_ssh.sh cantina@192.168.1.100

# Subir y actualizar automáticamente
AUTO_UPDATE=1 ./tools/deploy_ssh.sh cantina@192.168.1.100
```

## 📋 Requisitos Previos

### 1. Acceso SSH a la máquina caja

```bash
# Probar conexión
ssh cantina@192.168.1.100

# Si funciona, estás listo ✅
```

### 2. Conocer la IP de la caja

**Opción A: Desde la caja**
```bash
ip addr show | grep "inet "
# O más simple:
hostname -I
```

**Opción B: Desde tu red**
```bash
# Escanear red local (requiere nmap)
nmap -sn 192.168.1.0/24 | grep -B 2 "cantina\|caja"
```

### 3. Usuario con permisos sudo

El usuario SSH debe poder ejecutar `sudo` para:
- Instalar dependencias
- Configurar servicios systemd
- Copiar archivos a `/opt/cantina-face`

## 🔧 Configuración SSH (Primera vez)

### Si no tienes acceso SSH aún:

#### 1. Habilitar SSH en la caja Ubuntu

```bash
# En la máquina caja:
sudo apt update
sudo apt install openssh-server -y
sudo systemctl enable ssh
sudo systemctl start ssh
```

#### 2. Crear usuario (si no existe)

```bash
# En la máquina caja:
sudo adduser cantina
sudo usermod -aG sudo cantina
```

#### 3. Configurar clave SSH (opcional pero recomendado)

```bash
# En tu máquina de desarrollo:
ssh-keygen -t ed25519 -C "deploy-cantina"

# Copiar clave a la caja:
ssh-copy-id cantina@192.168.1.100

# Ahora puedes conectar sin contraseña:
ssh cantina@192.168.1.100
```

## 📦 Uso Detallado

### Comando 1: Deploy sin actualizar

```bash
make deploy-ssh HOST=cantina@192.168.1.100
```

**¿Qué hace?**
1. Construye el deploy bundle
2. Valida la estructura
3. Sube el bundle por SSH a `/tmp/cantina-deploy`
4. **NO ejecuta la actualización** (la deja lista)

**Cuándo usar:**
- Quieres revisar los archivos antes de actualizar
- Vas a ejecutar la actualización manualmente
- Estás probando el proceso

**Después de esto:**
```bash
# Conectar por SSH
ssh cantina@192.168.1.100

# Ejecutar actualización manualmente
bash /tmp/cantina-deploy/run_update.sh
```

### Comando 2: Deploy con actualización automática

```bash
make deploy-ssh-auto HOST=cantina@192.168.1.100
```

**¿Qué hace?**
1. Construye el deploy bundle
2. Valida la estructura
3. Sube el bundle por SSH
4. **Ejecuta automáticamente** `run_update.sh`
5. Muestra el output en tiempo real

**Cuándo usar:**
- Actualizaciones de confianza
- Quieres automatizar todo el proceso
- No necesitas revisar antes de actualizar

### Variables de Entorno

```bash
# Puerto SSH personalizado
SSH_PORT=2222 make deploy-ssh HOST=user@host

# Clave SSH específica
SSH_KEY=~/.ssh/id_caja make deploy-ssh HOST=user@host

# Directorio remoto personalizado
REMOTE_DIR=/home/cantina/deploy make deploy-ssh HOST=user@host

# Combinar varias opciones
SSH_PORT=2222 SSH_KEY=~/.ssh/id_caja AUTO_UPDATE=1 \
  ./tools/deploy_ssh.sh cantina@192.168.1.100
```

## 🔍 Proceso Paso a Paso

### Lo que hace el script:

```
[1/5] Verificando bundle...
  ✅ Verifica que existan todos los archivos críticos
  ✅ Valida project.zip, scripts, modelos

[2/5] Probando conexión SSH...
  ✅ Intenta conectar al host
  ✅ Verifica credenciales

[3/5] Creando directorio remoto...
  ✅ Crea /tmp/cantina-deploy (o el que especifiques)

[4/5] Subiendo bundle...
  ✅ Usa rsync para transferencia eficiente
  ✅ Muestra progreso en tiempo real
  ✅ Solo sube archivos modificados

[5/5] Verificando archivos remotos...
  ✅ Confirma que los archivos llegaron correctamente
```

### Si AUTO_UPDATE=1:

```
[6/6] Ejecutando actualización...
  ✅ Ejecuta run_update.sh remotamente
  ✅ Muestra output en tiempo real
  ✅ Espera a que termine
```

## 🌐 Casos de Uso

### Caso 1: Desarrollo local → Caja en la misma red

```bash
# Tu Mac/Linux → Caja Ubuntu en 192.168.1.100
make deploy-ssh-auto HOST=cantina@192.168.1.100
```

### Caso 2: Desarrollo remoto → Caja con IP pública

```bash
# Tu Mac → Caja con IP pública 203.0.113.50
make deploy-ssh-auto HOST=cantina@203.0.113.50
```

### Caso 3: Desarrollo → Caja detrás de Cloudflare Tunnel

```bash
# Si configuraste SSH sobre el túnel de Cloudflare
make deploy-ssh-auto HOST=cantina@caja.tudominio.com
```

### Caso 4: Puerto SSH no estándar

```bash
# Caja con SSH en puerto 2222
SSH_PORT=2222 make deploy-ssh-auto HOST=cantina@192.168.1.100
```

### Caso 5: Múltiples cajas

```bash
# Actualizar varias cajas en secuencia
for host in caja1 caja2 caja3; do
  make deploy-ssh-auto HOST=cantina@$host.local
done
```

## ✅ Verificación Post-Deploy

### Verificar que la actualización funcionó:

```bash
# Ver logs del servicio
ssh cantina@192.168.1.100 'sudo journalctl -u cantina-face -n 50'

# Verificar versión
ssh cantina@192.168.1.100 'cat /opt/cantina-face-deploy/.current_version'

# Verificar que el servicio esté corriendo
ssh cantina@192.168.1.100 'sudo systemctl status cantina-face'

# Probar endpoint
ssh cantina@192.168.1.100 'curl -s http://localhost:8000/health'
```

## 🐛 Troubleshooting

### Error: "Permission denied (publickey)"

**Problema:** No tienes clave SSH configurada

**Solución:**
```bash
ssh-copy-id cantina@192.168.1.100
# O usa contraseña (menos seguro)
```

### Error: "Connection refused"

**Problema:** SSH no está corriendo en la caja

**Solución:**
```bash
# En la caja:
sudo systemctl start ssh
sudo systemctl enable ssh
```

### Error: "Host key verification failed"

**Problema:** La clave del host cambió (reinstalación, etc.)

**Solución:**
```bash
ssh-keygen -R 192.168.1.100
# Luego intenta de nuevo
```

### Error: "sudo: a password is required"

**Problema:** El usuario no puede hacer sudo sin contraseña

**Solución:**
```bash
# En la caja, agregar usuario a sudoers NOPASSWD (opcional):
echo "cantina ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/cantina
```

### Error: "No se puede conectar a 192.168.1.100"

**Problema:** IP incorrecta o caja apagada

**Solución:**
```bash
# Verificar IP de la caja
ping 192.168.1.100

# Escanear red
nmap -sn 192.168.1.0/24
```

### La actualización se cuelga

**Problema:** Falta alguna dependencia o error en el script

**Solución:**
```bash
# Conectar por SSH y ver logs en tiempo real
ssh cantina@192.168.1.100
tail -f /opt/cantina-face-deploy/deploy/backups/*.log
```

## 🔐 Seguridad

### Buenas prácticas:

1. **Usa claves SSH, no contraseñas**
   ```bash
   ssh-keygen -t ed25519
   ssh-copy-id cantina@host
   ```

2. **Limita acceso SSH por IP** (opcional)
   ```bash
   # En la caja: /etc/ssh/sshd_config
   AllowUsers cantina@192.168.1.*
   ```

3. **Usa puerto SSH no estándar** (opcional)
   ```bash
   # En la caja: /etc/ssh/sshd_config
   Port 2222
   ```

4. **Deshabilita login root**
   ```bash
   # En la caja: /etc/ssh/sshd_config
   PermitRootLogin no
   ```

5. **Usa fail2ban** (opcional)
   ```bash
   sudo apt install fail2ban
   ```

## 📊 Comparación: USB vs SSH

| Característica | USB | SSH |
|----------------|-----|-----|
| Velocidad | 🟡 Media (depende del USB) | 🟢 Rápida (depende de red) |
| Conveniencia | 🔴 Requiere acceso físico | 🟢 Remoto |
| Seguridad | 🟢 Física | 🟡 Red (usa SSH keys) |
| Automatización | 🔴 Manual | 🟢 Scripteable |
| Múltiples cajas | 🔴 Una por una | 🟢 Paralelo/secuencial |
| Offline | 🟢 Funciona sin red | 🔴 Requiere red |

## 💡 Tips y Trucos

### Alias útiles

```bash
# Agregar a ~/.bashrc o ~/.zshrc
alias deploy-caja='make deploy-ssh-auto HOST=cantina@192.168.1.100'
alias ssh-caja='ssh cantina@192.168.1.100'
alias logs-caja='ssh cantina@192.168.1.100 "sudo journalctl -u cantina-face -f"'
```

### Script de deploy a múltiples cajas

```bash
#!/bin/bash
# deploy-all.sh

CAJAS=(
  "cantina@192.168.1.100"
  "cantina@192.168.1.101"
  "cantina@192.168.1.102"
)

for caja in "${CAJAS[@]}"; do
  echo "Desplegando a $caja..."
  make deploy-ssh-auto HOST=$caja
  echo "✅ $caja completado"
  echo ""
done
```

### Monitoreo post-deploy

```bash
# Ver logs de todas las cajas en paralelo
tmux new-session \; \
  send-keys 'ssh caja1 "sudo journalctl -u cantina-face -f"' C-m \; \
  split-window -h \; \
  send-keys 'ssh caja2 "sudo journalctl -u cantina-face -f"' C-m \; \
  split-window -h \; \
  send-keys 'ssh caja3 "sudo journalctl -u cantina-face -f"' C-m
```

## 🆘 Soporte

### Archivos de log importantes:

```bash
# En la caja:
/opt/cantina-face-deploy/deploy/backups/  # Backups
/var/log/syslog                           # Sistema
sudo journalctl -u cantina-face           # Servicio
```

### Comandos de diagnóstico:

```bash
# Estado del servicio
ssh cantina@host 'sudo systemctl status cantina-face'

# Últimos 100 logs
ssh cantina@host 'sudo journalctl -u cantina-face -n 100'

# Verificar puerto 8000
ssh cantina@host 'sudo netstat -tlnp | grep 8000'

# Procesos Python
ssh cantina@host 'ps aux | grep python'
```

## 📚 Recursos Adicionales

- **Deploy Bundle**: `README_DEPLOY.md`
- **Cloudflare Tunnel**: `README_CLOUDFLARE_TUNNEL.md`
- **Remote Update**: `REMOTE_UPDATE_SETUP.md`
- **Makefile**: Ver `make help`

---

**¡Listo para deploy remoto! 🚀**

# Configuración de Actualización Remota de Máquina Caja

## 🎯 Objetivo

Permitir que el backend en HostGator pueda comunicarse con la máquina caja (local) para disparar actualizaciones remotas.

## 🔧 Requisitos

1. **Backend en HostGator**: Ya configurado con los endpoints de actualización
2. **Máquina Caja**: Corriendo FastAPI en puerto 8000
3. **Conexión a Internet**: Ambos servidores deben poder comunicarse

---

## 📡 Opciones para Exponer la Máquina Caja

### Opción 1: Ngrok (Rápido - Testing/Desarrollo)

**Instalación:**
```bash
# Descargar ngrok desde https://ngrok.com/download
# O con brew en Mac:
brew install ngrok

# Autenticar (crear cuenta gratuita en ngrok.com)
ngrok config add-authtoken TU_TOKEN_AQUI
```

**Uso:**
```bash
# En la máquina caja, ejecutar:
ngrok http 8000
```

Esto mostrará algo como:
```
Forwarding  https://abc123-xyz.ngrok.io -> http://localhost:8000
```

**Configurar en HostGator:**
```php
// En app/config.php
'caja' => [
    'url' => 'https://abc123-xyz.ngrok.io',  // URL de ngrok
    'internal_token' => 'tu-token-secreto-aqui',
],
```

**⚠️ Nota:** La URL de ngrok cambia cada vez que reinicias (versión gratuita). Para URL fija, necesitas plan de pago.

---

### Opción 2: Cloudflare Tunnel (Recomendada - Producción)

**Ventajas:**
- ✅ Gratuito
- ✅ URL permanente
- ✅ HTTPS automático
- ✅ No requiere abrir puertos en router

**Instalación:**
```bash
# En la máquina caja (Ubuntu/Linux):
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Autenticar con Cloudflare
cloudflared tunnel login
```

**Crear túnel permanente:**
```bash
# Crear túnel
cloudflared tunnel create cantina-caja

# Esto creará un archivo de credenciales en ~/.cloudflared/

# Configurar túnel
cat > ~/.cloudflared/config.yml << EOF
tunnel: cantina-caja
credentials-file: /home/usuario/.cloudflared/TUNNEL_ID.json

ingress:
  - hostname: caja.tudominio.com
    service: http://localhost:8000
  - service: http_status:404
EOF

# Configurar DNS en Cloudflare
cloudflared tunnel route dns cantina-caja caja.tudominio.com

# Ejecutar túnel
cloudflared tunnel run cantina-caja
```

**Ejecutar como servicio (systemd):**
```bash
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

**Configurar en HostGator:**
```php
'caja' => [
    'url' => 'https://caja.tudominio.com',
    'internal_token' => 'tu-token-secreto-aqui',
],
```

---

### Opción 3: IP Pública + Port Forwarding

**Si tienes IP pública estática:**

1. **Configurar port forwarding en el router:**
   - Puerto externo: 8000
   - IP interna: IP de la máquina caja
   - Puerto interno: 8000

2. **Obtener IP pública:**
   ```bash
   curl ifconfig.me
   ```

3. **Configurar en HostGator:**
   ```php
   'caja' => [
       'url' => 'http://TU_IP_PUBLICA:8000',
       'internal_token' => 'tu-token-secreto-aqui',
   ],
   ```

**⚠️ Consideraciones de seguridad:**
- Usar HTTPS (requiere certificado SSL)
- Configurar firewall para permitir solo IPs específicas
- Cambiar puerto por defecto (ej: 8443)

---

## 🔐 Configuración de Seguridad

### 1. En la Máquina Caja

Crear archivo `.env-claves` o agregar a `.env`:
```bash
INTERNAL_TOKEN=genera-un-token-aleatorio-largo-y-seguro
```

**Generar token seguro:**
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. En el Backend (HostGator)

Editar `app/config.php`:
```php
'caja' => [
    'url' => 'https://tu-url-publica-aqui',
    'internal_token' => 'el-mismo-token-que-en-la-caja',
],
```

**⚠️ IMPORTANTE:** El token debe ser **exactamente el mismo** en ambos lados.

---

## 🧪 Probar la Conexión

### Desde línea de comandos:

```bash
# Probar endpoint de estado
curl -H "X-Internal-Token: tu-token-aqui" \
     https://tu-url-caja/api/admin/update-status

# Probar endpoint de actualización
curl -X POST \
     -H "X-Internal-Token: tu-token-aqui" \
     -H "Content-Type: application/json" \
     https://tu-url-caja/api/admin/trigger-update
```

### Desde el Backend:

1. Iniciar sesión en `/backend`
2. Ir a la sección "Actualización de Máquina Caja"
3. Hacer clic en "Verificar estado"
4. Si funciona, verás la versión actual de la caja

---

## 🔍 Troubleshooting

### Error: "Could not connect to server"
- ✅ Verificar que la máquina caja esté corriendo (`ps aux | grep python`)
- ✅ Verificar que el túnel/ngrok esté activo
- ✅ Probar la URL desde el navegador

### Error: "Invalid internal token"
- ✅ Verificar que el token sea idéntico en ambos lados
- ✅ No debe tener espacios ni caracteres especiales al inicio/final
- ✅ Reiniciar la aplicación de la caja después de cambiar el token

### Error: "Update script not found"
- ✅ Verificar que exista `/opt/cantina-face/deploy/check_update.sh`
- ✅ Verificar permisos de ejecución: `chmod +x check_update.sh`

### La actualización no se ejecuta
- ✅ Ver logs de la caja: `tail -f /opt/cantina-face/logs/app.log`
- ✅ Verificar que el script tenga permisos de sudo si es necesario
- ✅ Ejecutar manualmente el script para ver errores

---

## 📊 Monitoreo

### Ver logs de la caja:
```bash
# Logs de la aplicación
tail -f /opt/cantina-face/logs/app.log

# Logs del túnel (si usas cloudflared)
sudo journalctl -u cloudflared -f

# Logs de ngrok
# Ver en la interfaz web: http://localhost:4040
```

---

## 🚀 Recomendación Final

**Para producción, usa Cloudflare Tunnel:**
1. Es gratuito y confiable
2. URL permanente
3. HTTPS automático
4. No requiere configurar router
5. Incluye protección DDoS

**Para testing rápido, usa Ngrok:**
1. Setup en 2 minutos
2. Perfecto para pruebas
3. Fácil de detener/iniciar

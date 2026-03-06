# Cloudflare Tunnel - Acceso Remoto a Máquina Caja

## 🎯 ¿Qué es esto?

Cloudflare Tunnel permite que el backend en HostGator se comunique con la máquina caja (local) a través de internet de forma segura, sin necesidad de:
- Abrir puertos en el router
- Configurar port forwarding
- Tener IP pública estática
- Configurar certificados SSL manualmente

## ✨ Instalación Automática

El sistema de actualización remota **instala automáticamente cloudflared** cuando actualizas la máquina caja. No necesitas hacer nada especial.

Después de actualizar, verás un mensaje como:
```
[update] Verificando cloudflared para acceso remoto...
[cloudflare] ✅ cloudflared ya está instalado (versión X.X.X)
```

## 🔧 Configuración del Túnel (Primera vez)

Después de instalar/actualizar, ejecuta el script de configuración:

```bash
cd /opt/cantina-face
./deploy/setup_cloudflare_tunnel.sh
```

El script te guiará paso a paso:

### Paso 1: Autenticación
```bash
cloudflared tunnel login
```
- Se abrirá tu navegador
- Inicia sesión en Cloudflare (crea cuenta gratuita si no tienes)
- Autoriza el acceso

### Paso 2: Crear túnel
El script crea automáticamente un túnel llamado `cantina-caja`

### Paso 3: Configurar dominio (opcional pero recomendado)

**Opción A: Con dominio propio**
- Si tienes un dominio en Cloudflare (ej: `tudominio.com`)
- El script configurará: `caja.tudominio.com`
- DNS se configura automáticamente

**Opción B: Sin dominio**
- Usa túnel temporal
- URL será algo como: `https://random-name.trycloudflare.com`
- ⚠️ La URL cambia si reinicias el túnel

### Paso 4: Servicio systemd
El script crea un servicio que inicia automáticamente con el sistema.

## 📋 Comandos Útiles

```bash
# Ver estado del túnel
sudo systemctl status cloudflared-cantina

# Ver logs en tiempo real
sudo journalctl -u cloudflared-cantina -f

# Reiniciar túnel
sudo systemctl restart cloudflared-cantina

# Detener túnel
sudo systemctl stop cloudflared-cantina

# Iniciar túnel
sudo systemctl start cloudflared-cantina

# Ver información del túnel
cloudflared tunnel info cantina-caja

# Listar todos los túneles
cloudflared tunnel list
```

## 🌐 Configurar Backend (HostGator)

Una vez que tengas la URL del túnel (ej: `https://caja.tudominio.com`), configúrala en el backend:

### 1. Editar `app/config.php` en HostGator:

```php
'caja' => [
    'url' => 'https://caja.tudominio.com',  // Tu URL del túnel
    'internal_token' => 'genera-token-seguro-aqui',
],
```

### 2. Generar token seguro:

```bash
# En cualquier máquina con Python
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Configurar el mismo token en la máquina caja:

Editar `/opt/cantina-face/.env-claves` o crear `.env`:

```bash
INTERNAL_TOKEN=el-mismo-token-que-en-hostgator
```

### 4. Reiniciar aplicación de la caja:

```bash
sudo systemctl restart cantina-face
```

## ✅ Probar la Conexión

### Desde línea de comandos:

```bash
# Reemplaza TU_URL y TU_TOKEN con tus valores
curl -H "X-Internal-Token: TU_TOKEN" https://TU_URL/api/admin/update-status
```

Deberías ver algo como:
```json
{
  "current_version": "1.0.0",
  "last_check": "2026-02-26 10:00:00",
  "update_available": false,
  "remote_version": null
}
```

### Desde el Backend:

1. Ir a `https://tudominio.com/backend`
2. Iniciar sesión como admin
3. Scroll hasta "Actualización de Máquina Caja"
4. Hacer clic en "Verificar estado"
5. Deberías ver la versión actual de la caja

## 🔍 Troubleshooting

### Error: "cloudflared: command not found"
```bash
# Reinstalar cloudflared
cd /opt/cantina-face
./deploy/install_cloudflare_auto.sh
```

### Error: "tunnel credentials file not found"
```bash
# Reautenticar
cloudflared tunnel login

# Recrear túnel
cloudflared tunnel create cantina-caja
```

### El túnel no inicia automáticamente
```bash
# Verificar servicio
sudo systemctl status cloudflared-cantina

# Habilitar servicio
sudo systemctl enable cloudflared-cantina
sudo systemctl start cloudflared-cantina
```

### Error: "Could not connect to server"
```bash
# Verificar que el túnel esté corriendo
sudo systemctl status cloudflared-cantina

# Verificar que la app esté corriendo
sudo systemctl status cantina-face

# Ver logs del túnel
sudo journalctl -u cloudflared-cantina -n 50
```

### Error: "Invalid internal token"
```bash
# Verificar token en la caja
cat /opt/cantina-face/.env-claves | grep INTERNAL_TOKEN

# Verificar token en HostGator (app/config.php)
# Deben ser EXACTAMENTE iguales
```

## 🔐 Seguridad

### Buenas prácticas:

1. **Usa un token fuerte**: Mínimo 32 caracteres aleatorios
2. **Nunca compartas el token**: Es como una contraseña
3. **Usa HTTPS**: Cloudflare lo proporciona automáticamente
4. **Monitorea los logs**: Revisa accesos sospechosos
5. **Rota el token periódicamente**: Cámbialo cada 3-6 meses

### Cambiar el token:

```bash
# 1. Generar nuevo token
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# 2. Actualizar en la caja (.env-claves)
INTERNAL_TOKEN=nuevo-token-aqui

# 3. Actualizar en HostGator (app/config.php)
'internal_token' => 'nuevo-token-aqui',

# 4. Reiniciar aplicación de la caja
sudo systemctl restart cantina-face
```

## 📊 Monitoreo

### Ver estadísticas del túnel:

```bash
# Logs en tiempo real
sudo journalctl -u cloudflared-cantina -f

# Últimas 100 líneas
sudo journalctl -u cloudflared-cantina -n 100

# Logs de hoy
sudo journalctl -u cloudflared-cantina --since today
```

### Dashboard de Cloudflare:

1. Ir a https://dash.cloudflare.com
2. Seleccionar tu dominio
3. Ir a "Traffic" → "Cloudflare Tunnel"
4. Ver métricas de uso, requests, etc.

## 🚀 Ventajas de Cloudflare Tunnel

✅ **Gratuito**: Sin costos, sin límites de tráfico  
✅ **Seguro**: Encriptación end-to-end  
✅ **Fácil**: No requiere configurar router  
✅ **Confiable**: 99.9% uptime  
✅ **Rápido**: Red global de Cloudflare  
✅ **Automático**: Inicia con el sistema  
✅ **HTTPS**: Certificados SSL automáticos  

## 📚 Recursos Adicionales

- [Documentación oficial de Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
- [Guía de troubleshooting](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/troubleshooting/)
- [Dashboard de Cloudflare](https://dash.cloudflare.com)

## 💡 Consejos

1. **Usa un dominio propio**: Más profesional y URL permanente
2. **Monitorea los logs**: Detecta problemas temprano
3. **Haz backup de la configuración**: Guarda `~/.cloudflared/` en lugar seguro
4. **Documenta tu URL**: Anótala en un lugar seguro
5. **Prueba regularmente**: Verifica que la conexión funcione

## 🆘 Soporte

Si tienes problemas:
1. Revisa los logs: `sudo journalctl -u cloudflared-cantina -n 100`
2. Verifica la configuración: `cat ~/.cloudflared/config.yml`
3. Prueba la conexión local: `curl http://localhost:8000/health`
4. Prueba la conexión remota: `curl https://tu-url/health`

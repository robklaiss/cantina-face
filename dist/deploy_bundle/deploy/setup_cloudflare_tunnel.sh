#!/usr/bin/env bash
set -euo pipefail

# Script para instalar y configurar Cloudflare Tunnel en la máquina caja
# Permite acceso remoto seguro sin abrir puertos en el router

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_APP_DIR="${TARGET_APP_DIR:-/opt/cantina-face}"
CLOUDFLARED_VERSION="latest"
TUNNEL_NAME="${TUNNEL_NAME:-cantina-caja}"
CONFIG_DIR="$HOME/.cloudflared"
TUNNEL_CONFIG="$CONFIG_DIR/config.yml"

echo "🌐 Configuración de Cloudflare Tunnel para Cantina Face"
echo "======================================================="
echo ""

# Verificar si ya está instalado
if command -v cloudflared &> /dev/null; then
    echo "✅ cloudflared ya está instalado"
    cloudflared --version
else
    echo "📦 Instalando cloudflared..."
    
    # Detectar arquitectura
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            PACKAGE="cloudflared-linux-amd64.deb"
            ;;
        aarch64|arm64)
            PACKAGE="cloudflared-linux-arm64.deb"
            ;;
        armv7l)
            PACKAGE="cloudflared-linux-arm.deb"
            ;;
        *)
            echo "❌ Arquitectura no soportada: $ARCH"
            exit 1
            ;;
    esac
    
    # Descargar e instalar
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    echo "   Descargando $PACKAGE..."
    wget -q "https://github.com/cloudflare/cloudflared/releases/latest/download/$PACKAGE"
    
    echo "   Instalando paquete..."
    sudo dpkg -i "$PACKAGE" || {
        echo "   Instalando dependencias faltantes..."
        sudo apt-get update
        sudo apt-get install -f -y
    }
    
    cd - > /dev/null
    rm -rf "$TEMP_DIR"
    
    echo "✅ cloudflared instalado correctamente"
fi

echo ""
echo "📋 Configuración del túnel"
echo "=========================="
echo ""

# Verificar si ya existe configuración
if [ -f "$TUNNEL_CONFIG" ]; then
    echo "⚠️  Ya existe una configuración de túnel en $TUNNEL_CONFIG"
    echo ""
    read -p "¿Deseas reconfigurar? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Manteniendo configuración existente"
        exit 0
    fi
fi

# Crear directorio de configuración
mkdir -p "$CONFIG_DIR"

echo "Para configurar el túnel necesitas:"
echo "1. Una cuenta de Cloudflare (gratuita)"
echo "2. Un dominio configurado en Cloudflare (opcional pero recomendado)"
echo ""
echo "Pasos a seguir:"
echo ""
echo "PASO 1: Autenticar con Cloudflare"
echo "----------------------------------"
echo "Ejecuta el siguiente comando y sigue las instrucciones en el navegador:"
echo ""
echo "  cloudflared tunnel login"
echo ""
read -p "Presiona Enter cuando hayas completado la autenticación..."
echo ""

# Verificar autenticación
if [ ! -f "$CONFIG_DIR/cert.pem" ]; then
    echo "❌ No se encontró el certificado de autenticación"
    echo "Por favor ejecuta: cloudflared tunnel login"
    exit 1
fi

echo "✅ Autenticación completada"
echo ""

echo "PASO 2: Crear túnel"
echo "-------------------"

# Verificar si el túnel ya existe
if cloudflared tunnel list 2>/dev/null | grep -q "$TUNNEL_NAME"; then
    echo "⚠️  El túnel '$TUNNEL_NAME' ya existe"
    TUNNEL_ID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
else
    echo "Creando túnel '$TUNNEL_NAME'..."
    cloudflared tunnel create "$TUNNEL_NAME"
    TUNNEL_ID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
    echo "✅ Túnel creado con ID: $TUNNEL_ID"
fi

echo ""
echo "PASO 3: Configurar túnel"
echo "------------------------"

# Preguntar por el hostname
echo ""
echo "Opciones de configuración:"
echo "1. Usar un subdominio (recomendado): caja.tudominio.com"
echo "2. Usar túnel temporal sin dominio: https://RANDOM.trycloudflare.com"
echo ""
read -p "¿Tienes un dominio en Cloudflare? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    read -p "Ingresa el hostname completo (ej: caja.tudominio.com): " HOSTNAME
    
    # Crear configuración con hostname
    cat > "$TUNNEL_CONFIG" <<EOF
tunnel: $TUNNEL_ID
credentials-file: $CONFIG_DIR/$TUNNEL_ID.json

ingress:
  - hostname: $HOSTNAME
    service: http://localhost:8000
  - service: http_status:404
EOF

    echo ""
    echo "Configurando DNS en Cloudflare..."
    if cloudflared tunnel route dns "$TUNNEL_NAME" "$HOSTNAME"; then
        echo "✅ DNS configurado correctamente"
        TUNNEL_URL="https://$HOSTNAME"
    else
        echo "⚠️  Error al configurar DNS. Configúralo manualmente en Cloudflare:"
        echo "   Tipo: CNAME"
        echo "   Nombre: $(echo $HOSTNAME | cut -d. -f1)"
        echo "   Destino: $TUNNEL_ID.cfargotunnel.com"
        TUNNEL_URL="https://$HOSTNAME"
    fi
else
    # Configuración sin dominio (túnel temporal)
    cat > "$TUNNEL_CONFIG" <<EOF
tunnel: $TUNNEL_ID
credentials-file: $CONFIG_DIR/$TUNNEL_ID.json

ingress:
  - service: http://localhost:8000
EOF
    
    echo "⚠️  Usando túnel temporal. La URL será asignada al iniciar el túnel."
    TUNNEL_URL="(se mostrará al iniciar el túnel)"
fi

echo ""
echo "✅ Configuración guardada en $TUNNEL_CONFIG"
echo ""

echo "PASO 4: Configurar servicio systemd"
echo "------------------------------------"

# Crear servicio systemd
SERVICE_FILE="/etc/systemd/system/cloudflared-cantina.service"

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Cloudflare Tunnel for Cantina Face
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=/usr/bin/cloudflared tunnel --config $TUNNEL_CONFIG run $TUNNEL_NAME
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Servicio systemd creado"
echo ""

# Habilitar y arrancar servicio
echo "Habilitando servicio..."
sudo systemctl daemon-reload
sudo systemctl enable cloudflared-cantina.service

read -p "¿Deseas iniciar el túnel ahora? (Y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    sudo systemctl start cloudflared-cantina.service
    echo "✅ Túnel iniciado"
    
    # Esperar un momento para que inicie
    sleep 3
    
    # Mostrar estado
    echo ""
    echo "Estado del servicio:"
    sudo systemctl status cloudflared-cantina.service --no-pager -l | head -20
fi

echo ""
echo "=========================================="
echo "✅ Configuración completada"
echo "=========================================="
echo ""
echo "📝 Información del túnel:"
echo "   Nombre: $TUNNEL_NAME"
echo "   ID: $TUNNEL_ID"
if [ -n "${HOSTNAME:-}" ]; then
    echo "   URL: $TUNNEL_URL"
fi
echo ""
echo "🔧 Comandos útiles:"
echo "   Ver estado:    sudo systemctl status cloudflared-cantina"
echo "   Ver logs:      sudo journalctl -u cloudflared-cantina -f"
echo "   Reiniciar:     sudo systemctl restart cloudflared-cantina"
echo "   Detener:       sudo systemctl stop cloudflared-cantina"
echo "   Deshabilitar:  sudo systemctl disable cloudflared-cantina"
echo ""
echo "📋 Próximos pasos:"
echo "1. Copia la URL del túnel: $TUNNEL_URL"
echo "2. Configúrala en el backend (HostGator) en app/config.php:"
echo "   'caja' => ["
echo "       'url' => '$TUNNEL_URL',"
echo "       'internal_token' => 'tu-token-secreto',"
echo "   ]"
echo ""
echo "3. Configura el mismo token en la máquina caja (.env-claves):"
echo "   INTERNAL_TOKEN=tu-token-secreto"
echo ""
echo "4. Reinicia la aplicación de la caja para aplicar cambios"
echo ""

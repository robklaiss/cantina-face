#!/bin/bash
# Script para configurar túnel ngrok para actualización remota de máquina caja

set -e

echo "🚀 Configuración de Túnel Ngrok para Máquina Caja"
echo "=================================================="
echo ""

# Verificar si ngrok está instalado
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok no está instalado"
    echo ""
    echo "Opciones de instalación:"
    echo "  - Mac: brew install ngrok"
    echo "  - Linux: snap install ngrok"
    echo "  - Manual: https://ngrok.com/download"
    echo ""
    exit 1
fi

echo "✅ ngrok encontrado"
echo ""

# Verificar si la app está corriendo
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  La aplicación no parece estar corriendo en puerto 8000"
    echo ""
    echo "Inicia la aplicación primero:"
    echo "  cd /opt/cantina-face"
    echo "  source venv/bin/activate"
    echo "  python app.py"
    echo ""
    read -p "¿Continuar de todas formas? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "🔧 Iniciando túnel ngrok..."
echo ""
echo "IMPORTANTE: Deja esta terminal abierta mientras uses el túnel"
echo "La URL pública se mostrará abajo. Cópiala y úsala en la configuración del backend."
echo ""
echo "Presiona Ctrl+C para detener el túnel cuando termines."
echo ""
echo "=================================================="
echo ""

# Iniciar ngrok
ngrok http 8000 --log=stdout

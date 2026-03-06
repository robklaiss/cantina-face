.PHONY: deploy-bundle deploy-bundle-validate deploy-bundle-usb deploy-ssh deploy-ssh-auto test-deploy-bundle clean-bundle help

help:
	@echo "Cantina Face - Makefile"
	@echo ""
	@echo "Targets disponibles:"
	@echo "  deploy-bundle              - Construir bundle de deploy para USB"
	@echo "  deploy-bundle-validate     - Validar estructura del bundle"
	@echo "  test-deploy-bundle         - Test completo (build + validate + simulate USB)"
	@echo "  deploy-bundle-usb USB=path - Construir + validar + copiar a USB"
	@echo "  deploy-ssh HOST=user@host  - Construir + subir por SSH (sin actualizar)"
	@echo "  deploy-ssh-auto HOST=...   - Construir + subir por SSH + actualizar automáticamente"
	@echo "  clean-bundle               - Limpiar bundle generado"
	@echo "  help                       - Mostrar esta ayuda"
	@echo ""
	@echo "Ejemplos:"
	@echo "  make test-deploy-bundle"
	@echo "  make deploy-bundle-usb USB=/Volumes/OS-FLEX"
	@echo "  make deploy-bundle-usb USB=/media/\$$USER/OS-FLEX"
	@echo "  make deploy-ssh HOST=cantina@192.168.1.100"
	@echo "  make deploy-ssh-auto HOST=ubuntu@caja.local"
	@echo "  SSH_PORT=2222 make deploy-ssh HOST=user@host"

deploy-bundle:
	@bash tools/build_deploy_bundle.sh

deploy-bundle-validate:
	@bash tools/validate_deploy_bundle.sh

test-deploy-bundle:
	@bash tools/test_deploy_bundle.sh

deploy-bundle-usb:
ifndef USB
	@echo "❌ ERROR: USB parameter is required" >&2
	@echo "Usage: make deploy-bundle-usb USB=/Volumes/OS-FLEX" >&2
	@echo "   or: make deploy-bundle-usb USB=/media/\$$USER/OS-FLEX" >&2
	@exit 1
endif
	@echo "============================================"
	@echo "Building and deploying to USB"
	@echo "============================================"
	@echo "Target: $(USB)/deploy_bundle/"
	@echo ""
	@bash tools/build_deploy_bundle.sh
	@bash tools/validate_deploy_bundle.sh
	@echo ""
	@echo "============================================"
	@echo "Copying to USB..."
	@echo "============================================"
	@if [ ! -d "$(USB)" ]; then \
		echo "❌ ERROR: USB path does not exist: $(USB)" >&2; \
		echo "Available volumes:" >&2; \
		ls -la /Volumes/ 2>/dev/null || ls -la /media/$$USER/ 2>/dev/null || echo "No volumes found" >&2; \
		exit 1; \
	fi
	@echo "Removing old bundle from USB..."
	@rm -rf "$(USB)/deploy_bundle"
	@echo "Copying new bundle to USB..."
	@ditto "dist/deploy_bundle" "$(USB)/deploy_bundle"
	@echo "Syncing filesystem..."
	@sync
	@echo ""
	@echo "✅ Bundle deployed successfully to: $(USB)/deploy_bundle/"
	@echo ""
	@echo "En Ubuntu, ejecutar:"
	@echo "  bash /media/\$$USER/OS-FLEX/deploy_bundle/run_update.sh"
	@echo "  bash /media/\$$USER/OS-FLEX/deploy_bundle/run_install.sh"

deploy-ssh:
ifndef HOST
	@echo "❌ ERROR: HOST parameter is required" >&2
	@echo "Usage: make deploy-ssh HOST=user@host" >&2
	@echo "Examples:" >&2
	@echo "  make deploy-ssh HOST=cantina@192.168.1.100" >&2
	@echo "  make deploy-ssh HOST=ubuntu@caja.local" >&2
	@echo "  SSH_PORT=2222 make deploy-ssh HOST=user@host" >&2
	@exit 1
endif
	@echo "============================================"
	@echo "Building and deploying via SSH"
	@echo "============================================"
	@echo "Target: $(HOST)"
	@echo ""
	@bash tools/build_deploy_bundle.sh
	@bash tools/validate_deploy_bundle.sh
	@echo ""
	@bash tools/deploy_ssh.sh $(HOST)

deploy-ssh-auto:
ifndef HOST
	@echo "❌ ERROR: HOST parameter is required" >&2
	@echo "Usage: make deploy-ssh-auto HOST=user@host" >&2
	@exit 1
endif
	@echo "============================================"
	@echo "Building and auto-deploying via SSH"
	@echo "============================================"
	@echo "Target: $(HOST)"
	@echo ""
	@bash tools/build_deploy_bundle.sh
	@bash tools/validate_deploy_bundle.sh
	@echo ""
	@AUTO_UPDATE=1 bash tools/deploy_ssh.sh $(HOST)

clean-bundle:
	@rm -rf dist/deploy_bundle
	@echo "Bundle limpiado"

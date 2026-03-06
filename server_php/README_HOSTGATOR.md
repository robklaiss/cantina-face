# Cantina Face PHP Backend – Guía HostGator

Esta guía explica cómo desplegar el backend PHP en un hosting compartido HostGator (Apache + PHP 8.x + MySQL/MariaDB). El repositorio original también contiene el backend FastAPI; este documento cubre exclusivamente el backend PHP ubicado en `server_php/`.

## 1. Arquitectura resumida

- `public/` → raíz web (coloca su contenido en `public_html/` o subcarpeta). Incluye `.htaccess` y `index.php` de arranque.
- `api/*.php` → endpoints JSON (`/api/auth.php`, `/api/parents.php`, etc.).
- `app/` → helpers (DB, sesiones, validadores, dominio, logging).
- `storage/` → logs y cachés (requiere permisos de escritura).
- `config/` → `config.php` con credenciales reales.
- `migrations/` → SQL para crear/actualizar el esquema MySQL.
- `tools/` → scripts CLI (`create_admin.php`, `self_test.php`).

## 2. Requisitos HostGator

1. **PHP 8.1 o superior** (MultiPHP Manager → seleccionar PHP 8.x para el dominio/subdominio).
2. **MySQL/MariaDB** con usuario y base creados desde cPanel → apuntar a `localhost`.
3. **Acceso SSH opcional** (facilita subir archivos y ejecutar migraciones). Si no está disponible, usar phpMyAdmin + Administrador de archivos.

## 3. Preparar archivos

1. Clona o descarga el repositorio localmente.
2. Copia la carpeta `server_php/` completa a tu máquina local.
3. Comprime su contenido (no incluya `.git`). Ejemplo: `zip -r cantina-php.zip server_php/*`.
4. Sube el zip a tu hosting (`public_html/server_php.zip`) usando FTP o Administrador de archivos.
5. Descomprime dentro de `public_html/` y renombra la carpeta si quieres algo como `cantina/`.

Estructura final recomendada (servidor):

```
public_html/
├── api/               # endpoints PHP
├── app/
├── config/
├── migrations/
├── public/            # mover contenido a raíz pública
├── storage/
├── tools/
└── ...
```

> **Importante:** el contenido de `public/` debe copiarse a `public_html/` (raíz web). El resto puede residir en `public_html/` o en una carpeta hermana y apuntar con `RewriteRule`.

## 4. Configuración (`config/config.php`)

1. Copia `config/config.example.php` a `config/config.php`.
2. Edita:
   - `app.url` → URL pública (ej. `https://padres.tudominio.com`).
   - `app.internal_token` → cadena aleatoria para scripts internos.
   - `db.*` → credenciales creadas en cPanel (host `localhost`).
   - Ajusta `timezone` a tu localidad (`America/Asuncion`).
3. Opcional: modifica `session.name` si vas a compartir dominio con otros sitios.

## 5. Migraciones MySQL

1. Entra a **phpMyAdmin** o usa **SSH**:
   - Ejecuta `migrations/001_init.sql` para crear tablas.
   - Ejecuta `migrations/002_indexes.sql` para índices secundarios.
2. Si lo haces vía SSH: `mysql -u USUARIO -p -h localhost BASE < migrations/001_init.sql`.

## 6. Permisos y carpetas

1. Asegura que `storage/`, `storage/logs/` y `storage/cache/` existan y sean escribibles (`chmod 775` o ajuste vía cPanel → Permisos 775).
2. Verifica que `.htaccess` (en `public/`) contenga las reglas de reescritura para rutas limpias:

```
RewriteEngine On
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule ^ index.php [L]
```

3. Si ubicaste los archivos fuera de `public_html`, ajusta rutas en `.htaccess` o crea un symlink.

## 7. Autenticación y sesiones

- Las sesiones PHP usan `storage/sessions` (carpeta se crea automáticamente). Debe ser escribible.
- Para CSRF en logout/post, el endpoint `auth.php?action=me` retorna `csrf_token`.

## 8. Scripts útiles (ejecutar por SSH)

```bash
# Crear admin/cajera
php tools/create_admin.php --email=admin@colegio.com --password=TuPass123 --role=admin

# Prueba rápida del entorno
php tools/self_test.php
```

## 9. Integración con frontends

- Las SPA existentes (`project/static/parents`, `deploy-gator/backend`, etc.) deben apuntar a los endpoints PHP (ej. `/api/parents.php?action=students`).
- Si sirven los frontends desde el mismo dominio HostGator, mantén `API_BASE = ""`.
- Para dominios separados, coloca la URL completa (ej. `https://api.tudominio.com/api`).

### Mapeo rápido de endpoints

| FastAPI anterior | PHP actual |
|------------------|------------|
| `/auth/token`    | `/api/auth.php?action=login` |
| `/auth/me`       | `/api/auth.php?action=me` |
| `/api/parents/*` | `/api/parents.php?action=...` |
| `/api/backend/*` | `/api/backend.php?action=...` |
| `/api/students`  | `/api/students.php?action=list` (más acciones) |
| `/api/transactions` | `/api/transactions.php?action=list` |
| `/api/health/timing` | `/api/health.php?action=timing` |

## 10. Verificación final

1. Abre `https://tu-dominio.com/index.php` → debería devolver `{ ok: true, message: "Cantina Face API" }` (si configuras un index básico).
2. Llama `https://tu-dominio.com/api/health.php?action=timing` (requiere login) para validar métricas.
3. Inicia sesión desde los frontends (`/padres`, `/backend`) y confirma que leen/escriben datos.

## 11. Mantenimiento

- **Logs**: revisa `storage/logs/app-YYYY-MM-DD.log` para errores.
- **Actualizaciones**: reemplaza archivos del backend y ejecuta nuevas migraciones si aparecen. Respeta `config.php` y `storage/`.
- **Backups**: exporta la base MySQL (cPanel → Backup Wizard) y respalda `storage/` periódicamente.

---

¿Problemas? Ejecuta `php tools/self_test.php` y revisa `storage/logs/`. Para dudas adicionales, documenta el error y credenciales (no las compartas públicamente) antes de pedir soporte.


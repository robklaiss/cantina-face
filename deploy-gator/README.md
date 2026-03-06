# Deploy en HostGator (hosting compartido)

Guía para subir los frontends estáticos de Cantina Face (padres/backend) y los artefactos de actualización de la caja a un hosting compartido HostGator (Apache + cPanel). Aquí no se instala FastAPI ni servicios de sistema; solo se sirven archivos estáticos y se apuntan al backend PHP desplegado en `server_php/` (ver `server_php/README_HOSTGATOR.md`).

El servidor HostGator queda a cargo de:

- **Portal de Padres**
- **Panel Backend**
- **Manifest y ZIP de actualizaciones** para las cajas locales

## Portal de Padres

Una vez desplegado, el portal queda accesible en:

| URL | Descripción |
|-----|-------------|
| `https://sistema.siloe.com.py/padres` | Portal de padres (login, saldos, pedidos, menú) |
| `https://sistema.siloe.com.py/backend` | Backend general (usuarios, cajeras, stock, vínculos, cargas) |
| `https://sistema.siloe.com.py/admin.html` | Panel de administración clásico |
| `https://sistema.siloe.com.py/docs` | Documentación interactiva de la API |
| `https://sistema.siloe.com.py/updates/manifest.json` | Manifest de actualizaciones para la caja |

Estos frontends se sirven desde Apache (HostGator) como archivos estáticos; todas las llamadas de API se envían al backend PHP alojado en el mismo dominio o subdominio.

## Frontends estáticos (backend/, padres/)

Dentro de `deploy-gator/` hay frontends 100% estáticos (HTML+JS+CSS) listos para cualquier hosting compartido:

```
deploy-gator/
├── backend/
│   ├── index.html         ← panel admin
│   ├── backend.css
│   ├── backend.js
│   ├── config.js          ← ⚠️ EDITAR con URL del API
│   ├── siloe-logo-blanco.png
│   └── default-avatar.png
├── padres/
│   ├── index.html         ← portal padres
│   ├── styles.css
│   ├── app.js
│   ├── config.js          ← ⚠️ EDITAR con URL del API
│   └── siloe-logo-blanco.png
├── updates/
│   ├── manifest.json      ← ⚠️ EDITAR versión + sha256
│   └── cantina-face/
│       └── project.zip    ← subir aquí cada nueva versión
└── .htaccess               ← reglas pensadas para Apache HostGator

### Configurar la URL del API

Editá los archivos `config.js` en `backend/` y `padres/` para que apunten al API:

- **`backend/config.js`** → `API_BASE: ""` (vacío = mismo servidor)
- **`padres/config.js`** → idem

> Dejar `API_BASE` vacío significa que los frontends llaman al API en el mismo dominio (HostGator). Solo cambiarlo si servís estos HTML desde otro host o subdominio.

### Dónde subir cada carpeta

1. Ingresá a **cPanel → Administrador de archivos**.
2. Dentro de `public_html/` (o el subdominio que uses) creá directorios `padres/`, `backend/` y `updates/`.
3. Subí el contenido de `deploy-gator/padres/` dentro de `public_html/padres/` (puedes comprimirlo en `.zip` y extraerlo desde cPanel para acelerar).
4. Repite el proceso con `deploy-gator/backend/` y `deploy-gator/updates/`.
5. Coloca el `.htaccess` en la raíz donde vivan `padres/` y `backend/` para endurecer Apache.

> **Tip:** si querés servirlos desde subdominios separados (`padres.midominio.com`, `backend.midominio.com`), crea subdominios en cPanel y repite el paso de copia respetando las carpetas de destino.

### Actualización de la caja (updates/)

Cada vez que generes un nuevo `project.zip`:

1. Calculá el SHA256:
   ```bash
   sha256sum project.zip
   ```
2. Subí `project.zip` a `updates/cantina-face/`.
3. Editá `updates/manifest.json` con la nueva `version`, `released`, `sha256`.

### Seguridad (.htaccess)

El `.htaccess` incluido y pensado para HostGator:
- Bloquea acceso a `*.sqlite`, `*.db`, `*.py`, `*.env`, archivos ocultos
- Bloquea rutas `/app`, `/database`, `/data`
- Desactiva el listado de directorios
- Agrega headers CORS básicos para permitir llamadas al API
- Aplica caché de 1 semana a assets estáticos

Colócalo en la misma carpeta donde viven `padres/` y `backend/` (por ejemplo `public_html/`). Ajusta reglas si usas subdominios con DocumentRoot distinto.

## Requisitos HostGator

- Plan compartido con **PHP 8.x y Apache** (cualquier plan Baby/Business funciona).
- Acceso a **cPanel** y al **Administrador de archivos** (o FTP/SFTP).
- (Opcional) Acceso SSH para automatizar el upload.
- El backend PHP (`server_php/`) debe estar desplegado en el mismo dominio o en un subdominio accesible desde los frontends.

## Pasos resumidos

1. Prepara el backend PHP siguiendo `server_php/README_HOSTGATOR.md`.
2. Subí `deploy-gator/padres/` y `deploy-gator/backend/` a `public_html/` o al subdominio deseado.
3. Editá `padres/config.js` y `backend/config.js` según corresponda.
4. Copiá `deploy-gator/updates/` (incluyendo `manifest.json`) a `public_html/updates/`.
5. Protegé la carpeta `updates/` con un archivo `index.html` simple o reglas extra, si no querés listados públicos (HostGator respeta `Options -Indexes`).
6. Verificá accesos: `https://tu-dominio/padres`, `https://tu-dominio/backend` y `https://tu-dominio/updates/manifest.json`.

## Actualizar los frontends

Cada vez que modifiques archivos estáticos:

1. Actualiza localmente la carpeta (`padres/`, `backend/` o `updates/`).
2. Comprime solo la carpeta cambiada en `.zip`.
3. Sube el zip vía cPanel/FTP y extrae dentro de la carpeta remota.
4. Limpia archivos viejos si ya no se usan (cPanel → seleccionar → eliminar).

## ¿Y el archivo `install.sh`?

Permanece en el repo por compatibilidad histórica (despliegues EC2). **No se ejecuta en HostGator** porque requiere acceso root y paquetes de Ubuntu. Ignóralo cuando trabajes con hosting compartido.

## Checklist de verificación

| Ítem | Cómo validar |
|------|--------------|
| Config.js apunta al backend correcto | Abrí la consola del navegador y verify los `fetch` a `/api/*.php` responden 200 |
| Portal Padres carga assets | Comprueba que `styles.css` y `app.js` responden 200 en el navegador |
| Backend Admin carga | Navega a `/backend` y revisa que el login funcione |
| Manifest accesible | `curl https://tu-dominio/updates/manifest.json` debe devolver JSON |
| project.zip descargable | Verifica un HEAD/GET al archivo (opcionalmente protegelo con auth básica si el hosting lo permite) |

## Compatibilidad con HostGator

- Todo el contenido son archivos estáticos (HTML/CSS/JS/JSON) y un `.htaccess`, por lo que **no requiere Node, Python ni procesos residentes**.
- Las únicas dependencias del servidor son Apache + PHP (para el backend PHP mencionado) y la posibilidad de servir archivos grandes (~100 MB para `project.zip`).
- Si necesitas más control (por ejemplo, versiones múltiples del ZIP), crea subcarpetas dentro de `updates/` manteniendo el `manifest.json` apuntando a la versión vigente.

Con esto, el paquete `deploy-gator/` queda alineado con un despliegue en HostGator compartido.

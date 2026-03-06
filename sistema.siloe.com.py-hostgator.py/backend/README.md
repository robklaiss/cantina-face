# Deploy-Gator Backend (HostGator friendly)

Mini backend en PHP nativo para subir releases `project.zip` y publicar la última versión consumida por la caja (`caja-siloe`).

## Archivos clave

```
deploy-gator/backend/
├── config.php          ← coloca aquí el token (DEPLOY_TOKEN)
├── index.php           ← panel mínimo (subir + publicar)
├── upload.php          ← endpoint POST para subir ZIP
├── publish.php         ← endpoint POST para publicar latest
├── latest.json         ← metadata de la versión publicada
└── releases/           ← se guardan los `project_YYYYMMDD_HHMMSS.zip`
```

## Cambiar el token

Abre `config.php` y reemplaza `CHANGE_ME_DEPLOY_TOKEN` por un secreto fuerte. Es el mismo valor que se envía en los formularios/curl como `token`.

## Probar localmente

```
php -S 127.0.0.1:8001 -t deploy-gator/backend
```

Luego visita <http://127.0.0.1:8001/index.php> para usar el panel o llama a los endpoints con `curl` (ver abajo).

## Flujo básico

1. Genera `project.zip` desde la caja / build pipeline.
2. `POST upload.php` con el token + archivo → guarda `releases/project_YYYYMMDD_HHMMSS.zip` y responde con metadata (`version`, `sha256`, `bytes`).
3. `POST publish.php` con el token + nombre del archivo (por ejemplo `project_20240211_180530.zip`). Esto escribe `latest.json`.
4. La caja consulta `GET latest.json` para saber qué release descargar.

## Endpoints

| Método | Ruta                        | Uso |
|--------|-----------------------------|-----|
| GET    | `/deploy-gator/backend/index.php`  | Panel HTML para subir/publicar |
| POST   | `/deploy-gator/backend/upload.php` | Subir `project.zip` |
| POST   | `/deploy-gator/backend/publish.php`| Publicar un release existente como latest |
| GET    | `/deploy-gator/backend/latest.json`| Metadata pública consumida por la caja |

Todas las acciones POST requieren `token` en el body (mismo valor que `DEPLOY_TOKEN`).

## Ejemplos `curl`

Subir un ZIP:

```bash
curl -X POST \
  -F "token=SUPER_SECRETO" \
  -F "zipfile=@project.zip" \
  https://tu-dominio.com/deploy-gator/backend/upload.php
```

Publicar un release existente:

```bash
curl -X POST \
  -F "token=SUPER_SECRETO" \
  -F "filename=project_20240211_180530.zip" \
  https://tu-dominio.com/deploy-gator/backend/publish.php
```

Consultar `latest.json` (no requiere token):

```bash
curl https://tu-dominio.com/deploy-gator/backend/latest.json | jq
```

## Notas

- Solo acepta archivos con extensión `.zip`.
- Los uploads se renombran automáticamente usando hora UTC.
- `latest.json` se escribe con `JSON_PRETTY_PRINT` y siempre tiene las llaves (`version`, `filename`, `sha256`, `bytes`, `uploaded_at`).
- Este backend no depende de frameworks ni composer, ideal para hosting compartido (Apache + PHP 8.x).

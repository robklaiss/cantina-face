# HostGator Sync Report

## Objetivo de esta fase
Sincronizar de forma conservadora el repo local con la snapshot de HostGator, trayendo solo cambios funcionales confirmados y evitando secretos, bases reales, scripts temporales y archivos basura.

## Archivos actualizados

### Backend PHP
- `server_php/api/sync.php`
  - Se agregó autenticación por token interno o admin.
  - El sync de alumnos ahora acepta el identificador externo en `students[].id`.
  - El upsert usa `external_id` cuando la columna existe.
  - Se agregó compatibilidad transicional para esquemas viejos donde el identificador estable todavía vive en `students.id` o `students.identifier`.
  - El `GET` de sync ahora devuelve `external_id` cuando está disponible.

- `server_php/api/students.php`
  - Se agregó acceso por token interno o roles existentes.
  - El listado de alumnos ahora expone `external_id` cuando existe.
  - La consulta es tolerante a bases que todavía no tengan `external_id` o `updated_at`.

- `server_php/api/backend.php`
  - El listado de solicitudes de vínculo ahora intenta resolver y enriquecer cada solicitud con:
    - `student_internal_id`
    - `student_external_id`
    - `student_code`
  - La aprobación de vínculo ahora acepta referencia interna o externa.
  - La resolución del alumno intenta por:
    - `student_id` enviado
    - `student_identifier` guardado en la solicitud
    - coincidencia por nombre/grado
  - Se preservó una salvaguarda importante: `parent_student.student_id` sigue guardando el `students.id` interno para no romper integridad referencial.

### Dominio y helpers
- `server_php/app/helpers.php`
  - Se agregó `db_table_has_column()` para introspección segura de columnas en SQLite/MySQL.

- `server_php/app/domain.php`
  - `format_student_record()` ahora expone `external_id`.
  - `update_student()` pasó a usar timestamp portable con `now()`.

### Esquema SQLite
- `server_php/app/schema.sql`
  - `students.external_id`
  - `students.updated_at`
  - `topup_requests.admin_notes`
  - `topup_requests.processed_at`
  - `topup_requests.updated_at`
  - `link_requests.processed_at`

- `server_php/app/db_schema.php`
  - Se agregó auto-reparación conservadora para SQLite existente.
  - Si la base ya existe pero le faltan columnas nuevas, se agregan sin recrear tablas.
  - Se crea índice único para `students.external_id` si corresponde.

### Migraciones MySQL
- `server_php/migrations/001_init.sql`
  - Se agregó `students.external_id` para instalaciones nuevas.

- `server_php/migrations/003_add_students_external_id.sql`
  - Nueva migración incremental para instalaciones MySQL existentes.

### Frontend admin
- `deploy-gator-dist/backend/backend.js`
  - El listado de alumnos muestra el ID largo (`external_id`) cuando existe.
  - El botón copiar copia `external_id` y usa `id` solo como fallback.
  - La aprobación de vínculo precompleta el prompt con `student_code` o `student_external_id`.
  - El valor enviado al backend ya no se fuerza a entero.

## Archivos deliberadamente ignorados
- `sistema.siloe.com.py-hostgator/.../config.php`
- bases reales SQLite
- archivos `error_log`
- archivos `.bak_*`
- scripts temporales/debug/reset/migración ad hoc
- cambios no esenciales en UI fuera del flujo de mostrar/copiar ID largo y aprobar vínculo por referencia externa
- `parents.php` de la snapshot

## Riesgos detectados y mitigación
- La snapshot mezcla lógica nueva de `external_id` con esquemas no totalmente alineados.
  - Mitigación: se agregó compatibilidad transicional por columna presente (`external_id`, `identifier` o `id`).

- En producción existía riesgo de terminar usando `external_id` dentro de `parent_student.student_id`.
  - Mitigación: en este repo se forzó explícitamente que la relación siga usando el `id` interno del alumno.

- SQLite local venía con desfase respecto al código actual (`processed_at`, `admin_notes`, `updated_at`).
  - Mitigación: se actualizó `schema.sql` y además `db_schema.php` ahora agrega columnas faltantes en bases existentes.

- La migración MySQL incremental `003_add_students_external_id.sql` no es idempotente si se ejecuta dos veces.
  - Mitigación: está pensada como migración manual de una sola ejecución.

## Validación realizada
- `php -l server_php/api/sync.php`
- `php -l server_php/api/students.php`
- `php -l server_php/api/backend.php`
- `php -l server_php/app/domain.php`
- `php -l server_php/app/db_schema.php`

Todos esos archivos pasaron validación de sintaxis.

## Paso manual pendiente
- Si el entorno productivo o staging usa MySQL existente, ejecutar una sola vez:
  - `server_php/migrations/003_add_students_external_id.sql`

## Cambios no aplicados en esta fase
- No se tocaron secretos ni configuración real.
- No se copiaron scripts temporales de la snapshot.
- No se hicieron mejoras nuevas de UI no relacionadas con el sync conservador.

## Resultado
La base local quedó alineada con los cambios funcionales confirmados de producción para:
- sync con token interno
- soporte de `students.external_id`
- aprobación de vínculos por referencia interna/externa
- `link_requests.processed_at`
- exposición de `external_id` en endpoints
- visualización/copiado del ID largo en backend admin

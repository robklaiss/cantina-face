<?php

require_once __DIR__ . '/../app/bootstrap.php';

$action = strtolower((string) input('action', 'stats'));
$currentUser = require_auth([ROLE_ADMIN, ROLE_STOCK, ROLE_CAJERA]);

switch ($action) {
    case 'stats':
        ensure_roles([ROLE_ADMIN, ROLE_STOCK, ROLE_CAJERA]);
        json_ok(backend_stats());
        break;

    case 'users':
        ensure_roles([ROLE_ADMIN]);
        handle_users();
        break;

    case 'user':
        ensure_roles([ROLE_ADMIN]);
        handle_user_detail();
        break;

    case 'user_reset':
        ensure_roles([ROLE_ADMIN]);
        handle_user_reset();
        break;

    case 'link_requests':
        ensure_roles([ROLE_ADMIN]);
        handle_link_requests();
        break;

    case 'link_request_decision':
        ensure_roles([ROLE_ADMIN]);
        handle_link_request_decision();
        break;

    case 'topups':
        ensure_roles([ROLE_ADMIN, ROLE_STOCK]);
        handle_topups();
        break;

    case 'topup_decision':
        ensure_roles([ROLE_ADMIN, ROLE_STOCK]);
        handle_topup_decision();
        break;

    case 'trigger_update':
        ensure_roles([ROLE_ADMIN]);
        handle_trigger_update();
        break;

    case 'update_status':
        ensure_roles([ROLE_ADMIN]);
        handle_update_status();
        break;

    default:
        json_error('Acción no soportada', 404);
}

function ensure_roles(array $roles): void
{
    $user = current_user();
    if (!$user || !in_array($user['role'], $roles, true)) {
        json_error('Permisos insuficientes', 403);
    }
}

function require_method(string $method): void
{
    if (http_method() !== strtoupper($method)) {
        json_error('Método no permitido', 405);
    }
}

function handle_users(): void
{
    if (http_method() === 'GET') {
        json_ok(list_users_admin());
        return;
    }

    if (http_method() === 'POST') {
        $payload = request_json();
        json_ok(create_user_admin($payload), 201);
        return;
    }

    json_error('Método no soportado', 405);
}

function handle_user_detail(): void
{
    $userId = parse_int(input('id'));
    if (!$userId) {
        json_error('id requerido', 422);
    }

    if (http_method() === 'GET') {
        $user = find_user_admin($userId);
        if (!$user) {
            json_error('Usuario no encontrado', 404);
        }
        json_ok($user);
        return;
    }

    if (in_array(http_method(), ['POST', 'PUT', 'PATCH'], true)) {
        $payload = request_json();
        json_ok(update_user_admin($userId, $payload));
        return;
    }

    json_error('Método no soportado', 405);
}

function handle_user_reset(): void
{
    require_method('POST');
    $userId = parse_int(input('id'));
    if (!$userId) {
        json_error('id requerido', 422);
    }
    $body = request_json();
    require_fields($body, ['new_password']);
    json_ok(reset_user_password_admin($userId, $body['new_password']));
}

function handle_link_requests(): void
{
    $status = sanitize_string(input('status'));
    $requests = list_link_requests_filtered(null, $status ?: null);

    // Enriquecer cada request con IDs del alumno (internal/external) si se puede resolver
    $enriched = [];
    foreach ($requests as $req) {
        try {
            $student = resolve_student_for_link_request($req);
            $req['student_internal_id'] = $student['id'] ?? null;
            $req['student_external_id'] = $student['external_id'] ?? null;
            // “student_code” = el que se usa para vincular desde la caja (UUID/external_id)
            $req['student_code'] = $student['external_id'] ?? null;
        } catch (Throwable $e) {
            $req['student_internal_id'] = null;
            $req['student_external_id'] = null;
            $req['student_code'] = null;
        }
        $enriched[] = $req;
    }

    json_ok($enriched);
}

function handle_link_request_decision(): void
{
    try {
        require_method('POST');
        $requestId = parse_int(input('id'));
        if (!$requestId) {
            json_error('id requerido', 422);
        }

        $body = request_json();
        require_fields($body, ['decision']);

        $decision = validate_enum($body['decision'], ['approved', 'rejected'], 'decision');
        $studentRef = isset($body['student_id']) ? trim((string) $body['student_id']) : null;
        $notes = sanitize_string($body['admin_notes'] ?? null);

        $pdo = db();

        $stmt = $pdo->prepare('SELECT * FROM link_requests WHERE id = :id');
        $stmt->execute(['id' => $requestId]);
        $request = $stmt->fetch(PDO::FETCH_ASSOC);

        if (!$request) {
            json_error('Solicitud no encontrada', 404);
        }

        $studentIdForUpdate = null;

        if ($decision === 'approved') {
            $student = null;

            if ($studentRef !== null && $studentRef !== '') {
                $stmt = $pdo->prepare('
                    SELECT id, external_id, name, grade
                    FROM students
                    WHERE external_id = :ext
                    LIMIT 1
                ');
                $stmt->execute(['ext' => $studentRef]);
                $student = $stmt->fetch(PDO::FETCH_ASSOC);

                if (!$student && ctype_digit($studentRef)) {
                    $stmt = $pdo->prepare('
                        SELECT id, external_id, name, grade
                        FROM students
                        WHERE id = :id
                        LIMIT 1
                    ');
                    $stmt->execute(['id' => (int) $studentRef]);
                    $student = $stmt->fetch(PDO::FETCH_ASSOC);
                }
            }

            if (!$student) {
                $stmt = $pdo->prepare('
                    SELECT id, external_id, name, grade
                    FROM students
                    WHERE name = :name AND grade = :grade
                    LIMIT 1
                ');
                $stmt->execute([
                    'name' => $request['student_name'] ?? '',
                    'grade' => $request['student_grade'] ?? '',
                ]);
                $student = $stmt->fetch(PDO::FETCH_ASSOC);
            }

            if (
                !$student &&
                !empty($request['student_name']) &&
                !empty($request['student_grade'])
            ) {
                $variants = array_values(array_unique([
                    $request['student_name'],
                    str_replace('-', ' ', $request['student_name']),
                    str_replace(' ', '-', $request['student_name']),
                ]));

                foreach ($variants as $variant) {
                    $stmt = $pdo->prepare('
                        SELECT id, external_id, name, grade
                        FROM students
                        WHERE name = :name AND grade = :grade
                        LIMIT 1
                    ');
                    $stmt->execute([
                        'name' => $variant,
                        'grade' => $request['student_grade'],
                    ]);
                    $student = $stmt->fetch(PDO::FETCH_ASSOC);
                    if ($student) {
                        break;
                    }
                }
            }

            if (!$student) {
                json_error('No se pudo resolver el alumno para aprobar el vínculo', 422, [
                    'student_ref' => $studentRef,
                    'student_name' => $request['student_name'] ?? null,
                    'student_grade' => $request['student_grade'] ?? null,
                ]);
            }

            $studentIdForUpdate = (int) $student['id'];
        }

        update_link_request_status($requestId, $decision, $studentIdForUpdate, $notes);

        $stmt = $pdo->prepare('SELECT * FROM link_requests WHERE id = :id');
        $stmt->execute(['id' => $requestId]);
        $updated = $stmt->fetch(PDO::FETCH_ASSOC);

        if ($decision === 'approved' && $request) {
            $parent = find_user_by_id((int) $request['parent_id']);
            if ($parent && !empty($parent['email'])) {
                $firstName = $parent['first_name'] ?? ($parent['full_name'] ?? 'Padre/Madre');
                send_link_approved_email($parent['email'], $firstName, $request['student_name']);
            }
        }

        json_ok($updated);

    } catch (Throwable $e) {
        json_error('DEBUG link_request_decision: ' . $e->getMessage(), 500, [
            'file' => $e->getFile(),
            'line' => $e->getLine(),
        ]);
    }
}
{
    require_method('POST');
    $requestId = parse_int(input('id'));
    if (!$requestId) {
        json_error('id requerido', 422);
    }

    $body = request_json();
    require_fields($body, ['decision']);

    $decision = validate_enum($body['decision'], ['approved', 'rejected'], 'decision');

    // Puede venir como “student_id” desde el front; acá aceptamos:
    // - internal id (int)
    // - external_id (uuid string)
    // - vacío (y resolvemos por nombre/grado/identifier)
    $studentRef = isset($body['student_id']) ? trim((string) $body['student_id']) : '';
    $notes = sanitize_string($body['admin_notes'] ?? null);

    // Fetch request before updating so we have parent_id and student_name
    $request = db_fetch('SELECT * FROM link_requests WHERE id = :id', ['id' => $requestId]);
    if (!$request) {
        json_error('Solicitud no encontrada', 404);
    }

    // Si aprueba, intentar resolver alumno (internal/external) y guardar el “código” correcto
    $studentIdToStore = null;

    if ($decision === 'approved') {
        $student = resolve_student_for_link_request($request, $studentRef);

        if (!$student) {
            json_error('No se pudo resolver el alumno para aprobar el vínculo. Verificá nombre/grado o seleccioná un alumno.', 422);
        }

        // Guardamos “student_identifier” como external_id (UUID) si existe;
        // si no, caemos a internal id (int) como string.
        $studentIdToStore = $student['external_id'] ?? (string) $student['id'];
    }

    update_link_request_status($requestId, $decision, $studentIdToStore, $notes);
    $updated = db_fetch('SELECT * FROM link_requests WHERE id = :id', ['id' => $requestId]);

    // Send approval email to parent
    if ($decision === 'approved' && $request) {
        $parent = find_user_by_id((int) $request['parent_id']);
        if ($parent && $parent['email']) {
            $firstName = $parent['first_name'] ?? ($parent['full_name'] ?? 'Padre/Madre');
            send_link_approved_email($parent['email'], $firstName, $request['student_name']);
        }
    }

    json_ok($updated);
}

function handle_topups(): void
{
    $status = sanitize_string(input('status'));
    json_ok(list_all_topups($status ?: null));
}

function handle_topup_decision(): void
{
    require_method('POST');
    $topupId = parse_int(input('id'));
    if (!$topupId) {
        json_error('id requerido', 422);
    }
    $body = request_json();
    require_fields($body, ['decision']);
    $decision = validate_enum($body['decision'], ['approved', 'rejected'], 'decision');
    $reference = sanitize_string($body['payment_reference'] ?? null);
    $notes = sanitize_string($body['admin_notes'] ?? null);
    $result = process_topup_decision($topupId, $decision, $reference, $notes);
    json_ok($result);
}

/**
 * Check all students linked to a parent and send low-credits email
 * if any balance is below the threshold. Called after any balance change.
 */
function maybe_send_low_credits_emails(int $parentId, float $threshold = 5000): void
{
    $parent = find_user_by_id($parentId);
    if (!$parent || !$parent['email']) {
        return;
    }
    $students = get_parent_students($parentId);
    foreach ($students as $student) {
        $balance = (float) ($student['balance'] ?? 0);
        if ($balance < $threshold) {
            $firstName = $parent['first_name'] ?? ($parent['full_name'] ?? 'Padre/Madre');
            send_low_credits_email($parent['email'], $firstName, $student['name'], $balance);
        }
    }
}

function handle_trigger_update(): void
{
    require_method('POST');
    $result = trigger_caja_update();
    if ($result['success']) {
        json_ok($result);
    } else {
        json_error($result['message'], 500);
    }
}

function handle_update_status(): void
{
    require_method('GET');
    $result = check_caja_update_status();
    if ($result['success']) {
        json_ok($result['data']);
    } else {
        json_error($result['message'], 500);
    }
}

/**
 * Resolver alumno para un link_request.
 * Devuelve un array con columnas de la tabla students: id, external_id, name, grade, balance, etc.
 *
 * Reglas:
 * - Si viene $selectedStudentRef (desde UI), lo intenta primero.
 * - Si link_requests.student_identifier está seteado, lo intenta.
 * - Si no, intenta por (student_name + student_grade) con variantes simples (guion/espacio/_).
 * - Fallback por name sola (con variantes).
 */
function resolve_student_for_link_request(array $request, string $selectedStudentRef = ''): ?array
{
    $pdo = db();

    $selectedStudentRef = trim($selectedStudentRef);
    if ($selectedStudentRef !== '') {
        $student = find_student_by_any_reference($pdo, $selectedStudentRef);
        if ($student) {
            return $student;
        }
    }

    $identifier = isset($request['student_identifier']) ? trim((string) $request['student_identifier']) : '';
    if ($identifier !== '') {
        $student = find_student_by_any_reference($pdo, $identifier);
        if ($student) {
            return $student;
        }
    }

    $name = isset($request['student_name']) ? trim((string) $request['student_name']) : '';
    $grade = isset($request['student_grade']) ? trim((string) $request['student_grade']) : '';

    // 1) exacto por nombre + grado
    if ($name !== '' && $grade !== '') {
        $st = $pdo->prepare('
            SELECT id, external_id, name, grade, balance, photo_path, created_at
            FROM students
            WHERE name = :name AND grade = :grade
            LIMIT 1
        ');
        $st->execute([
            'name' => $name,
            'grade' => $grade,
        ]);
        $row = $st->fetch(PDO::FETCH_ASSOC);
        if ($row) {
            return $row;
        }
    }

    // 2) fallback: variantes con espacio/guion/underscore (nombre + grado)
    if ($name !== '' && $grade !== '') {
        $variants = array_values(array_unique([
            $name,
            str_replace('-', ' ', $name),
            str_replace(' ', '-', $name),
            str_replace('_', ' ', $name),
            str_replace(' ', '_', $name),
        ]));

        foreach ($variants as $variant) {
            $st = $pdo->prepare('
                SELECT id, external_id, name, grade, balance, photo_path, created_at
                FROM students
                WHERE name = :name AND grade = :grade
                LIMIT 1
            ');
            $st->execute([
                'name' => $variant,
                'grade' => $grade,
            ]);
            $row = $st->fetch(PDO::FETCH_ASSOC);
            if ($row) {
                return $row;
            }
        }
    }

    // 3) exacto por nombre solo
    if ($name !== '') {
        $st = $pdo->prepare('
            SELECT id, external_id, name, grade, balance, photo_path, created_at
            FROM students
            WHERE name = :name
            LIMIT 1
        ');
        $st->execute(['name' => $name]);
        $row = $st->fetch(PDO::FETCH_ASSOC);
        if ($row) {
            return $row;
        }
    }

    // 4) fallback por nombre solo con variantes
    if ($name !== '') {
        $variants = array_values(array_unique([
            $name,
            str_replace('-', ' ', $name),
            str_replace(' ', '-', $name),
            str_replace('_', ' ', $name),
            str_replace(' ', '_', $name),
        ]));

        foreach ($variants as $variant) {
            $st = $pdo->prepare('
                SELECT id, external_id, name, grade, balance, photo_path, created_at
                FROM students
                WHERE name = :name
                LIMIT 1
            ');
            $st->execute(['name' => $variant]);
            $row = $st->fetch(PDO::FETCH_ASSOC);
            if ($row) {
                return $row;
            }
        }
    }

    return null;
}

/**
 * Buscar alumno por:
 * - id (int)
 * - external_id (uuid string)
 * - name exacto (último recurso)
 */
function find_student_by_any_reference(PDO $pdo, string $ref): ?array
{
    $ref = trim($ref);
    if ($ref === '') return null;

    // 1) si es numérico => students.id
    if (ctype_digit($ref)) {
        $st = $pdo->prepare('
            SELECT id, external_id, name, grade, balance, photo_path, created_at
            FROM students
            WHERE id = :id
            LIMIT 1
        ');
        $st->execute(['id' => (int) $ref]);
        $row = $st->fetch(PDO::FETCH_ASSOC);
        if ($row) return $row;
    }

    // 2) external_id
    $st = $pdo->prepare('
        SELECT id, external_id, name, grade, balance, photo_path, created_at
        FROM students
        WHERE external_id = :ext
        LIMIT 1
    ');
    $st->execute(['ext' => $ref]);
    $row = $st->fetch(PDO::FETCH_ASSOC);
    if ($row) return $row;

    // 3) por nombre exacto (fallback)
    $st = $pdo->prepare('
        SELECT id, external_id, name, grade, balance, photo_path, created_at
        FROM students
        WHERE name = :name
        LIMIT 1
    ');
    $st->execute(['name' => $ref]);
    $row = $st->fetch(PDO::FETCH_ASSOC);
    if ($row) return $row;

    return null;
}
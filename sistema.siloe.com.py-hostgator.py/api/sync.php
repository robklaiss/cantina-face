<?php
require_once __DIR__ . '/../app/bootstrap.php';

/**
 * DEBUG seguro
 */
if (isset($_GET['debug']) && $_GET['debug'] === '1') {
    @ini_set('display_errors', '1');
    @ini_set('display_startup_errors', '1');
    @error_reporting(E_ALL);

    header('Content-Type: application/json; charset=utf-8');

    $token = (string)($_SERVER['HTTP_X_INTERNAL_TOKEN'] ?? '');
    $preview = ($token !== '') ? (substr($token, 0, 4) . '...' . substr($token, -4)) : '';

    $raw = file_get_contents('php://input');
    $rawLen = is_string($raw) ? strlen($raw) : 0;

    $json = null;
    $jsonOk = false;
    $topKeys = null;
    $studentsCount = null;
    $firstStudentKeys = null;

    if (is_string($raw) && $raw !== '') {
        $json = json_decode($raw, true);
        $jsonOk = (json_last_error() === JSON_ERROR_NONE);
        if (is_array($json)) {
            $topKeys = array_keys($json);
            if (isset($json['students']) && is_array($json['students'])) {
                $studentsCount = count($json['students']);
                if (isset($json['students'][0]) && is_array($json['students'][0])) {
                    $firstStudentKeys = array_keys($json['students'][0]);
                }
            }
        }
    }

    echo json_encode([
        'ok' => true,
        'php_version' => PHP_VERSION,
        'method' => ($_SERVER['REQUEST_METHOD'] ?? null),
        'content_type' => ($_SERVER['CONTENT_TYPE'] ?? null),
        'has_internal_token' => ($token !== ''),
        'internal_token_preview' => $preview,
        'raw_len' => $rawLen,
        'json_ok' => $jsonOk,
        'json_top_keys' => $topKeys,
        'students_count' => $studentsCount,
        'first_student_keys' => $firstStudentKeys,
    ], JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
    exit;
}

$action = strtolower((string) input('action', 'students'));

if ($action === 'students' && http_method() === 'GET') {
    handle_sync_students_get();
    exit;
}

switch ($action) {
    case 'students':
        handle_sync_students_post();
        break;
    default:
        json_error('Acción no soportada', 404);
}

function handle_sync_students_post(): void
{
    if (http_method() !== 'POST') {
        json_error('Método no permitido', 405);
    }

    require_internal_or_admin();

    $body = request_json();
    if (!is_array($body) || !isset($body['students']) || !is_array($body['students'])) {
        json_error('Se requiere un array de estudiantes en body.students', 422);
    }

    $pdo = db();
    $students = $body['students'];

    $synced = 0;
    $errors = [];

    try {
        $pdo->beginTransaction();

        foreach ($students as $studentData) {
            try {
                if (!is_array($studentData)) {
                    throw new Exception('Formato de estudiante inválido (no es objeto)');
                }
                upsert_student_by_external_id($pdo, $studentData);
                $synced++;
            } catch (Throwable $e) {
                $errors[] = [
                    'student' => $studentData['name'] ?? ($studentData['id'] ?? 'unknown'),
                    'error' => $e->getMessage(),
                ];
            }
        }

        $pdo->commit();

        json_ok([
            'success' => true,
            'synced' => $synced,
            'total' => count($students),
            'errors' => $errors,
        ]);
    } catch (Throwable $e) {
        if ($pdo->inTransaction()) $pdo->rollBack();
        error_log('[sync_students_post] ' . $e->getMessage());
        json_error('Error en sincronización: ' . $e->getMessage(), 500);
    }
}

function handle_sync_students_get(): void
{
    require_internal_or_admin();

    try {
        $pdo = db();
        // mostramos id INTEGER + external_id UUID
        $stmt = $pdo->query('
            SELECT id, external_id, name, grade, balance, photo_path, created_at
            FROM students
            ORDER BY name
        ');
        $students = $stmt->fetchAll(PDO::FETCH_ASSOC);
        json_ok($students);
    } catch (Throwable $e) {
        error_log('[sync_students_get] ' . $e->getMessage());
        json_error('Internal error in sync get: ' . $e->getMessage(), 500);
    }
}

function require_internal_or_admin(): void
{
    $internalToken = $_SERVER['HTTP_X_INTERNAL_TOKEN'] ?? input('internal_token', '');
    $expectedToken = config('caja.internal_token', 'cantina-update-secret-2026');

    if (!is_string($internalToken)) $internalToken = '';
    if (!is_string($expectedToken)) $expectedToken = '';

    if ($internalToken !== '' && hash_equals($expectedToken, $internalToken)) {
        return;
    }

    require_auth([ROLE_ADMIN]);
}

/**
 * Upsert compatible: usa external_id (UUID) como clave estable.
 * Requiere columna students.external_id (TEXT) y un índice único recomendado.
 */
function upsert_student_by_external_id(PDO $pdo, array $data): void
{
    // En la caja, viene UUID en data['id']
    if (empty($data['id'])) throw new Exception('external_id requerido (viene en students[].id)');
    if (empty($data['name'])) throw new Exception('Nombre de estudiante requerido');

    $externalId = trim((string)$data['id']);
    $name = trim((string)$data['name']);
    $grade = sanitize_string($data['grade'] ?? null);
    $balance = isset($data['balance']) ? (float)$data['balance'] : 0.0;
    $photoPath = sanitize_string($data['photo_path'] ?? null);

    // buscar por external_id
    $stmt = $pdo->prepare('SELECT id FROM students WHERE external_id = :external_id LIMIT 1');
    $stmt->execute(['external_id' => $externalId]);
    $row = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($row) {
        $stmt = $pdo->prepare(
            'UPDATE students
             SET name = :name,
                 grade = :grade,
                 balance = :balance,
                 photo_path = :photo_path
             WHERE external_id = :external_id'
        );
        $stmt->execute([
            'external_id' => $externalId,
            'name' => $name,
            'grade' => $grade,
            'balance' => $balance,
            'photo_path' => $photoPath,
        ]);
    } else {
        // id se auto-genera (INTEGER)
        $stmt = $pdo->prepare(
            "INSERT INTO students (external_id, name, grade, balance, photo_path, created_at)
             VALUES (:external_id, :name, :grade, :balance, :photo_path, datetime('now'))"
        );
        $stmt->execute([
            'external_id' => $externalId,
            'name' => $name,
            'grade' => $grade,
            'balance' => $balance,
            'photo_path' => $photoPath,
        ]);
    }
}
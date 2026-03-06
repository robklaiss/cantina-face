<?php

require_once __DIR__ . '/../app/bootstrap.php';

$action = strtolower((string) input('action', 'students'));
if ($action === 'students' && http_method() === 'GET') {
    handle_sync_students_get();
    exit;
}

switch ($action) {
    case 'students':
        handle_sync_students();
        break;
    default:
        json_error('Acción no soportada', 404);
}

function handle_sync_students(): void
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
                    'error' => $e->getMessage()
                ];
            }
        }

        $pdo->commit();

        json_ok([
            'success' => true,
            'synced' => $synced,
            'total' => count($students),
            'errors' => $errors
        ]);

    } catch (Throwable $e) {
        if ($pdo->inTransaction()) {
            $pdo->rollBack();
        }
        error_log('[sync_students_post] ' . $e->getMessage());
        json_error('Error en sincronización: ' . $e->getMessage(), 500);
    }
}

function handle_sync_students_get(): void
{
    require_internal_or_admin();

    try {
        $pdo = db();
        $stableIdColumn = student_sync_id_column($pdo);
        $hasSeparateInternalId = student_sync_has_separate_internal_id($pdo);
        $selectStableId = $stableIdColumn !== null
            ? $stableIdColumn . ' AS external_id,'
            : (!$hasSeparateInternalId ? 'id AS external_id,' : 'NULL AS external_id,');
        $stmt = $pdo->query('SELECT id, ' . $selectStableId . ' name, grade, balance, photo_path, created_at FROM students ORDER BY name');
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

    if (!is_string($internalToken)) {
        $internalToken = '';
    }
    if (!is_string($expectedToken)) {
        $expectedToken = '';
    }

    if ($internalToken !== '' && hash_equals($expectedToken, $internalToken)) {
        return;
    }

    require_auth([ROLE_ADMIN]);
}

function upsert_student_by_external_id(PDO $pdo, array $data): void
{
    if (empty($data['id'])) {
        throw new Exception('external_id requerido (viene en students[].id)');
    }
    if (empty($data['name'])) {
        throw new Exception('Nombre de estudiante requerido');
    }

    $externalId = trim((string) $data['id']);
    $name = trim((string) $data['name']);
    $grade = sanitize_string($data['grade'] ?? null);
    $balance = isset($data['balance']) ? (float) $data['balance'] : 0.0;
    $photoPath = sanitize_string($data['photo_path'] ?? null);
    $stableIdColumn = student_sync_id_column($pdo);
    $hasSeparateInternalId = student_sync_has_separate_internal_id($pdo);

    if ($stableIdColumn !== null && $hasSeparateInternalId) {
        $stmt = $pdo->prepare('SELECT id FROM students WHERE ' . $stableIdColumn . ' = :external_id LIMIT 1');
        $stmt->execute(['external_id' => $externalId]);
        $exists = $stmt->fetch();
    } else {
        $stmt = $pdo->prepare('SELECT id FROM students WHERE id = :id LIMIT 1');
        $stmt->execute(['id' => $externalId]);
        $exists = $stmt->fetch();
    }

    if ($exists) {
        if ($stableIdColumn !== null && $hasSeparateInternalId) {
            $stmt = $pdo->prepare(
                'UPDATE students 
                 SET name = :name, grade = :grade, balance = :balance, photo_path = :photo
                 WHERE ' . $stableIdColumn . ' = :external_id'
            );
            $stmt->execute([
                'external_id' => $externalId,
                'name' => $name,
                'grade' => $grade,
                'balance' => $balance,
                'photo' => $photoPath
            ]);
            return;
        }

        if ($stableIdColumn !== null) {
            $stmt = $pdo->prepare(
                'UPDATE students 
                 SET external_id = :external_id, name = :name, grade = :grade, balance = :balance, photo_path = :photo
                 WHERE id = :id'
            );
            $stmt->execute([
                'id' => $externalId,
                'external_id' => $externalId,
                'name' => $name,
                'grade' => $grade,
                'balance' => $balance,
                'photo' => $photoPath
            ]);
            return;
        }

        $stmt = $pdo->prepare(
            'UPDATE students 
             SET name = :name, grade = :grade, balance = :balance, photo_path = :photo
             WHERE id = :id'
        );
        $stmt->execute([
            'id' => $externalId,
            'name' => $name,
            'grade' => $grade,
            'balance' => $balance,
            'photo' => $photoPath
        ]);
    } else {
        if ($stableIdColumn !== null && $hasSeparateInternalId) {
            $stmt = $pdo->prepare(
                'INSERT INTO students (' . $stableIdColumn . ', name, grade, balance, photo_path, created_at)
                 VALUES (:external_id, :name, :grade, :balance, :photo, :created_at)'
            );
            $stmt->execute([
                'external_id' => $externalId,
                'name' => $name,
                'grade' => $grade,
                'balance' => $balance,
                'photo' => $photoPath,
                'created_at' => now()
            ]);
            return;
        }

        if ($stableIdColumn !== null) {
            $stmt = $pdo->prepare(
                'INSERT INTO students (id, external_id, name, grade, balance, photo_path, created_at)
                 VALUES (:id, :external_id, :name, :grade, :balance, :photo, :created_at)'
            );
            $stmt->execute([
                'id' => $externalId,
                'external_id' => $externalId,
                'name' => $name,
                'grade' => $grade,
                'balance' => $balance,
                'photo' => $photoPath,
                'created_at' => now()
            ]);
            return;
        }

        $stmt = $pdo->prepare(
            'INSERT INTO students (id, name, grade, balance, photo_path, created_at)
             VALUES (:id, :name, :grade, :balance, :photo, :created_at)'
        );
        $stmt->execute([
            'id' => $externalId,
            'name' => $name,
            'grade' => $grade,
            'balance' => $balance,
            'photo' => $photoPath,
            'created_at' => now()
        ]);
    }
}

function student_sync_id_column(PDO $pdo): ?string
{
    if (db_table_has_column($pdo, 'students', 'external_id')) {
        return 'external_id';
    }

    if (db_table_has_column($pdo, 'students', 'identifier')) {
        return 'identifier';
    }

    return null;
}

function student_sync_has_separate_internal_id(PDO $pdo): bool
{
    $driver = (string) $pdo->getAttribute(PDO::ATTR_DRIVER_NAME);

    if ($driver === 'sqlite') {
        $stmt = $pdo->query('PRAGMA table_info(students)');
        $columns = $stmt ? $stmt->fetchAll(PDO::FETCH_ASSOC) : [];
        foreach ($columns as $info) {
            if (($info['name'] ?? null) === 'id') {
                return strtoupper((string) ($info['type'] ?? '')) === 'INTEGER';
            }
        }
        return false;
    }

    return true;
}

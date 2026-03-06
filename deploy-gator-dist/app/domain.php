<?php

declare(strict_types=1);

require_once __DIR__ . '/response.php';
require_once __DIR__ . '/auth/login.php';

function ensure_parent_student(int $parentId, string $studentId): void
{
    $stmt = db()->prepare(
        'SELECT 1 FROM parent_student WHERE parent_id = :pid AND student_id = :sid LIMIT 1'
    );
    $stmt->execute(['pid' => $parentId, 'sid' => $studentId]);
    $exists = $stmt->fetch();
    if (!$exists) {
        json_error('El alumno no está vinculado a este tutor', 403);
    }
}

function list_student_orders(int $studentId, ?string $status = null): array
{
    $params = ['sid' => $studentId];
    $sql = 'SELECT * FROM scheduled_orders WHERE student_id = :sid';
    if ($status) {
        $sql .= ' AND status = :status';
        $params['status'] = $status;
    }
    $sql .= ' ORDER BY scheduled_for ASC';
    $stmt = db()->prepare($sql);
    $stmt->execute($params);
    $orders = $stmt->fetchAll();
    if (!$orders) {
        return [];
    }
    $orderIds = array_column($orders, 'id');
    $stmt = db()->prepare(
        'SELECT * FROM scheduled_order_items WHERE order_id IN (' . implode(',', array_map('intval', $orderIds)) . ')'
    );
    $stmt->execute();
    $items = $stmt->fetchAll();
    $itemsByOrder = [];
    foreach ($items as $item) {
        $itemsByOrder[$item['order_id']][] = $item;
    }
    foreach ($orders as &$order) {
        $order['items'] = $itemsByOrder[$order['id']] ?? [];
    }
    return $orders;
}

function format_student_record(array $row): array
{
    $photo = $row['photo_path'] ?? null;
    return [
        'id' => (int) $row['id'],
        'name' => $row['name'],
        'grade' => $row['grade'] ?? '',
        'identifier' => $row['identifier'] ?? null,
        'balance' => isset($row['balance']) ? (float) $row['balance'] : 0.0,
        'photo_path' => $photo,
        'photo_url' => $photo ?: null,
        'created_at' => $row['created_at'] ?? null,
        'updated_at' => $row['updated_at'] ?? null,
    ];
}

function get_parent_students(int $parentId): array
{
    $stmt = db()->prepare(
        'SELECT s.* FROM students s
         INNER JOIN parent_student ps ON ps.student_id = s.id
         WHERE ps.parent_id = :pid
         ORDER BY s.name ASC'
    );
    $stmt->execute(['pid' => $parentId]);
    return $stmt->fetchAll();
}

function get_parent_topups(int $parentId): array
{
    $stmt = db()->prepare(
        'SELECT * FROM topup_requests WHERE parent_id = :pid ORDER BY created_at DESC'
    );
    $stmt->execute(['pid' => $parentId]);
    $rows = $stmt->fetchAll();
    return array_map('format_topup', $rows);
}

function resolve_allocation_details(?array $allocations): array
{
    if (!$allocations || !is_array($allocations) || !count($allocations)) {
        return [];
    }
    $ids = array_keys($allocations);
    $placeholders = implode(',', array_fill(0, count($ids), '?'));
    $stmt = db()->prepare("SELECT id, name, grade FROM students WHERE id IN ($placeholders)");
    $stmt->execute($ids);
    $map = [];
    foreach ($stmt->fetchAll() as $row) {
        $map[$row['id']] = ['name' => $row['name'], 'grade' => $row['grade']];
    }
    $details = [];
    foreach ($allocations as $sid => $amount) {
        $details[] = [
            'student_id' => $sid,
            'student_name' => $map[$sid]['name'] ?? null,
            'student_grade' => $map[$sid]['grade'] ?? null,
            'amount' => (float) $amount,
        ];
    }
    return $details;
}

function format_topup(array $row): array
{
    $allocations = $row['allocations_json'] ? json_decode($row['allocations_json'], true) : null;

    return [
        'id' => (int) $row['id'],
        'parent_id' => (int) $row['parent_id'],
        'parent_name' => $row['parent_name'] ?? null,
        'parent_email' => $row['parent_email'] ?? null,
        'parent_phone' => $row['parent_phone'] ?? null,
        'total_amount' => (float) $row['total_amount'],
        'allocation_mode' => $row['allocation_mode'],
        'allocations' => $allocations,
        'allocation_details' => resolve_allocation_details($allocations),
        'payment_reference' => $row['payment_reference'] ?? null,
        'status' => $row['status'],
        'admin_notes' => $row['admin_notes'] ?? null,
        'processed_at' => $row['processed_at'] ?? null,
        'created_at' => $row['created_at'],
        'updated_at' => $row['updated_at'],
    ];
}

function list_products(): array
{
    $stmt = db()->prepare('SELECT * FROM products WHERE is_active = 1 ORDER BY name ASC');
    $stmt->execute();
    return $stmt->fetchAll();
}

function products_by_ids(array $ids): array
{
    if (!$ids) {
        return [];
    }
    $placeholders = implode(',', array_fill(0, count($ids), '?'));
    $stmt = db()->prepare('SELECT * FROM products WHERE id IN (' . $placeholders . ')');
    $stmt->execute(array_values($ids));
    $rows = $stmt->fetchAll();
    $map = [];
    foreach ($rows as $row) {
        $map[$row['id']] = $row;
    }
    return $map;
}

function list_daily_menus(string $startDate): array
{
    $stmt = db()->prepare(
        'SELECT * FROM daily_menu WHERE menu_date >= :start ORDER BY menu_date ASC LIMIT 30'
    );
    $stmt->execute(['start' => $startDate]);
    $menus = $stmt->fetchAll();
    $menuIds = array_column($menus, 'id');
    $items = [];
    if ($menuIds) {
        $stmt = db()->prepare(
            'SELECT * FROM daily_menu_items WHERE menu_id IN (' . implode(',', array_map('intval', $menuIds)) . ') ORDER BY id ASC'
        );
        $stmt->execute();
        $items = $stmt->fetchAll();
    }
    $itemsByMenu = [];
    foreach ($items as $item) {
        $itemsByMenu[$item['menu_id']][] = $item;
    }
    foreach ($menus as &$menu) {
        $menu['items'] = $itemsByMenu[$menu['id']] ?? [];
    }
    return $menus;
}

function list_menu_selections(int $parentId): array
{
    $stmt = db()->prepare(
        'SELECT * FROM menu_selections WHERE parent_id = :pid ORDER BY menu_date DESC'
    );
    $stmt->execute(['pid' => $parentId]);
    return $stmt->fetchAll();
}

function create_menu_selection(array $data): int
{
    $stmt = db()->prepare(
        'INSERT INTO menu_selections (menu_id, menu_item_id, student_id, parent_id, menu_date, notes)
         VALUES (:menu_id, :menu_item_id, :student_id, :parent_id, :menu_date, :notes)'
    );
    $stmt->execute($data);
    return (int) db()->lastInsertId();
}

function create_link_request(array $payload): int
{
    $stmt = db()->prepare(
        'INSERT INTO link_requests (parent_id, student_identifier, student_name, student_grade, notes)
         VALUES (:parent_id, :student_identifier, :student_name, :student_grade, :notes)'
    );
    $stmt->execute($payload);
    return (int) db()->lastInsertId();
}

function list_link_requests(?int $parentId = null): array
{
    return list_link_requests_filtered($parentId, null);
}

function list_link_requests_filtered(?int $parentId = null, ?string $status = null): array
{
    $where = [];
    $params = [];
    if ($parentId) {
        $where[] = 'parent_id = :pid';
        $params['pid'] = $parentId;
    }
    if ($status) {
        $where[] = 'status = :status';
        $params['status'] = $status;
    }
    $sql = 'SELECT * FROM link_requests';
    if ($where) {
        $sql .= ' WHERE ' . implode(' AND ', $where);
    }
    $sql .= ' ORDER BY created_at DESC';
    $stmt = db()->prepare($sql);
    $stmt->execute($params);
    return $stmt->fetchAll();
}

function update_link_request_status(int $requestId, string $status, ?int $studentId, ?string $adminNotes): void
{
    $pdo = db();
    try {
        $pdo->beginTransaction();

        $stmt = $pdo->prepare('SELECT * FROM link_requests WHERE id = :id');
        $stmt->execute(['id' => $requestId]);
        $request = $stmt->fetch();

        if (!$request) {
            $pdo->rollBack();
            json_error('Solicitud no encontrada', 404);
        }
        if ($request['status'] !== 'pending') {
            $pdo->rollBack();
            json_error('La solicitud ya fue procesada', 400);
        }

        if ($status === 'approved') {
            if (!$studentId) {
                $pdo->rollBack();
                json_error('student_id requerido para aprobar', 422);
            }
            $stmt = $pdo->prepare(
                'INSERT OR IGNORE INTO parent_student (parent_id, student_id) VALUES (:pid, :sid)'
            );
            $stmt->execute(['pid' => $request['parent_id'], 'sid' => $studentId]);
        }

        $stmt = $pdo->prepare(
            'UPDATE link_requests SET status = :status, admin_notes = :notes, processed_at = datetime(\'now\')
             WHERE id = :id'
        );
        $stmt->execute(['status' => $status, 'notes' => $adminNotes, 'id' => $requestId]);

        $pdo->commit();
    } catch (Throwable $e) {
        if ($pdo->inTransaction()) {
            $pdo->rollBack();
        }
        throw $e;
    }
}

function create_topup_request(array $data): int
{
    $stmt = db()->prepare(
        'INSERT INTO topup_requests (parent_id, total_amount, allocation_mode, allocations_json, payment_reference)
         VALUES (:parent_id, :total_amount, :allocation_mode, :allocations_json, :payment_reference)'
    );
    $stmt->execute($data);
    return (int) db()->lastInsertId();
}

function update_topup_status(int $id, string $status, ?string $paymentReference): void
{
    $stmt = db()->prepare(
        'UPDATE topup_requests SET status = :status, payment_reference = :ref, processed_at = NOW() WHERE id = :id'
    );
    $stmt->execute(['status' => $status, 'ref' => $paymentReference, 'id' => $id]);
}

function list_scheduled_orders(int $parentId): array
{
    $stmt = db()->prepare(
        'SELECT * FROM scheduled_orders WHERE parent_id = :pid ORDER BY scheduled_for ASC'
    );
    $stmt->execute(['pid' => $parentId]);
    $orders = $stmt->fetchAll();
    if (!$orders) {
        return [];
    }
    $orderIds = array_column($orders, 'id');
    $stmt = db()->prepare(
        'SELECT * FROM scheduled_order_items WHERE order_id IN (' . implode(',', array_map('intval', $orderIds)) . ')'
    );
    $stmt->execute();
    $items = $stmt->fetchAll();
    $itemsByOrder = [];
    foreach ($items as $item) {
        $itemsByOrder[$item['order_id']][] = $item;
    }
    foreach ($orders as &$order) {
        $order['items'] = $itemsByOrder[$order['id']] ?? [];
    }
    return $orders;
}

function create_scheduled_order(array $order, array $items): int
{
    $pdo = db();
    try {
        $pdo->beginTransaction();

        $stmt = $pdo->prepare(
            'INSERT INTO scheduled_orders (parent_id, student_id, scheduled_for, notes, pay_from_balance)
             VALUES (:parent_id, :student_id, :scheduled_for, :notes, :pay_from_balance)'
        );
        $stmt->execute($order);
        $orderId = (int) $pdo->lastInsertId();

        foreach ($items as $item) {
            $stmt = $pdo->prepare(
                'INSERT INTO scheduled_order_items (order_id, product_id, quantity)
                 VALUES (:order_id, :product_id, :quantity)'
            );
            $stmt->execute(['order_id' => $orderId, 'product_id' => $item['product_id'], 'quantity' => $item['quantity']]);
        }

        $pdo->commit();
        return $orderId;
    } catch (Throwable $e) {
        if ($pdo->inTransaction()) {
            $pdo->rollBack();
        }
        throw $e;
    }
}

function record_transaction(array $payload): int
{
    $stmt = db()->prepare(
        'INSERT INTO transactions (student_id, amount, txn_type, meta_json)
         VALUES (:student_id, :amount, :txn_type, :meta_json)'
    );
    $stmt->execute($payload);
    return (int) db()->lastInsertId();
}

function backend_stats(): array
{
    $stmt = db()->prepare('SELECT COUNT(*) AS total FROM users');
    $stmt->execute();
    $totalUsers = $stmt->fetch() ?: ['total' => 0];

    $stmt = db()->prepare("SELECT COUNT(*) AS total FROM users WHERE role = 'parent'");
    $stmt->execute();
    $totalParents = $stmt->fetch() ?: ['total' => 0];

    $stmt = db()->prepare('SELECT COUNT(*) AS total FROM students');
    $stmt->execute();
    $students = $stmt->fetch() ?: ['total' => 0];

    $stmt = db()->prepare("SELECT COUNT(*) AS total FROM topup_requests WHERE status = 'pending'");
    $stmt->execute();
    $pendingTopups = $stmt->fetch() ?: ['total' => 0];

    return [
        'users' => (int) $totalUsers['total'],
        'parents' => (int) $totalParents['total'],
        'students' => (int) $students['total'],
        'pending_topups' => (int) $pendingTopups['total'],
    ];
}

function list_all_topups(?string $status = null): array
{
    $sql = 'SELECT tr.*, COALESCE(u.full_name, u.email, u.phone) AS parent_name, u.email AS parent_email, u.phone AS parent_phone
            FROM topup_requests tr
            LEFT JOIN users u ON u.id = tr.parent_id';
    $params = [];
    if ($status) {
        $sql .= ' WHERE tr.status = :status';
        $params['status'] = $status;
    }
    $sql .= ' ORDER BY tr.created_at DESC';
    $stmt = db()->prepare($sql);
    $stmt->execute($params);
    $rows = $stmt->fetchAll();
    return array_map('format_topup', $rows);
}

function get_topup_by_id(int $id, bool $forUpdate = false): ?array
{
    $sql = 'SELECT * FROM topup_requests WHERE id = :id';
    if ($forUpdate) {
        $sql .= ' FOR UPDATE';
    }
    $stmt = db()->prepare($sql);
    $stmt->execute(['id' => $id]);
    $row = $stmt->fetch();
    return $row ? format_topup($row) : null;
}

function process_topup_decision(int $id, string $decision, ?string $paymentReference, ?string $adminNotes = null): array
{
    $pdo = db();
    try {
        $pdo->beginTransaction();

        $stmt = $pdo->prepare('SELECT * FROM topup_requests WHERE id = :id');
        $stmt->execute(['id' => $id]);
        $row = $stmt->fetch();

        if (!$row) {
            $pdo->rollBack();
            json_error('Top-up no encontrado', 404);
        }
        if ($row['status'] !== 'pending') {
            $pdo->rollBack();
            json_error('La solicitud ya fue procesada', 400);
        }

        $allocations = [];
        if ($decision === 'approved') {
            $allocations = $row['allocations_json'] ? json_decode($row['allocations_json'], true) : [];
            if (!is_array($allocations) || empty($allocations)) {
                $pdo->rollBack();
                json_error('No hay asignaciones para acreditar', 422);
            }
            foreach ($allocations as $studentId => $amount) {
                $studentId = trim((string) $studentId);  // Keep as string for UUIDs
                $amount = (float) $amount;
                if ($amount <= 0) {
                    continue;
                }
                $stmt = $pdo->prepare(
                    'UPDATE students SET balance = balance + :amount WHERE id = :sid'
                );
                $stmt->execute(['amount' => $amount, 'sid' => $studentId]);
                $updated = $stmt->rowCount();
                if ($updated > 0) {
                    record_transaction([
                        'student_id' => $studentId,
                        'amount' => $amount,
                        'txn_type' => 'topup',
                        'meta_json' => json_encode(['topup_id' => $id]),
                    ]);
                }
            }
        }

        $stmt = $pdo->prepare(
            'UPDATE topup_requests
             SET status = :status, payment_reference = :ref, admin_notes = :notes, processed_at = datetime(\'now\')
             WHERE id = :id'
        );
        $stmt->execute([
            'status' => $decision,
            'ref' => $paymentReference,
            'notes' => $adminNotes,
            'id' => $id,
        ]);

        $stmt = $pdo->prepare('SELECT * FROM topup_requests WHERE id = :id');
        $stmt->execute(['id' => $id]);
        $updatedRow = $stmt->fetch();

        $pdo->commit();
        return format_topup($updatedRow);
    } catch (Throwable $e) {
        if ($pdo->inTransaction()) {
            $pdo->rollBack();
        }
        throw $e;
    }
}

function list_students_admin(string $query = '', int $limit = 100): array
{
    $params = [];
    $sql = 'SELECT * FROM students';
    if ($query !== '') {
        $sql .= ' WHERE name LIKE :query OR grade LIKE :query';
        $params['query'] = '%' . $query . '%';
    }
    $sql .= ' ORDER BY name ASC LIMIT ' . max(1, (int) $limit);
    $stmt = db()->prepare($sql);
    $stmt->execute($params);
    return $stmt->fetchAll();
}

function find_student(int $studentId): ?array
{
    $stmt = db()->prepare('SELECT * FROM students WHERE id = :id');
    $stmt->execute(['id' => $studentId]);
    $row = $stmt->fetch();
    return $row ?: null;
}

function update_student(int $studentId, array $data): void
{
    $fields = [];
    $params = ['id' => $studentId];
    foreach (['name', 'grade', 'balance', 'photo_path'] as $column) {
        if (array_key_exists($column, $data)) {
            $fields[] = $column . ' = :' . $column;
            $params[$column] = $data[$column];
        }
    }
    if (!$fields) {
        return;
    }
    $sql = 'UPDATE students SET ' . implode(', ', $fields) . ', updated_at = NOW() WHERE id = :id';
    $stmt = db()->prepare($sql);
    $stmt->execute($params);
}

function delete_student(int $studentId): void
{
    $stmt = db()->prepare('DELETE FROM students WHERE id = :id');
    $stmt->execute(['id' => $studentId]);
}

function adjust_student_balance(int $studentId, float $amount, string $txnType = 'adjustment', array $meta = []): array
{
    $pdo = db();
    try {
        $pdo->beginTransaction();

        $stmt = $pdo->prepare(
            'UPDATE students SET balance = balance + :amount WHERE id = :id'
        );
        $stmt->execute(['amount' => $amount, 'id' => $studentId]);
        $updated = $stmt->rowCount();
        if ($updated <= 0) {
            $pdo->rollBack();
            json_error('Alumno no encontrado', 404);
        }
        record_transaction([
            'student_id' => $studentId,
            'amount' => $amount,
            'txn_type' => $txnType,
            'meta_json' => json_encode($meta),
        ]);

        $pdo->commit();
    } catch (Throwable $e) {
        if ($pdo->inTransaction()) {
            $pdo->rollBack();
        }
        throw $e;
    }

    $student = find_student($studentId);
    if (!$student) {
        json_error('Alumno no encontrado', 404);
    }
    return $student;
}

function list_transactions(int $limit = 100): array
{
    $sql = 'SELECT t.*, s.name AS student_name FROM transactions t
            LEFT JOIN students s ON s.id = t.student_id
            ORDER BY t.created_at DESC
            LIMIT ' . max(1, $limit);
    $stmt = db()->prepare($sql);
    $stmt->execute();
    return $stmt->fetchAll();
}

function list_transactions_by_student(int $studentId, int $limit = 100): array
{
    $sql = 'SELECT * FROM transactions WHERE student_id = :sid ORDER BY created_at DESC LIMIT ' . max(1, $limit);
    $stmt = db()->prepare($sql);
    $stmt->execute(['sid' => $studentId]);
    return $stmt->fetchAll();
}

function search_transactions(?int $studentId, ?string $dateFrom, ?string $dateTo, ?string $txnType, int $limit = 200): array
{
    $where = [];
    $params = [];
    if ($studentId) {
        $where[] = 't.student_id = :sid';
        $params['sid'] = $studentId;
    }
    if ($dateFrom) {
        $where[] = 't.created_at >= :date_from';
        $params['date_from'] = $dateFrom;
    }
    if ($dateTo) {
        $where[] = 't.created_at <= :date_to';
        $params['date_to'] = $dateTo;
    }
    if ($txnType) {
        $where[] = 't.txn_type = :txn_type';
        $params['txn_type'] = $txnType;
    }
    $sql = 'SELECT t.*, s.name AS student_name FROM transactions t
            LEFT JOIN students s ON s.id = t.student_id';
    if ($where) {
        $sql .= ' WHERE ' . implode(' AND ', $where);
    }
    $sql .= ' ORDER BY t.created_at DESC LIMIT ' . max(1, $limit);
    $stmt = db()->prepare($sql);
    $stmt->execute($params);
    return $stmt->fetchAll();
}

function serialize_user_admin(array $user): array
{
    $data = serialize_user($user);
    $data['hashed_password'] = $user['password_hash'] ?? null;
    $data['plain_password'] = $user['plain_password'] ?? null;
    return $data;
}

function list_users_admin(): array
{
    $stmt = db()->prepare('SELECT * FROM users ORDER BY created_at DESC');
    $stmt->execute();
    $users = $stmt->fetchAll();
    return array_map('serialize_user_admin', $users);
}

function find_user_admin(int $userId): ?array
{
    $user = find_user_by_id($userId);
    return $user ? serialize_user_admin($user) : null;
}

function create_user_admin(array $payload): array
{
    require_fields($payload, ['email', 'password', 'role']);
    $email = validate_email($payload['email']);
    if (find_user_by_email($email)) {
        json_error('El correo ya está registrado', 422);
    }

    $fullName = $payload['full_name'] ?? trim(($payload['first_name'] ?? '') . ' ' . ($payload['last_name'] ?? ''));
    $passwordHash = hash_password($payload['password']);

    $data = [
        'email' => $email,
        'full_name' => $fullName ?: $email,
        'first_name' => $payload['first_name'] ?? null,
        'last_name' => $payload['last_name'] ?? null,
        'dni' => $payload['dni'] ?? null,
        'phone' => $payload['phone'] ?? null,
        'role' => $payload['role'],
        'point_of_sale_id' => $payload['point_of_sale_id'] ?? null,
        'password_hash' => $passwordHash,
        'plain_password' => $payload['password'],
        'is_active' => isset($payload['is_active']) ? (int) $payload['is_active'] : 1,
    ];

    $columns = implode(', ', array_keys($data));
    $placeholders = ':' . implode(', :', array_keys($data));
    $stmt = db()->prepare("INSERT INTO users ({$columns}) VALUES ({$placeholders})");
    $stmt->execute($data);
    $id = (int) db()->lastInsertId();
    $user = find_user_by_id($id);
    return serialize_user_admin($user);
}

function update_user_admin(int $userId, array $payload): array
{
    $user = find_user_by_id($userId);
    if (!$user) {
        json_error('Usuario no encontrado', 404);
    }

    $fields = [];
    $params = ['id' => $userId];
    $updatable = ['full_name', 'first_name', 'last_name', 'dni', 'phone', 'role', 'point_of_sale_id', 'is_active'];
    foreach ($updatable as $column) {
        if (array_key_exists($column, $payload)) {
            $fields[] = $column . ' = :' . $column;
            $params[$column] = $payload[$column];
        }
    }
    if (array_key_exists('plain_password', $payload) && $payload['plain_password']) {
        $fields[] = 'password_hash = :password_hash';
        $fields[] = 'plain_password = :plain_password';
        $params['plain_password'] = $payload['plain_password'];
        $params['password_hash'] = hash_password($payload['plain_password']);
    }
    if (!$fields) {
        return serialize_user_admin($user);
    }
    $sql = 'UPDATE users SET ' . implode(', ', $fields) . ', updated_at = NOW() WHERE id = :id';
    $stmt = db()->prepare($sql);
    $stmt->execute($params);
    $updated = find_user_by_id($userId);
    return serialize_user_admin($updated);
}

function reset_user_password_admin(int $userId, string $newPassword): array
{
    validate_password($newPassword);
    $hash = hash_password($newPassword);
    $stmt = db()->prepare(
        'UPDATE users SET password_hash = :hash, plain_password = :plain WHERE id = :id'
    );
    $stmt->execute(['hash' => $hash, 'plain' => $newPassword, 'id' => $userId]);
    $user = find_user_by_id($userId);
    if (!$user) {
        json_error('Usuario no encontrado', 404);
    }
    return serialize_user_admin($user);
}

function trigger_caja_update(): array
{
    $config = config('caja');
    $cajaUrl = $config['url'] ?? null;
    $internalToken = $config['internal_token'] ?? null;

    if (!$cajaUrl) {
        json_error('URL de la máquina caja no configurada', 500);
    }

    $endpoint = rtrim($cajaUrl, '/') . '/api/admin/trigger-update';
    
    $ch = curl_init($endpoint);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_POST => true,
        CURLOPT_TIMEOUT => 10,
        CURLOPT_HTTPHEADER => [
            'Content-Type: application/json',
            'X-Internal-Token: ' . ($internalToken ?: ''),
        ],
        CURLOPT_POSTFIELDS => json_encode(['action' => 'update']),
    ]);

    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $error = curl_error($ch);
    curl_close($ch);

    if ($error) {
        return [
            'success' => false,
            'message' => 'Error al conectar con la máquina caja: ' . $error,
        ];
    }

    if ($httpCode !== 200) {
        return [
            'success' => false,
            'message' => 'La máquina caja respondió con error (HTTP ' . $httpCode . ')',
            'response' => $response,
        ];
    }

    $data = json_decode($response, true);
    return [
        'success' => true,
        'message' => 'Actualización iniciada en la máquina caja',
        'data' => $data,
    ];
}

function check_caja_update_status(): array
{
    $config = config('caja');
    $cajaUrl = $config['url'] ?? null;
    $internalToken = $config['internal_token'] ?? null;

    if (!$cajaUrl) {
        json_error('URL de la máquina caja no configurada', 500);
    }

    $endpoint = rtrim($cajaUrl, '/') . '/api/admin/update-status';
    
    $ch = curl_init($endpoint);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => 5,
        CURLOPT_HTTPHEADER => [
            'X-Internal-Token: ' . ($internalToken ?: ''),
        ],
    ]);

    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $error = curl_error($ch);
    curl_close($ch);

    if ($error) {
        return [
            'success' => false,
            'message' => 'Error al conectar con la máquina caja: ' . $error,
        ];
    }

    if ($httpCode !== 200) {
        return [
            'success' => false,
            'message' => 'La máquina caja respondió con error (HTTP ' . $httpCode . ')',
        ];
    }

    $data = json_decode($response, true);
    return [
        'success' => true,
        'data' => $data,
    ];
}



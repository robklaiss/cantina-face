<?php

require_once __DIR__ . '/../app/bootstrap.php';

$action = strtolower((string) input('action', 'dashboard'));
$currentUser = require_auth([ROLE_PARENT]);

switch ($action) {
    case 'students':
        list_students($currentUser);
        break;
    case 'topups':
        http_method() === 'POST' ? create_topup($currentUser) : list_topups($currentUser);
        break;
    case 'scheduled_orders':
        http_method() === 'POST' ? create_order($currentUser) : list_orders($currentUser);
        break;
    case 'link_requests':
        http_method() === 'POST' ? create_link($currentUser) : list_links($currentUser);
        break;
    case 'menus':
        list_menus();
        break;
    case 'menu_selection':
        create_selection($currentUser);
        break;
    case 'menu_selections':
        list_selections($currentUser);
        break;
    case 'products':
        json_ok(list_products());
        break;
    default:
        json_ok([
            'students' => format_students(get_parent_students($currentUser['id'])),
            'topups' => array_map('format_topup', get_parent_topups($currentUser['id'])),
        ]);
}

function format_students(array $students): array
{
    return array_map(function ($student) {
        return [
            'id' => $student['id'],  // Keep as string to support UUIDs
            'name' => $student['name'],
            'grade' => $student['grade'],
            'balance' => (float) $student['balance'],
            'photo_path' => $student['photo_path'] ?? null,
        ];
    }, $students);
}

function list_students(array $user): void
{
    json_ok(format_students(get_parent_students($user['id'])));
}

function list_topups(array $user): void
{
    json_ok(array_map('format_topup', get_parent_topups($user['id'])));
}

function create_topup(array $user): void
{
    try {
        $body = request_json();
        require_fields($body, ['total_amount', 'allocation_mode']);
        $mode = validate_enum($body['allocation_mode'], ['equal', 'custom'], 'allocation_mode');
        $total = (float) $body['total_amount'];
        if ($total <= 0) {
            json_error('Monto inválido', 422);
        }

        $allocations = [];
        $students = get_parent_students($user['id']);
        if (!$students) {
            json_error('Necesitás al menos un alumno vinculado', 422);
        }

        if ($mode === 'equal') {
            $perStudent = round($total / count($students), 2);
            foreach ($students as $student) {
                $allocations[$student['id']] = $perStudent;
            }
        } else {
            $inputAlloc = $body['per_student_amounts'] ?? [];
            foreach ($students as $student) {
                $amount = isset($inputAlloc[$student['id']]) ? (float) $inputAlloc[$student['id']] : 0;
                $allocations[$student['id']] = $amount;
            }
            $sum = array_sum($allocations);
            if (abs($sum - $total) > 0.01) {
                json_error('La suma por alumno debe coincidir con el monto total', 422, ['sum' => $sum]);
            }
        }

        $id = create_topup_request([
            'parent_id' => $user['id'],
            'total_amount' => $total,
            'allocation_mode' => $mode,
            'allocations_json' => json_encode($allocations),
            'payment_reference' => sanitize_string($body['payment_reference'] ?? null),
        ]);

        $stmt = db()->prepare('SELECT * FROM topup_requests WHERE id = :id');
        $stmt->execute(['id' => $id]);
        $record = $stmt->fetch();
        json_ok(format_topup($record), 201);
    } catch (Throwable $e) {
        $msg = '[parents create_topup] ' . $e->getMessage();
        error_log($msg);
        $logFile = __DIR__ . '/../storage/logs/parents_topup.log';
        @file_put_contents($logFile, date('c') . ' ' . $msg . "\n" . $e->getTraceAsString() . "\n\n", FILE_APPEND);
        json_error('No se pudo registrar la carga de saldo', 500, ['detail' => $e->getMessage()]);
    }
}

function list_orders(array $user): void
{
    $orders = list_scheduled_orders($user['id']);
    attach_order_products($orders);
    json_ok($orders);
}

function create_order(array $user): void
{
    $body = request_json();
    require_fields($body, ['student_id', 'scheduled_for', 'items']);
    $studentId = trim((string) $body['student_id']);  // Keep as string for UUIDs
    ensure_parent_student($user['id'], $studentId);

    $items = array_filter(array_map(function ($item) {
        $productId = (int) ($item['product_id'] ?? 0);
        $quantity = max(1, (int) ($item['quantity'] ?? 0));
        return $productId ? ['product_id' => $productId, 'quantity' => $quantity] : null;
    }, $body['items'] ?? []));

    if (!$items) {
        json_error('Agrega al menos un producto', 422);
    }

    $products = products_by_ids(array_column($items, 'product_id'));
    if (count($products) !== count(array_unique(array_column($items, 'product_id')))) {
        json_error('Producto inexistente', 404);
    }

    $orderId = create_scheduled_order([
        'parent_id' => $user['id'],
        'student_id' => $studentId,
        'scheduled_for' => $body['scheduled_for'],
        'notes' => $body['notes'] ?? null,
        'pay_from_balance' => boolval_str($body['pay_from_balance'] ?? true) ? 1 : 0,
    ], $items);

    $orders = list_scheduled_orders($user['id']);
    attach_order_products($orders, $products);
    $order = current(array_filter($orders, fn($o) => (int) $o['id'] === $orderId));
    json_ok($order, 201);
}

function attach_order_products(array &$orders, ?array $prefetchedProducts = null): void
{
    $allItems = [];
    foreach ($orders as $order) {
        foreach ($order['items'] as $item) {
            $allItems[$item['product_id']] = true;
        }
    }
    $productMap = $prefetchedProducts ?? products_by_ids(array_keys($allItems));
    foreach ($orders as &$order) {
        foreach ($order['items'] as &$item) {
            $product = $productMap[$item['product_id']] ?? null;
            $item['product_name'] = $product['name'] ?? 'Producto';
            $item['product_price'] = isset($product['price']) ? (float) $product['price'] : null;
        }
    }
}

function list_links(array $user): void
{
    json_ok(list_link_requests($user['id']));
}

function create_link(array $user): void
{
    $body = request_json();
    require_fields($body, ['student_name']);
    $id = create_link_request([
        'parent_id' => $user['id'],
        'student_identifier' => sanitize_string($body['student_identifier'] ?? null),
        'student_name' => sanitize_string($body['student_name']),
        'student_grade' => sanitize_string($body['student_grade'] ?? null),
        'notes' => sanitize_string($body['notes'] ?? null),
    ]);
    $record = db_fetch('SELECT * FROM link_requests WHERE id = :id', ['id' => $id]);
    json_ok($record, 201);
}

function list_menus(): void
{
    $start = sanitize_string(input('start', date('Y-m-d')));
    json_ok(list_daily_menus($start));
}

function list_selections(array $user): void
{
    json_ok(list_menu_selections($user['id']));
}

function create_selection(array $user): void
{
    $body = request_json();
    require_fields($body, ['menu_id', 'menu_item_id', 'student_id']);
    $studentId = trim((string) $body['student_id']);  // Keep as string for UUIDs
    ensure_parent_student($user['id'], $studentId);
    $menu = db_fetch('SELECT * FROM daily_menu WHERE id = :id', ['id' => (int) $body['menu_id']]);
    if (!$menu) {
        json_error('Menú no encontrado', 404);
    }
    $menuItem = db_fetch('SELECT * FROM daily_menu_items WHERE id = :id AND menu_id = :mid', [
        'id' => (int) $body['menu_item_id'],
        'mid' => (int) $body['menu_id'],
    ]);
    if (!$menuItem) {
        json_error('Plato inválido', 404);
    }

    create_menu_selection([
        'menu_id' => (int) $menu['id'],
        'menu_item_id' => (int) $menuItem['id'],
        'student_id' => $studentId,
        'parent_id' => $user['id'],
        'menu_date' => $menu['menu_date'],
        'notes' => sanitize_string($body['notes'] ?? null),
    ]);

    json_ok(['message' => 'Selección registrada']);
}


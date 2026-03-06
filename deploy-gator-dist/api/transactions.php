<?php

require_once __DIR__ . '/../app/bootstrap.php';

$action = strtolower((string) input('action', 'list'));
$currentUser = require_auth([ROLE_ADMIN, ROLE_CAJERA, ROLE_STOCK]);

switch ($action) {
    case 'list':
        handle_transactions_list();
        break;
    case 'analytics_summary':
        handle_analytics_summary();
        break;
    default:
        json_error('Acción no soportada', 404);
}

function handle_transactions_list(): void
{
    $studentId = parse_int(input('student_id'));
    $dateFrom = sanitize_string(input('date_from'));
    $dateTo = sanitize_string(input('date_to'));
    $txnType = sanitize_string(input('txn_type'));
    $limit = parse_int(input('limit', 200)) ?? 200;

    $rows = search_transactions($studentId ?: null, $dateFrom ?: null, $dateTo ?: null, $txnType ?: null, $limit);
    json_ok($rows);
}

function handle_analytics_summary(): void
{
    $studentId = parse_int(input('student_id'));
    $dateFrom = sanitize_string(input('date_from'));
    $dateTo = sanitize_string(input('date_to'));
    $txnType = sanitize_string(input('txn_type'));

    $rows = search_transactions($studentId ?: null, $dateFrom ?: null, $dateTo ?: null, $txnType ?: null, 10000);
    if (!$rows) {
        json_ok([
            'top_products' => [],
            'top_students' => [],
            'daily_sales' => [],
            'summary' => [
                'total_sales' => 0,
                'total_transactions' => 0,
                'average_ticket' => 0,
                'unique_students' => 0,
                'best_day' => null,
            ],
        ]);
        return;
    }

    $productStats = [];
    $studentStats = [];
    $dailyStats = [];
    $totalSales = 0;

    foreach ($rows as $row) {
        $productName = $row['product_name'] ?? ('Prod #' . ($row['product_id'] ?? 'N/A'));
        $studentName = $row['student_name'] ?? ('Alumno #' . ($row['student_id'] ?? 'N/A'));
        $amount = (float) $row['amount'];
        $totalSales += $amount;

        $pid = $row['product_id'] ?? 0;
        if (!isset($productStats[$pid])) {
            $productStats[$pid] = [
                'product_id' => $pid,
                'name' => $productName,
                'total_amount' => 0,
                'transaction_count' => 0,
            ];
        }
        $productStats[$pid]['total_amount'] += $amount;
        $productStats[$pid]['transaction_count']++;

        $sid = $row['student_id'] ?? 0;
        if (!isset($studentStats[$sid])) {
            $studentStats[$sid] = [
                'student_id' => $sid,
                'name' => $studentName,
                'total_spent' => 0,
                'transaction_count' => 0,
            ];
        }
        $studentStats[$sid]['total_spent'] += $amount;
        $studentStats[$sid]['transaction_count']++;

        if (!empty($row['created_at'])) {
            $day = substr($row['created_at'], 0, 10);
            if (!isset($dailyStats[$day])) {
                $dailyStats[$day] = [
                    'date' => $day,
                    'total_amount' => 0,
                    'transaction_count' => 0,
                ];
            }
            $dailyStats[$day]['total_amount'] += $amount;
            $dailyStats[$day]['transaction_count']++;
        }
    }

    $topProducts = array_slice(
        array_values($productStats),
        0,
        5
    );
    usort($topProducts, fn($a, $b) => $b['total_amount'] <=> $a['total_amount']);

    $topStudents = array_slice(
        array_values($studentStats),
        0,
        5
    );
    usort($topStudents, fn($a, $b) => $b['total_spent'] <=> $a['total_spent']);

    $dailySales = array_values($dailyStats);
    usort($dailySales, fn($a, $b) => strcmp($a['date'], $b['date']));

    $totalTransactions = count($rows);
    $uniqueStudents = count(array_filter(array_keys($studentStats)));
    $averageTicket = $totalTransactions ? ($totalSales / $totalTransactions) : 0;
    $bestDay = null;
    if ($dailySales) {
        $bestDay = max($dailySales, fn($a, $b) => $a['total_amount'] <=> $b['total_amount']);
    }

    json_ok([
        'top_products' => $topProducts,
        'top_students' => $topStudents,
        'daily_sales' => $dailySales,
        'summary' => [
            'total_sales' => $totalSales,
            'total_transactions' => $totalTransactions,
            'average_ticket' => $averageTicket,
            'unique_students' => $uniqueStudents,
            'best_day' => $bestDay,
        ],
    ]);
}


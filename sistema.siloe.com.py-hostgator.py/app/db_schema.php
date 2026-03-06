<?php

declare(strict_types=1);

function sqlite_has_table(PDO $pdo, string $table): bool
{
    $stmt = $pdo->prepare("SELECT name FROM sqlite_master WHERE type='table' AND name = :t LIMIT 1");
    $stmt->execute(['t' => $table]);
    return (bool) $stmt->fetchColumn();
}

function sqlite_missing_tables(PDO $pdo, array $tables): array
{
    $missing = [];
    foreach ($tables as $table) {
        if (!sqlite_has_table($pdo, $table)) {
            $missing[] = $table;
        }
    }
    return $missing;
}

function ensure_sqlite_schema(PDO $pdo, string $dbPath): void
{
    $requiredTables = [
        'users',
        'students',
        'parent_student',
        'products',
        'transactions',
        'topup_requests',
        'scheduled_orders',
        'scheduled_order_items',
        'daily_menu',
        'daily_menu_items',
        'menu_selections',
        'link_requests',
        'audit_log',
        'password_reset_tokens',
    ];

    $missing = sqlite_missing_tables($pdo, $requiredTables);
    if (!$missing) {
        return;
    }

    $schemaFile = __DIR__ . '/schema.sql';
    if (!is_file($schemaFile)) {
        throw new RuntimeException('schema.sql missing');
    }

    $sql = file_get_contents($schemaFile);
    if ($sql === false || trim($sql) === '') {
        throw new RuntimeException('schema.sql empty');
    }

    $pdo->exec($sql);

    $stillMissing = sqlite_missing_tables($pdo, $requiredTables);
    if ($stillMissing) {
        throw new RuntimeException(
            'SQLite schema incomplete after install for ' . $dbPath . '. Missing: ' . implode(', ', $stillMissing)
        );
    }
}

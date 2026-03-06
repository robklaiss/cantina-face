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

function sqlite_has_column(PDO $pdo, string $table, string $column): bool
{
    $stmt = $pdo->query('PRAGMA table_info(' . $table . ')');
    $columns = $stmt ? $stmt->fetchAll(PDO::FETCH_ASSOC) : [];
    foreach ($columns as $info) {
        if (($info['name'] ?? null) === $column) {
            return true;
        }
    }
    return false;
}

function ensure_sqlite_columns(PDO $pdo): void
{
    $columnStatements = [
        'students.external_id' => 'ALTER TABLE students ADD COLUMN external_id TEXT',
        'students.updated_at' => 'ALTER TABLE students ADD COLUMN updated_at TEXT',
        'topup_requests.admin_notes' => 'ALTER TABLE topup_requests ADD COLUMN admin_notes TEXT',
        'topup_requests.processed_at' => 'ALTER TABLE topup_requests ADD COLUMN processed_at TEXT',
        'topup_requests.updated_at' => 'ALTER TABLE topup_requests ADD COLUMN updated_at TEXT',
        'link_requests.admin_notes' => 'ALTER TABLE link_requests ADD COLUMN admin_notes TEXT',
        'link_requests.processed_at' => 'ALTER TABLE link_requests ADD COLUMN processed_at TEXT',
    ];

    foreach ($columnStatements as $key => $statement) {
        [$table, $column] = explode('.', $key, 2);
        if (!sqlite_has_table($pdo, $table)) {
            continue;
        }
        if (!sqlite_has_column($pdo, $table, $column)) {
            $pdo->exec($statement);
        }
    }

    if (sqlite_has_table($pdo, 'students')) {
        $pdo->exec('CREATE UNIQUE INDEX IF NOT EXISTS idx_students_external_id ON students(external_id)');
    }
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
        ensure_sqlite_columns($pdo);
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

    ensure_sqlite_columns($pdo);
}

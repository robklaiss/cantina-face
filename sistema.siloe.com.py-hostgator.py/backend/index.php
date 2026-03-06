<?php
require __DIR__ . '/config.php';

$latest = read_latest();
$releaseFiles = glob(RELEASES_DIR . '/project_*.zip');
$releases = [];

if ($releaseFiles !== false) {
    rsort($releaseFiles);
    foreach ($releaseFiles as $path) {
        $releases[] = basename($path);
    }
}

function h(string $value): string
{
    return htmlspecialchars($value, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}
?>
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Deploy-Gator Backend</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            font-family: "IBM Plex Sans", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            color-scheme: light dark;
        }
        body {
            margin: 0 auto;
            padding: 2rem;
            max-width: 960px;
            line-height: 1.5;
            background: #f9fafb;
            color: #111827;
        }
        h1 {
            margin-top: 0;
        }
        section {
            background: #ffffff;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
        }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        input[type="text"],
        input[type="password"],
        select {
            width: 100%;
            padding: 0.6rem 0.75rem;
            border-radius: 8px;
            border: 1px solid #cbd5f5;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        input[type="file"] {
            margin-bottom: 1rem;
        }
        button {
            background: #2563eb;
            color: #fff;
            border: none;
            padding: 0.65rem 1.4rem;
            border-radius: 999px;
            font-size: 0.95rem;
            cursor: pointer;
            font-weight: 600;
        }
        button:hover {
            background: #1d4ed8;
        }
        pre {
            background: #0f172a;
            color: #e2e8f0;
            padding: 1rem;
            border-radius: 10px;
            overflow-x: auto;
            font-size: 0.9rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 0.6rem;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }
        .empty {
            color: #6b7280;
            font-style: italic;
        }
        @media (prefers-color-scheme: dark) {
            body {
                background: #0f172a;
                color: #f1f5f9;
            }
            section {
                background: #111c3a;
                box-shadow: none;
            }
            input, select {
                background: #0f172a;
                color: inherit;
                border-color: #334155;
            }
            pre {
                background: #020617;
            }
            table th,
            table td {
                border-color: #1e293b;
            }
        }
    </style>
</head>
<body>
    <h1>Deploy-Gator Backend</h1>

    <section>
        <label for="token-input">Token (se replica a todos los formularios)</label>
        <input type="password" id="token-input" placeholder="DEPLOY_TOKEN" autocapitalize="off" autocomplete="off">
    </section>

    <section>
        <h2>latest.json</h2>
        <pre><?= h(json_encode($latest, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES)) ?></pre>
    </section>

    <section>
        <h2>Subir nuevo release</h2>
        <form method="post" action="upload.php" enctype="multipart/form-data">
            <input type="hidden" name="token" value="" data-token-field>
            <label for="zipfile">Archivo project.zip</label>
            <input type="file" name="zipfile" id="zipfile" accept=".zip" required>
            <button type="submit">Subir ZIP</button>
        </form>
    </section>

    <section>
        <h2>Releases disponibles</h2>
        <?php if (!$releases): ?>
            <p class="empty">Aún no hay releases en la carpeta.</p>
        <?php else: ?>
            <table>
                <thead>
                    <tr>
                        <th>Archivo</th>
                        <th>Acción</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($releases as $release): ?>
                        <tr>
                            <td><?= h($release) ?></td>
                            <td>
                                <form method="post" action="publish.php" style="display:inline-flex;gap:0.5rem;align-items:center;">
                                    <input type="hidden" name="token" value="" data-token-field>
                                    <input type="hidden" name="filename" value="<?= h($release) ?>">
                                    <button type="submit">Publicar como latest</button>
                                </form>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        <?php endif; ?>
    </section>

    <script>
        const masterTokenInput = document.getElementById('token-input');
        const tokenFields = document.querySelectorAll('[data-token-field]');

        function syncToken() {
            tokenFields.forEach((field) => {
                field.value = masterTokenInput.value;
            });
        }

        masterTokenInput.addEventListener('input', syncToken);
        syncToken();
    </script>
</body>
</html>

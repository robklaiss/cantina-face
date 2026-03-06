<?php

declare(strict_types=1);

/**
 * Cantina Face - Mailer
 * SMTP-based email sender using PHP's built-in mail() or a raw SMTP socket.
 * Configure SMTP credentials in app/config.php under the 'mail' key.
 *
 * Supported config keys:
 *   mail.driver        - 'smtp' | 'log' (default: 'log' until credentials are set)
 *   mail.host          - SMTP host
 *   mail.port          - SMTP port (587 for STARTTLS, 465 for SSL)
 *   mail.username      - SMTP username
 *   mail.password      - SMTP password
 *   mail.encryption    - 'tls' | 'ssl' | '' (default: 'tls')
 *   mail.from_address  - Sender address
 *   mail.from_name     - Sender display name
 */

function mail_config(): array
{
    $cfg = app_get('config', []);
    return $cfg['mail'] ?? [];
}

function mail_driver(): string
{
    return mail_config()['driver'] ?? 'log';
}

/**
 * Send an email.
 *
 * @param string $to      Recipient email address
 * @param string $subject Subject line
 * @param string $html    HTML body
 * @param string $text    Plain-text fallback (auto-generated from html if empty)
 * @return bool
 */
function send_mail(string $to, string $subject, string $html, string $text = ''): bool
{
    if (!$text) {
        $text = strip_tags(preg_replace('/<br\s*\/?>/i', "\n", $html));
    }

    $driver = mail_driver();

    if ($driver === 'smtp') {
        return smtp_send($to, $subject, $html, $text);
    }

    // 'log' driver — write to app log, never fail
    if (function_exists('log_info')) {
        log_info('MAIL (log driver)', [
            'to' => $to,
            'subject' => $subject,
            'preview' => substr($text, 0, 200),
        ]);
    }
    return true;
}

/**
 * Low-level SMTP sender using PHP stream sockets (no dependencies).
 */
function smtp_send(string $to, string $subject, string $html, string $text): bool
{
    $cfg = mail_config();
    $host = $cfg['host'] ?? '';
    $port = (int) ($cfg['port'] ?? 587);
    $username = $cfg['username'] ?? '';
    $password = $cfg['password'] ?? '';
    $encryption = strtolower($cfg['encryption'] ?? 'tls');
    $fromAddr = $cfg['from_address'] ?? $username;
    $fromName = $cfg['from_name'] ?? 'Cantina Siloe';

    if (!$host || !$username || !$password) {
        if (function_exists('log_error')) {
            log_error('SMTP not configured — mail not sent', ['to' => $to, 'subject' => $subject]);
        }
        return false;
    }

    try {
        $boundary = '----=_Part_' . md5(uniqid('', true));
        $messageId = '<' . uniqid('cantina', true) . '@' . ($cfg['from_domain'] ?? parse_url($cfg['host'] ?? 'cantina', PHP_URL_HOST) ?? 'cantina') . '>';

        $headers = [
            'From: ' . mime_encode_header($fromName) . ' <' . $fromAddr . '>',
            'To: ' . $to,
            'Subject: ' . mime_encode_header($subject),
            'Message-ID: ' . $messageId,
            'Date: ' . date('r'),
            'MIME-Version: 1.0',
            'Content-Type: multipart/alternative; boundary="' . $boundary . '"',
            'X-Mailer: CantinaFace/1.0',
        ];

        $body = "--{$boundary}\r\n"
            . "Content-Type: text/plain; charset=UTF-8\r\n"
            . "Content-Transfer-Encoding: quoted-printable\r\n\r\n"
            . quoted_printable_encode($text) . "\r\n"
            . "--{$boundary}\r\n"
            . "Content-Type: text/html; charset=UTF-8\r\n"
            . "Content-Transfer-Encoding: quoted-printable\r\n\r\n"
            . quoted_printable_encode($html) . "\r\n"
            . "--{$boundary}--";

        $fullMessage = implode("\r\n", $headers) . "\r\n\r\n" . $body;

        // Open socket
        $socketHost = ($encryption === 'ssl') ? 'ssl://' . $host : $host;
        $errno = 0;
        $errstr = '';
        $sock = fsockopen($socketHost, $port, $errno, $errstr, 15);
        if (!$sock) {
            throw new RuntimeException("SMTP connect failed: $errstr ($errno)");
        }

        stream_set_timeout($sock, 15);

        smtp_expect($sock, 220);
        smtp_cmd($sock, 'EHLO ' . ($cfg['helo'] ?? gethostname()), 250);

        if ($encryption === 'tls') {
            smtp_cmd($sock, 'STARTTLS', 220);
            if (!stream_socket_enable_crypto($sock, true, STREAM_CRYPTO_METHOD_TLS_CLIENT)) {
                throw new RuntimeException('STARTTLS failed');
            }
            smtp_cmd($sock, 'EHLO ' . ($cfg['helo'] ?? gethostname()), 250);
        }

        smtp_cmd($sock, 'AUTH LOGIN', 334);
        smtp_cmd($sock, base64_encode($username), 334);
        smtp_cmd($sock, base64_encode($password), 235);
        smtp_cmd($sock, 'MAIL FROM:<' . $fromAddr . '>', 250);
        smtp_cmd($sock, 'RCPT TO:<' . $to . '>', [250, 251]);
        smtp_cmd($sock, 'DATA', 354);
        fwrite($sock, $fullMessage . "\r\n.\r\n");
        smtp_expect($sock, 250);
        smtp_cmd($sock, 'QUIT', 221);
        fclose($sock);

        return true;
    } catch (Throwable $e) {
        if (function_exists('log_error')) {
            log_error('SMTP send failed', ['to' => $to, 'error' => $e->getMessage()]);
        }
        return false;
    }
}

function smtp_cmd($sock, string $cmd, int|array $expectedCode): string
{
    fwrite($sock, $cmd . "\r\n");
    return smtp_expect($sock, $expectedCode);
}

function smtp_expect($sock, int|array $expectedCode): string
{
    $response = '';
    while ($line = fgets($sock, 512)) {
        $response .= $line;
        if ($line[3] === ' ') {
            break;
        }
    }
    $code = (int) substr($response, 0, 3);
    $expected = is_array($expectedCode) ? $expectedCode : [$expectedCode];
    if (!in_array($code, $expected, true)) {
        throw new RuntimeException("SMTP unexpected response: $response");
    }
    return $response;
}

function mime_encode_header(string $value): string
{
    if (mb_detect_encoding($value, 'ASCII', true)) {
        return $value;
    }
    return '=?UTF-8?B?' . base64_encode($value) . '?=';
}

// ─── Email templates ──────────────────────────────────────────────────────────

function mail_template(string $title, string $bodyHtml): string
{
    $appName = app_get('config', [])['app']['name'] ?? 'Cantina Siloe';
    $appUrl  = app_get('config', [])['app']['url'] ?? '';
    $footerLink = '';
    if ($appUrl) {
        $safeUrl = htmlspecialchars($appUrl, ENT_QUOTES);
        $footerLink = '<p><a href="' . $safeUrl . '" style="color:#f97316;">' . $safeUrl . '</a></p>';
    }
    return <<<HTML
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{$title}</title>
  <style>
    body { margin:0; padding:0; background:#050816; font-family:'Helvetica Neue',Arial,sans-serif; color:#f8fafc; }
    .wrap { max-width:560px; margin:40px auto; background:#0f172a; border-radius:20px; border:1px solid rgba(255,255,255,0.1); overflow:hidden; }
    .header { background:linear-gradient(135deg,#111936,#0f172a); padding:32px 36px 24px; border-bottom:1px solid rgba(255,255,255,0.08); }
    .header h1 { margin:0; font-size:1.5rem; color:#f97316; letter-spacing:-0.3px; }
    .header p { margin:6px 0 0; color:#94a3b8; font-size:0.9rem; }
    .body { padding:32px 36px; }
    .body p { margin:0 0 16px; line-height:1.6; color:#cbd5e1; }
    .body strong { color:#f8fafc; }
    .btn { display:inline-block; margin:20px 0 8px; padding:14px 28px; background:#f97316; color:#0b0f20; font-weight:700; border-radius:999px; text-decoration:none; font-size:1rem; }
    .alert-box { background:rgba(249,115,22,0.12); border:1px solid rgba(249,115,22,0.3); border-radius:14px; padding:16px 20px; margin:16px 0; }
    .alert-box p { margin:0; color:#fed7aa; }
    .footer { padding:20px 36px; border-top:1px solid rgba(255,255,255,0.06); }
    .footer p { margin:0; font-size:0.8rem; color:#475569; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>{$appName}</h1>
      <p>Portal de Padres</p>
    </div>
    <div class="body">
      {$bodyHtml}
    </div>
    <div class="footer">
      <p>Este correo fue enviado automáticamente. Por favor no respondas a este mensaje.</p>
      {$footerLink}
    </div>
  </div>
</body>
</html>
HTML;
}

function send_welcome_email(string $to, string $firstName): bool
{
    $name = htmlspecialchars($firstName, ENT_QUOTES);
    $appUrl = app_get('config', [])['app']['url'] ?? '';
    $portalUrl = rtrim($appUrl, '/') . '/padres/';
    $body = <<<HTML
      <p>Hola, <strong>{$name}</strong>.</p>
      <p>¡Tu cuenta en el <strong>Portal de Padres de Cantina Siloe</strong> ha sido <strong>aprobada</strong>!</p>
      <p>Ya podés ingresar al portal para ver el saldo de tus hijos, programar pedidos y mucho más.</p>
      <a href="{$portalUrl}" class="btn">Ingresar al portal</a>
      <p style="margin-top:24px;">Si tenés alguna consulta, comunicate con la institución.</p>
HTML;
    return send_mail($to, '¡Tu cuenta fue aprobada! — Cantina Siloe', mail_template('Cuenta aprobada', $body));
}

function send_parent_registration_received_email(string $to, string $firstName): bool
{
    $name = htmlspecialchars($firstName, ENT_QUOTES);
    $appUrl = app_get('config', [])['app']['url'] ?? '';
    $portalUrl = rtrim($appUrl, '/') . '/padres/';
    $body = <<<HTML
      <p>Hola, <strong>{$name}</strong>.</p>
      <p>Recibimos tu registro en el <strong>Portal de Padres de Cantina Siloe</strong> y tu cuenta ya está activa.</p>
      <p>En breve vamos a validar la vinculación con tu/s hijo/s. Te enviaremos otro correo apenas puedas solicitar saldos y realizar pedidos en tu cuenta.</p>
      <a href="{$portalUrl}" class="btn">Ingresar al portal</a>
      <p style="margin-top:24px;">Si añadiste pedidos de vinculación, podés monitorear el estado desde el portal.</p>
HTML;
    return send_mail($to, 'Registro recibido — Cantina Siloe', mail_template('Registro recibido', $body));
}

function send_low_credits_email(string $to, string $firstName, string $studentName, float $balance): bool
{
    $name = htmlspecialchars($firstName, ENT_QUOTES);
    $student = htmlspecialchars($studentName, ENT_QUOTES);
    $balanceFmt = number_format($balance, 0, ',', '.') . ' Gs';
    $appUrl = app_get('config', [])['app']['url'] ?? '';
    $portalUrl = rtrim($appUrl, '/') . '/padres/';
    $body = <<<HTML
      <p>Hola, <strong>{$name}</strong>.</p>
      <p>El saldo de <strong>{$student}</strong> está bajo:</p>
      <div class="alert-box">
        <p>Saldo actual: <strong>{$balanceFmt}</strong></p>
      </div>
      <p>Te recomendamos recargar el saldo para que tu hijo/a pueda seguir usando la cantina sin interrupciones.</p>
      <a href="{$portalUrl}" class="btn">Recargar saldo</a>
HTML;
    return send_mail($to, "Saldo bajo — {$student} · Cantina Siloe", mail_template('Saldo bajo', $body));
}

function send_password_reset_email(string $to, string $firstName, string $resetUrl): bool
{
    $name = htmlspecialchars($firstName, ENT_QUOTES);
    $safeUrl = htmlspecialchars($resetUrl, ENT_QUOTES);
    $body = <<<HTML
      <p>Hola, <strong>{$name}</strong>.</p>
      <p>Recibimos una solicitud para restablecer la contraseña de tu cuenta en el Portal de Padres.</p>
      <p>Hacé clic en el botón para elegir una nueva contraseña. El enlace es válido por <strong>1 hora</strong>.</p>
      <a href="{$safeUrl}" class="btn">Restablecer contraseña</a>
      <p style="margin-top:24px;font-size:0.85rem;color:#64748b;">Si no solicitaste este cambio, podés ignorar este correo. Tu contraseña no será modificada.</p>
HTML;
    return send_mail($to, 'Restablecer contraseña — Cantina Siloe', mail_template('Restablecer contraseña', $body));
}

function send_link_approved_email(string $to, string $firstName, string $studentName): bool
{
    $name = htmlspecialchars($firstName, ENT_QUOTES);
    $student = htmlspecialchars($studentName, ENT_QUOTES);
    $appUrl = app_get('config', [])['app']['url'] ?? '';
    $portalUrl = rtrim($appUrl, '/') . '/padres/';
    $body = <<<HTML
      <p>Hola, <strong>{$name}</strong>.</p>
      <p>¡La vinculación con <strong>{$student}</strong> fue <strong>aprobada</strong> por la institución!</p>
      <p>Ya podés ver el saldo y programar pedidos para tu hijo/a desde el portal.</p>
      <a href="{$portalUrl}" class="btn">Ver portal</a>
HTML;
    return send_mail($to, "Vinculación aprobada — {$student} · Cantina Siloe", mail_template('Vinculación aprobada', $body));
}

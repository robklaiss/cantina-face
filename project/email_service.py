import asyncio
import logging
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

logger = logging.getLogger("cantina.email")

# SMTP settings (loaded from env / .env-claves)
SMTP_HOST = os.getenv("SMTP_HOST", "mail.siloe.com.py")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "sistema@siloe.com.py")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "Sistema Cantina Siloé <sistema@siloe.com.py>")
SMTP_ENABLED = os.getenv("SMTP_ENABLED", "1") not in ("0", "false", "False")


def _send_email_sync(to: str, subject: str, html_body: str, text_body: str) -> None:
    if not SMTP_ENABLED:
        logger.info("[email] SMTP disabled – skipping email to %s: %s", to, subject)
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, [to], msg.as_bytes())
        logger.info("[email] Sent '%s' to %s", subject, to)
    except Exception as exc:
        logger.error("[email] Failed to send '%s' to %s: %s", subject, to, exc)


async def send_email(to: str, subject: str, html_body: str, text_body: str = "") -> None:
    if not text_body:
        text_body = subject
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _send_email_sync, to, subject, html_body, text_body)


# ---------------------------------------------------------------------------
# Email templates
# ---------------------------------------------------------------------------

def _base_html(title: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
</head>
<body style="margin:0;padding:0;background:#f4f4f4;font-family:Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f4;padding:30px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
        <tr>
          <td style="background:#1a56a0;padding:24px 32px;">
            <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700;">🏫 Cantina Siloé</h1>
          </td>
        </tr>
        <tr>
          <td style="padding:32px;">
            {body_html}
          </td>
        </tr>
        <tr>
          <td style="background:#f9f9f9;padding:16px 32px;border-top:1px solid #eeeeee;">
            <p style="margin:0;color:#888888;font-size:12px;">
              Este mensaje fue enviado automáticamente por el sistema de Cantina Siloé.<br>
              Por favor no responda a este correo. Contacte a la administración del colegio para consultas.
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


async def send_welcome_parent(to: str, first_name: str) -> None:
    subject = "Bienvenido/a al sistema de Cantina Siloé"
    body_html = f"""
    <h2 style="color:#1a56a0;margin-top:0;">¡Bienvenido/a, {first_name}!</h2>
    <p style="color:#333333;line-height:1.6;">
      Su cuenta de padre/madre/tutor ha sido creada exitosamente en el sistema de Cantina Siloé.
    </p>
    <p style="color:#333333;line-height:1.6;">
      Desde la aplicación podrá:
    </p>
    <ul style="color:#333333;line-height:1.8;">
      <li>Consultar el saldo de sus hijos</li>
      <li>Ver el historial de compras</li>
      <li>Solicitar recargas de saldo</li>
      <li>Programar pedidos</li>
    </ul>
    <p style="color:#333333;line-height:1.6;">
      Si tiene alguna consulta, comuníquese con la administración del colegio.
    </p>
    <p style="color:#555555;margin-top:24px;">Saludos,<br><strong>Equipo Cantina Siloé</strong></p>
    """
    text_body = (
        f"Bienvenido/a {first_name}!\n\n"
        "Su cuenta de padre/madre/tutor ha sido creada exitosamente en el sistema de Cantina Siloé.\n\n"
        "Desde la aplicación podrá consultar saldos, ver historial de compras, solicitar recargas y programar pedidos.\n\n"
        "Saludos,\nEquipo Cantina Siloé"
    )
    await send_email(to, subject, _base_html(subject, body_html), text_body)


async def send_password_reset(to: str, first_name: str, new_password: str) -> None:
    subject = "Restablecimiento de contraseña – Cantina Siloé"
    body_html = f"""
    <h2 style="color:#1a56a0;margin-top:0;">Restablecimiento de contraseña</h2>
    <p style="color:#333333;line-height:1.6;">Hola <strong>{first_name}</strong>,</p>
    <p style="color:#333333;line-height:1.6;">
      Su contraseña ha sido restablecida por un administrador del sistema.
      A continuación encontrará sus nuevas credenciales de acceso:
    </p>
    <table style="background:#f0f4fa;border-radius:6px;padding:16px 24px;margin:16px 0;">
      <tr>
        <td style="color:#555555;padding:4px 0;"><strong>Usuario:</strong></td>
        <td style="color:#1a56a0;padding:4px 0 4px 12px;">{to}</td>
      </tr>
      <tr>
        <td style="color:#555555;padding:4px 0;"><strong>Contraseña:</strong></td>
        <td style="color:#1a56a0;padding:4px 0 4px 12px;font-family:monospace;font-size:16px;">{new_password}</td>
      </tr>
    </table>
    <p style="color:#c0392b;font-size:13px;">
      ⚠️ Por seguridad, le recomendamos cambiar su contraseña luego de iniciar sesión.
    </p>
    <p style="color:#555555;margin-top:24px;">Saludos,<br><strong>Equipo Cantina Siloé</strong></p>
    """
    text_body = (
        f"Hola {first_name},\n\n"
        "Su contraseña ha sido restablecida.\n\n"
        f"Usuario: {to}\n"
        f"Contraseña: {new_password}\n\n"
        "Por seguridad, le recomendamos cambiar su contraseña luego de iniciar sesión.\n\n"
        "Saludos,\nEquipo Cantina Siloé"
    )
    await send_email(to, subject, _base_html(subject, body_html), text_body)


async def send_low_balance_alert(to: str, parent_name: str, student_name: str, balance: int, currency: str = "Gs.") -> None:
    subject = f"Alerta: saldo bajo de {student_name} – Cantina Siloé"
    body_html = f"""
    <h2 style="color:#e67e22;margin-top:0;">⚠️ Alerta de saldo bajo</h2>
    <p style="color:#333333;line-height:1.6;">Hola <strong>{parent_name}</strong>,</p>
    <p style="color:#333333;line-height:1.6;">
      Le informamos que el saldo de <strong>{student_name}</strong> en la cantina es bajo:
    </p>
    <table style="background:#fff8f0;border:1px solid #f0c080;border-radius:6px;padding:16px 24px;margin:16px 0;">
      <tr>
        <td style="color:#555555;padding:4px 0;"><strong>Alumno/a:</strong></td>
        <td style="color:#333333;padding:4px 0 4px 12px;">{student_name}</td>
      </tr>
      <tr>
        <td style="color:#555555;padding:4px 0;"><strong>Saldo actual:</strong></td>
        <td style="color:#e67e22;padding:4px 0 4px 12px;font-size:18px;font-weight:700;">{currency} {balance:,}</td>
      </tr>
    </table>
    <p style="color:#333333;line-height:1.6;">
      Para recargar el saldo, ingrese a la aplicación y realice una solicitud de recarga.
    </p>
    <p style="color:#555555;margin-top:24px;">Saludos,<br><strong>Equipo Cantina Siloé</strong></p>
    """
    text_body = (
        f"Hola {parent_name},\n\n"
        f"El saldo de {student_name} en la cantina es bajo.\n"
        f"Saldo actual: {currency} {balance:,}\n\n"
        "Para recargar, ingrese a la aplicación y realice una solicitud de recarga.\n\n"
        "Saludos,\nEquipo Cantina Siloé"
    )
    await send_email(to, subject, _base_html(subject, body_html), text_body)


async def send_topup_approved(to: str, parent_name: str, total_amount: int, currency: str = "Gs.") -> None:
    subject = "Recarga de saldo aprobada – Cantina Siloé"
    body_html = f"""
    <h2 style="color:#27ae60;margin-top:0;">✅ Recarga aprobada</h2>
    <p style="color:#333333;line-height:1.6;">Hola <strong>{parent_name}</strong>,</p>
    <p style="color:#333333;line-height:1.6;">
      Su solicitud de recarga de saldo ha sido <strong style="color:#27ae60;">aprobada</strong>.
    </p>
    <table style="background:#f0faf4;border:1px solid #a0dbb0;border-radius:6px;padding:16px 24px;margin:16px 0;">
      <tr>
        <td style="color:#555555;padding:4px 0;"><strong>Monto acreditado:</strong></td>
        <td style="color:#27ae60;padding:4px 0 4px 12px;font-size:18px;font-weight:700;">{currency} {total_amount:,}</td>
      </tr>
    </table>
    <p style="color:#333333;line-height:1.6;">
      El saldo ya está disponible en las cuentas de sus hijos. Puede verificarlo en la aplicación.
    </p>
    <p style="color:#555555;margin-top:24px;">Saludos,<br><strong>Equipo Cantina Siloé</strong></p>
    """
    text_body = (
        f"Hola {parent_name},\n\n"
        f"Su solicitud de recarga de {currency} {total_amount:,} ha sido aprobada.\n"
        "El saldo ya está disponible en las cuentas de sus hijos.\n\n"
        "Saludos,\nEquipo Cantina Siloé"
    )
    await send_email(to, subject, _base_html(subject, body_html), text_body)


async def send_topup_rejected(to: str, parent_name: str, total_amount: int, currency: str = "Gs.") -> None:
    subject = "Solicitud de recarga rechazada – Cantina Siloé"
    body_html = f"""
    <h2 style="color:#c0392b;margin-top:0;">❌ Solicitud de recarga rechazada</h2>
    <p style="color:#333333;line-height:1.6;">Hola <strong>{parent_name}</strong>,</p>
    <p style="color:#333333;line-height:1.6;">
      Lamentablemente su solicitud de recarga de saldo ha sido <strong style="color:#c0392b;">rechazada</strong>.
    </p>
    <table style="background:#fdf0f0;border:1px solid #e0a0a0;border-radius:6px;padding:16px 24px;margin:16px 0;">
      <tr>
        <td style="color:#555555;padding:4px 0;"><strong>Monto solicitado:</strong></td>
        <td style="color:#c0392b;padding:4px 0 4px 12px;font-size:18px;font-weight:700;">{currency} {total_amount:,}</td>
      </tr>
    </table>
    <p style="color:#333333;line-height:1.6;">
      Si cree que esto es un error, comuníquese con la administración del colegio.
    </p>
    <p style="color:#555555;margin-top:24px;">Saludos,<br><strong>Equipo Cantina Siloé</strong></p>
    """
    text_body = (
        f"Hola {parent_name},\n\n"
        f"Su solicitud de recarga de {currency} {total_amount:,} ha sido rechazada.\n"
        "Si cree que esto es un error, comuníquese con la administración del colegio.\n\n"
        "Saludos,\nEquipo Cantina Siloé"
    )
    await send_email(to, subject, _base_html(subject, body_html), text_body)


async def send_link_request_approved(to: str, parent_name: str, student_name: str) -> None:
    subject = f"Vinculación aprobada: {student_name} – Cantina Siloé"
    body_html = f"""
    <h2 style="color:#27ae60;margin-top:0;">✅ Vinculación aprobada</h2>
    <p style="color:#333333;line-height:1.6;">Hola <strong>{parent_name}</strong>,</p>
    <p style="color:#333333;line-height:1.6;">
      Su solicitud de vinculación con <strong>{student_name}</strong> ha sido <strong style="color:#27ae60;">aprobada</strong>.
    </p>
    <p style="color:#333333;line-height:1.6;">
      Ya puede ver el saldo, historial de compras y gestionar pedidos para este alumno/a desde la aplicación.
    </p>
    <p style="color:#555555;margin-top:24px;">Saludos,<br><strong>Equipo Cantina Siloé</strong></p>
    """
    text_body = (
        f"Hola {parent_name},\n\n"
        f"Su solicitud de vinculación con {student_name} ha sido aprobada.\n"
        "Ya puede gestionar la cuenta del alumno/a desde la aplicación.\n\n"
        "Saludos,\nEquipo Cantina Siloé"
    )
    await send_email(to, subject, _base_html(subject, body_html), text_body)


async def send_link_request_rejected(to: str, parent_name: str, student_name: str, admin_notes: Optional[str] = None) -> None:
    subject = f"Solicitud de vinculación rechazada: {student_name} – Cantina Siloé"
    notes_html = f'<p style="color:#555555;font-style:italic;">Motivo: {admin_notes}</p>' if admin_notes else ""
    body_html = f"""
    <h2 style="color:#c0392b;margin-top:0;">❌ Solicitud de vinculación rechazada</h2>
    <p style="color:#333333;line-height:1.6;">Hola <strong>{parent_name}</strong>,</p>
    <p style="color:#333333;line-height:1.6;">
      Su solicitud de vinculación con <strong>{student_name}</strong> ha sido <strong style="color:#c0392b;">rechazada</strong>.
    </p>
    {notes_html}
    <p style="color:#333333;line-height:1.6;">
      Si cree que esto es un error, comuníquese con la administración del colegio.
    </p>
    <p style="color:#555555;margin-top:24px;">Saludos,<br><strong>Equipo Cantina Siloé</strong></p>
    """
    notes_text = f"Motivo: {admin_notes}\n" if admin_notes else ""
    text_body = (
        f"Hola {parent_name},\n\n"
        f"Su solicitud de vinculación con {student_name} ha sido rechazada.\n"
        f"{notes_text}"
        "Si cree que esto es un error, comuníquese con la administración del colegio.\n\n"
        "Saludos,\nEquipo Cantina Siloé"
    )
    await send_email(to, subject, _base_html(subject, body_html), text_body)

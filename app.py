# ============================
# TecSoft AI - Unified app.py
# Versión final mejorada (archivo único, listo para Render)
# ============================

import os
import re
import time
import json
import base64
import secrets
import hashlib
import logging
import sqlite3
import threading
from io import BytesIO
from functools import wraps
from datetime import datetime, timedelta
from contextlib import contextmanager
from collections import defaultdict

from flask import (
    Flask, request, jsonify, render_template_string, Response,
    Blueprint, g, session, send_from_directory, make_response
)
from dotenv import load_dotenv
import requests

# -------------------------
# Cargar .env
# -------------------------
load_dotenv()

# -------------------------
# Config
# -------------------------
class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
    DATABASE = os.getenv("DATABASE", "tecsoft_ai.db")
    OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", 5000))
    MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", 50000))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 20))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))  # seconds
    CACHE_TTL = int(os.getenv("CACHE_TTL", 300))
    EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")
    PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "false").lower() == "true"

# -------------------------
# App init
# -------------------------
app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("tecsoft")

# -------------------------
# Validate critical config
# -------------------------
if not app.config['OPENROUTER_KEY']:
    logger.error("OPENROUTER_KEY no configurada. Define en .env")
    # No raise so local dev can still run some endpoints — but most AI features will error
    # raise ValueError("OPENROUTER_KEY no configurada")

# -------------------------
# Simple in-memory rate limiter & metrics
# -------------------------
_rate_limit_store = defaultdict(list)
_rate_limit_lock = threading.Lock()

_metrics = defaultdict(int)  # simple counters
_metrics_lock = threading.Lock()

def check_rate_limit(ip):
    """Return True if allowed, False if rate limited."""
    now = time.time()
    window = app.config['RATE_LIMIT_WINDOW']
    maxreq = app.config['RATE_LIMIT_REQUESTS']
    with _rate_limit_lock:
        lst = _rate_limit_store[ip]
        # remove old
        while lst and now - lst[0] > window:
            lst.pop(0)
        if len(lst) >= maxreq:
            return False
        lst.append(now)
        return True

def rate_limited(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        if not check_rate_limit(ip):
            return jsonify({'error': 'Demasiadas solicitudes. Inténtalo más tarde.'}), 429
        return f(*args, **kwargs)
    return wrapper

def track_metrics(name):
    """Decorator factory to increment a metric counter by endpoint name."""
    def _dec(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with _metrics_lock:
                _metrics[name] += 1
            return f(*args, **kwargs)
        return wrapper
    return _dec

# -------------------------
# DB helpers (sqlite)
# -------------------------
_db_lock = threading.Lock()

@contextmanager
def get_db():
    conn = sqlite3.connect(app.config['DATABASE'], check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with _db_lock, get_db() as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            verified INTEGER DEFAULT 0,
            verify_token TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """)
        db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        db.commit()

init_db()

# -------------------------
# Utilities: sanitization, hashing, tokens
# -------------------------
CLEAN_RE = re.compile(r'[^\w\s\.\,\!\?\-\(\)\[\]\{\}\:\;\'\"]', re.UNICODE)

def sanitize_text(text: str) -> str:
    if text is None:
        return ""
    return CLEAN_RE.sub("", str(text)).strip()

def hash_password(password: str) -> str:
    # bcrypt is preferable; using sha256 if bcrypt isn't installed.
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(password: str, hashval: str) -> bool:
    return hash_password(password) == hashval

def generate_token() -> str:
    return secrets.token_urlsafe(32)

def generate_verify_token() -> str:
    return secrets.token_urlsafe(20)

# -------------------------
# Simple token "auth" (replace with JWT in prod)
# -------------------------
# For simplicity: token -> user_id stored in-memory (short lived). In prod use JWT or DB table.
_auth_tokens = {}
_auth_lock = threading.Lock()
TOKEN_TTL = 60 * 60 * 24  # 24h

def create_auth_token(user_id: int) -> str:
    token = generate_token()
    with _auth_lock:
        _auth_tokens[token] = (user_id, time.time() + TOKEN_TTL)
    return token

def verify_token(token: str):
    if not token:
        return None
    with _auth_lock:
        data = _auth_tokens.get(token)
        if not data:
            return None
        user_id, expires = data
        if time.time() > expires:
            del _auth_tokens[token]
            return None
        return user_id

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        hdr = request.headers.get("Authorization", "")
        token = hdr.replace("Bearer ", "")
        user_id = verify_token(token)
        if not user_id:
            return jsonify({'error': 'Autenticación requerida'}), 401
        # attach user info to g
        with get_db() as db:
            row = db.execute("SELECT id, username, email, role FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                return jsonify({'error': 'Usuario no encontrado'}), 401
            g.user_id = row['id']
            g.username = row['username']
            g.user_role = row['role']
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    @require_auth
    def decorated_function(*args, **kwargs):
        if getattr(g, "user_role", None) != 'admin':
            return jsonify({'error': 'Acceso denegado'}), 403
        return f(*args, **kwargs)
    return decorated_function

# -------------------------
# Email sending (simulado/real)
# -------------------------
def send_email_simulated(to, subject, body):
    logger.info("Simulated email -> To: %s Subject: %s Body: %s", to, subject, body)
    return True

def send_email_real(to, subject, body):
    import smtplib
    from email.mime.text import MIMEText
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = app.config['SMTP_USER']
    msg['To'] = to
    try:
        server = smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT'])
        server.starttls()
        server.login(app.config['SMTP_USER'], app.config['SMTP_PASS'])
        server.sendmail(app.config['SMTP_USER'], [to], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        logger.error("Error sending email: %s", e)
        return False

def send_email(to, subject, body):
    if app.config['EMAIL_ENABLED']:
        return send_email_real(to, subject, body)
    else:
        return send_email_simulated(to, subject, body)

# -------------------------
# Cache for responses
# -------------------------
from cachetools import TTLCache
response_cache = TTLCache(maxsize=200, ttl=app.config['CACHE_TTL'])

# -------------------------
# OpenRouter wrapper
# -------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def query_model(model, messages, timeout=25):
    """Robust call with retries and basic error handling."""
    headers = {"Authorization": f"Bearer {app.config['OPENROUTER_KEY']}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}
    tries = 3
    for attempt in range(tries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                logger.warning("OpenRouter non-200: %s %s", resp.status_code, resp.text[:200])
                time.sleep(1 + attempt)
                continue
            parsed = resp.json()
            # flexible parsing
            content = parsed.get("choices", [{}])[0].get("message", {}).get("content")
            if content is None:
                content = parsed.get("choices", [{}])[0].get("text") or parsed.get("result") or ""
            return content or "Sin respuesta del modelo."
        except requests.exceptions.Timeout:
            logger.warning("Timeout calling OpenRouter (attempt %s)", attempt+1)
            time.sleep(1 + attempt)
        except Exception as e:
            logger.exception("Error calling OpenRouter: %s", e)
            time.sleep(1 + attempt)
    return "Error: no se pudo conectar con el modelo."

# -------------------------
# Blueprints & routes
# -------------------------
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Serve uploaded files
@app.route('/uploads/<path:filename>')
@require_auth
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

# Home / HTML template (your existing stylized template)
HTML_TEMPLATE = """ ... (omitted here for brevity) ... """
# Note: In your deploy copy the full HTML_TEMPLATE you already have.
# For brevity in this file I will render a small page if the big template is not set.
if HTML_TEMPLATE.strip().startswith("..."):
    HTML_TEMPLATE = "<html><body><h1>TecSoft AI</h1><p>Interfaz.</p></body></html>"

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

# -------------------------
# Register endpoint
# -------------------------
@api_bp.route('/register', methods=['POST'])
@rate_limited
@track_metrics('register')
def register():
    """
    Register a user:
      - validate username/email/password
      - store hashed password
      - create verification token and (simulate) send email
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': 'JSON inválido'}), 400

    username = sanitize_text(data.get('username', '')).lower()
    email = sanitize_text(data.get('email', '')).lower()
    password = data.get('password', '')

    if not username or not email or not password:
        return jsonify({'error': 'Todos los campos son requeridos'}), 400
    if len(username) < 3 or len(username) > 30:
        return jsonify({'error': 'Usuario debe tener entre 3 y 30 caracteres'}), 400
    if '@' not in email or len(email) < 5:
        return jsonify({'error': 'Email inválido'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Contraseña muy corta (min 6 caracteres)'}), 400

    password_hash = hash_password(password)
    verify_token = generate_verify_token()

    try:
        with _db_lock, get_db() as db:
            cur = db.execute("INSERT INTO users (username, email, password_hash, verify_token) VALUES (?, ?, ?, ?)",
                             (username, email, password_hash, verify_token))
            db.commit()
            user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Usuario o email ya existe'}), 400
    except Exception as e:
        logger.exception("Error registrando usuario: %s", e)
        return jsonify({'error': 'Error interno al registrar'}), 500

    # Send verification email (simulated unless configured)
    verify_url = f"https://tu-dominio/verify?token={verify_token}&uid={user_id}"
    subject = "Verifica tu cuenta - TecSoft AI"
    body = f"Hola {username},\n\nUsa este enlace para verificar tu cuenta:\n\n{verify_url}\n\nSi no solicitaste esto, ignora."
    send_email(email, subject, body)

    return jsonify({'message': 'Usuario registrado. Revisa tu correo para verificar la cuenta.'}), 201

# -------------------------
# Verify endpoint
# -------------------------
@api_bp.route('/verify', methods=['GET'])
def verify():
    token = request.args.get('token')
    uid = request.args.get('uid')
    if not token or not uid:
        return jsonify({'error': 'Token o uid faltante'}), 400
    try:
        with _db_lock, get_db() as db:
            row = db.execute("SELECT id FROM users WHERE id = ? AND verify_token = ?", (uid, token)).fetchone()
            if not row:
                return jsonify({'error': 'Token inválido o expirado'}), 400
            db.execute("UPDATE users SET verified = 1, verify_token = NULL WHERE id = ?", (uid,))
            db.commit()
        return jsonify({'message': 'Cuenta verificada correctamente.'})
    except Exception as e:
        logger.exception("Error verificando usuario: %s", e)
        return jsonify({'error': 'Error interno'}), 500

# -------------------------
# Login endpoint
# -------------------------
@api_bp.route('/login', methods=['POST'])
@rate_limited
@track_metrics('login')
def login():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': 'JSON inválido'}), 400

    username = sanitize_text(data.get('username', '')).lower()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Usuario y contraseña requeridos'}), 400

    try:
        with get_db() as db:
            row = db.execute("SELECT id, password_hash, verified, role FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return jsonify({'error': 'Credenciales inválidas'}), 401
        if not verify_password(password, row['password_hash']):
            return jsonify({'error': 'Credenciales inválidas'}), 401
        if not row['verified']:
            return jsonify({'error': 'Cuenta no verificada'}), 403
        token = create_auth_token(row['id'])
        return jsonify({'token': token})
    except Exception as e:
        logger.exception("Error en login: %s", e)
        return jsonify({'error': 'Error interno'}), 500

# -------------------------
# Text chat endpoint
# -------------------------
@api_bp.route('/text', methods=['POST'])
@require_auth
@rate_limited
@track_metrics('text')
def api_text():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': 'JSON inválido'}), 400

    messages = data.get('messages')
    if not isinstance(messages, list) or not messages:
        return jsonify({'error': 'Lista de mensajes requerida'}), 400

    # sanitize and limit
    sanitized = []
    total_len = 0
    for m in messages:
        if not isinstance(m, dict) or 'role' not in m or 'content' not in m:
            return jsonify({'error': 'Formato de mensaje inválido'}), 400
        content = sanitize_text(m['content'])
        if len(content) > app.config['MAX_MESSAGE_LENGTH']:
            return jsonify({'error': f'Mensaje demasiado largo (máx. {app.config["MAX_MESSAGE_LENGTH"]})'}), 400
        sanitized.append({'role': m['role'], 'content': content})
        total_len += len(content)
    if total_len > app.config['MAX_HISTORY_LENGTH']:
        return jsonify({'error': 'Historial demasiado largo'}), 400

    # persist messages to DB (session management)
    session_id = data.get('session_id') or secrets.token_hex(16)
    try:
        with _db_lock, get_db() as db:
            for m in sanitized:
                db.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                           (session_id, m['role'], m['content']))
            db.commit()
    except Exception as e:
        logger.exception("Error guardando mensajes: %s", e)

    # prepare payload for model
    # model expects messages as list of dicts role/content
    # try cache with a simple key
    cache_key = hashlib.sha256(json.dumps(sanitized, sort_keys=True).encode()).hexdigest()
    if cache_key in response_cache:
        reply = response_cache[cache_key]
    else:
        # call model (use first supported model)
        model = app.config.get('SUPPORTED_MODELS', ["x-ai/grok-4.1-fast"])[0] if "SUPPORTED_MODELS" in app.config else "x-ai/grok-4.1-fast"
        reply = query_model(model, sanitized)
        response_cache[cache_key] = reply

    try:
        with _db_lock, get_db() as db:
            db.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                       (session_id, 'assistant', reply))
            db.commit()
    except Exception:
        logger.exception("Error guardando respuesta de assistant")

    return jsonify({'reply': reply, 'session_id': session_id})

# -------------------------
# Image analysis endpoint
# -------------------------
@api_bp.route('/image', methods=['POST'])
@require_auth
@rate_limited
@track_metrics('image')
def api_image():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': 'JSON inválido'}), 400

    text = sanitize_text(data.get('text', ''))
    image_url = data.get('image_url', '').strip()
    if not text or not image_url:
        return jsonify({'error': 'Texto e image_url requeridos'}), 400
    if len(text) > app.config['MAX_MESSAGE_LENGTH']:
        return jsonify({'error': 'Texto demasiado largo'}), 400
    if not image_url.startswith(('http://', 'https://')) or len(image_url) > 2000:
        return jsonify({'error': 'URL de imagen inválida'}), 400

    # Build a model message — some models accept structured message, here we pass simple content.
    messages = [{'role': 'user', 'content': f"[IMAGE] {image_url}\n\n{text}"}]
    model = app.config.get('SUPPORTED_MODELS', ["x-ai/grok-4.1-fast"])[0] if "SUPPORTED_MODELS" in app.config else "x-ai/grok-4.1-fast"
    reply = query_model(model, messages)
    return jsonify({'reply': reply})

# -------------------------
# Voice to text endpoint (simulated)
# -------------------------
@api_bp.route('/voice-to-text', methods=['POST'])
@require_auth
@rate_limited
@track_metrics('voice_to_text')
def voice_to_text():
    # Expecting a form-data file field named 'audio' (blob)
    if 'audio' not in request.files:
        return jsonify({'error': 'Archivo de audio requerido (campo "audio")'}), 400
    audio = request.files['audio']
    # In this demo we don't call an STT service. Instead we simulate a short transcription.
    simulated = "Transcripción simulada: " + (audio.filename or "audio grabado")
    return jsonify({'text': simulated})

# -------------------------
# Analytics endpoint - returns small SVG base64 representing usage
# -------------------------
def build_simple_svg_chart(metrics_dict):
    # Build a tiny bar chart as SVG
    keys = list(metrics_dict.keys())[:6]
    values = [metrics_dict[k] for k in keys]
    maxv = max(values) if values else 1
    width = 600
    height = 200
    bar_w = width / max(len(values),1)
    svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    svg_parts.append('<style>.t{font:12px sans-serif; fill:#ddd}</style>')
    for i, (k, v) in enumerate(zip(keys, values)):
        h = (v / maxv) * (height - 40)
        x = i * bar_w + 10
        y = height - h - 20
        svg_parts.append(f'<rect x="{x}" y="{y}" width="{bar_w-20}" height="{h}" fill="#0f9dff"/>')
        svg_parts.append(f'<text x="{x+2}" y="{height-4}" class="t">{k}</text>')
    svg_parts.append('</svg>')
    svg = ''.join(svg_parts)
    b64 = base64.b64encode(svg.encode()).decode()
    return b64

@api_bp.route('/analytics', methods=['GET'])
@require_auth
@rate_limited
@track_metrics('analytics')
def analytics():
    # Only admin allowed
    if getattr(g, "user_role", None) != 'admin':
        return jsonify({'error': 'Acceso denegado'}), 403
    with _metrics_lock:
        snapshot = dict(_metrics)
    chart_b64 = build_simple_svg_chart(snapshot)
    return jsonify({'chart': chart_b64, 'metrics': snapshot})

# -------------------------
# Export chat endpoint (returns a small text file or PDF placeholder)
# -------------------------
@api_bp.route('/export-chat', methods=['GET'])
@require_auth
@rate_limited
@track_metrics('export_chat')
def export_chat():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'session_id requerido (query)'}), 400
    try:
        with get_db() as db:
            rows = db.execute("SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp", (session_id,)).fetchall()
            if not rows:
                return jsonify({'error': 'Sesión no encontrada'}), 404
            lines = []
            for r in rows:
                ts = r['timestamp']
                role = r['role']
                content = r['content']
                lines.append(f"[{ts}] {role}: {content}")
            payload = "\n".join(lines)
            # return as downloadable text file
            resp = make_response(payload)
            resp.headers['Content-Type'] = 'text/plain; charset=utf-8'
            resp.headers['Content-Disposition'] = f'attachment; filename=chat_{session_id}.txt'
            return resp
    except Exception as e:
        logger.exception("Error exporting chat: %s", e)
        return jsonify({'error': 'Error interno'}), 500

# -------------------------
# Metrics for Prometheus (optional)
# -------------------------
@app.route('/metrics')
@admin_required
def metrics():
    if not app.config['PROMETHEUS_ENABLED']:
        return handle_error('Métricas no habilitadas', 404)
    # Minimal Prometheus-like text
    with _metrics_lock:
        lines = []
        for k, v in _metrics.items():
            lines.append(f'tecsoft_metric{{name="{k}"}} {v}')
        return Response("\n".join(lines), mimetype='text/plain')

# -------------------------
# Register blueprint
# -------------------------
app.register_blueprint(api_bp)

# -------------------------
# Error handlers (reuse earlier handle_error)
# -------------------------
def handle_error(error_message, status_code=500):
    logger.error(error_message)
    return jsonify({'error': error_message}), status_code

@app.errorhandler(404)
def not_found(e):
    return handle_error('Ruta no encontrada', 404)

@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error: %s", e)
    return handle_error('Error interno del servidor', 500)

@app.errorhandler(429)
def too_many(e):
    return handle_error('Demasiadas solicitudes. Inténtalo más tarde.', 429)

# -------------------------
# Run
# -------------------------
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    logger.info("Arrancando TecSoft AI en puerto %s", port)
    app.run(host='0.0.0.0', port=port, debug=False)

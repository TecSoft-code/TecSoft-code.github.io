# app.py - Versión mejorada y corregida
import os
import re
import time
import json
import sqlite3
import logging
import secrets
import smtplib
from email.mime.text import MIMEText
from contextlib import contextmanager
from threading import Lock, Thread
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import wraps

import requests
import jwt
from dotenv import load_dotenv
from flask import (
    Flask, g, request, jsonify, abort, Response, Blueprint,
    stream_with_context
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from waitress import serve

# --- Cargar entorno ---
load_dotenv()

# --- Configuración ---
class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    JWT_SECRET = os.getenv("JWT_SECRET", SECRET_KEY)
    JWT_EXPIRATION_DAYS = int(os.getenv("JWT_EXPIRATION_DAYS", 7))
    CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(',')
    DATABASE = os.getenv("DATABASE_PATH", "tecsoft_ai.db")
    MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", 5000))
    MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", 50000))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 10))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))
    OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
    BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
    SUPPORTED_MODELS = os.getenv("SUPPORTED_MODELS", "x-ai/grok-4.1-fast,mistralai/mixtral-8x7b-instruct").split(",")
    DEFAULT_MODEL = SUPPORTED_MODELS[0] if SUPPORTED_MODELS else None
    EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")
    # Timeouts
    OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", 60))
    # Limits
    SESSION_ID_MAX_LEN = int(os.getenv("SESSION_ID_MAX_LEN", 64))

# --- Aplicación ---
app = Flask(__name__)
app.config.from_object(Config)

# --- CORS ---
CORS(app, resources={r"/api/*": {"origins": app.config['CORS_ALLOWED_ORIGINS']}})

# --- Validación inicial crítica ---
if not app.config['OPENROUTER_KEY']:
    raise ValueError("La variable de entorno OPENROUTER_KEY es obligatoria.")

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","message":"%(message)s"}',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- DB context manager ---
@contextmanager
def get_db():
    """
    Provee una conexión SQLite almacenada en flask.g por request.
    No usar la misma conexión entre hilos.
    """
    db = getattr(g, '_database', None)
    if db is None:
        # Por seguridad, no permitir check_same_thread en conexiones globales
        db = g._database = sqlite3.connect(app.config['DATABASE'], detect_types=sqlite3.PARSE_DECLTYPES)
        db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        # No cerramos aquí (teardown_appcontext lo hace) para preservar la conexión por request
        pass

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        try:
            db.close()
        except Exception as e:
            logger.warning(f"Error cerrando conexión DB: {e}")
        g._database = None

def init_db():
    """Inicializa las tablas (usar correctamente el context manager)."""
    with app.app_context():
        with get_db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'user' NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
            ''')
            db.commit()
            logger.info("Base de datos inicializada.")

# Inicializar DB
with app.app_context():
    init_db()

# --- Rate limiting (thread-safe) ---
rate_limit_store = defaultdict(list)
rate_limit_lock = Lock()

def _get_remote_ip():
    # Respetar cabecera X-Forwarded-For si existe (proxy/reverse-proxy)
    xff = request.headers.get('X-Forwarded-For', '')
    if xff:
        # tomar la primer IP
        return xff.split(',')[0].strip()
    return request.remote_addr or 'unknown'

def check_rate_limit(ip):
    with rate_limit_lock:
        now = time.time()
        window = app.config['RATE_LIMIT_WINDOW']
        history = rate_limit_store[ip]
        # limpiar entradas viejas
        rate_limit_store[ip] = [t for t in history if now - t < window]
        if len(rate_limit_store[ip]) >= app.config['RATE_LIMIT_REQUESTS']:
            return False
        rate_limit_store[ip].append(now)
        return True

# --- Utilidades DB/Seguridad ---
def generate_session_id():
    return secrets.token_urlsafe(32)[:app.config['SESSION_ID_MAX_LEN']]

def save_session(session_id, user_id):
    try:
        with get_db() as db:
            db.execute('INSERT OR IGNORE INTO sessions (id, user_id) VALUES (?, ?)', (session_id, user_id))
            db.commit()
    except Exception as e:
        logger.error(f"Error guardando sesión {session_id}: {e}")

def save_message_async(session_id, role, content):
    """
    Guarda mensajes en hilo separado. Se abre una nueva conexión para evitar problemas
    con el 'g' de Flask que no es seguro entre hilos.
    """
    def run_save(sid, r, c):
        try:
            conn = sqlite3.connect(app.config['DATABASE'])
            conn.row_factory = sqlite3.Row
            conn.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)', (sid, r, c))
            conn.commit()
            conn.close()
            logger.info(f"Mensaje guardado async para sesión {sid} (role={r})")
        except Exception as exc:
            logger.error(f"Error asíncrono guardando mensaje: {exc}")

    # Lanzar hilo daemon para que no impida shutdown
    t = Thread(target=run_save, args=(session_id, role, content), daemon=True)
    t.start()

def generate_auth_token(user_id):
    try:
        payload = {
            'exp': datetime.now(timezone.utc) + timedelta(days=app.config['JWT_EXPIRATION_DAYS']),
            'iat': datetime.now(timezone.utc),
            'sub': int(user_id)
        }
        return jwt.encode(payload, app.config['JWT_SECRET'], algorithm='HS256')
    except Exception as e:
        logger.error(f"Error generando JWT: {e}")
        return None

def verify_auth_token(token):
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
        return int(payload.get('sub'))
    except jwt.ExpiredSignatureError:
        return 'expired'
    except jwt.InvalidTokenError:
        return None

# --- Decoradores ---
def rate_limited(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = _get_remote_ip()
        if not check_rate_limit(ip):
            logger.warning(f"Rate limit exceeded for IP: {ip}")
            abort(429, description="Límite de peticiones excedido.")
        return f(*args, **kwargs)
    return decorated_function

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            abort(401, description='Autenticación requerida. Token Bearer faltante.')
        token = auth_header.split(' ', 1)[1]
        user_id = verify_auth_token(token)
        if user_id == 'expired':
            abort(401, description='Token de autenticación expirado.')
        if not user_id:
            abort(401, description='Token de autenticación inválido.')
        g.user_id = user_id
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = getattr(g, 'user_id', None)
        if not user_id:
            abort(401, description='Autenticación requerida.')
        with get_db() as db:
            row = db.execute('SELECT role FROM users WHERE id = ?', (user_id,)).fetchone()
            if not row or row['role'] != 'admin':
                abort(403, description='Permiso de administrador requerido.')
        return f(*args, **kwargs)
    return decorated_function

# --- Handlers de error centralizados ---
@app.errorhandler(400)
@app.errorhandler(401)
@app.errorhandler(403)
@app.errorhandler(404)
@app.errorhandler(429)
def handle_http_error(e):
    status_code = getattr(e, 'code', 500)
    description = getattr(e, 'description', 'Error')
    logger.error(f"HTTP {status_code} - {description}")
    return jsonify({'error': description}), status_code

@app.errorhandler(500)
def internal_error(e):
    logger.exception('Error interno del servidor')
    return jsonify({'error': 'Error interno del servidor. Revisa el log.'}), 500

# --- Implementación de funciones faltantes usadas por rutas ---
def send_email(to_email, subject, body):
    """Enviar email de forma segura si está habilitado."""
    if not app.config['EMAIL_ENABLED']:
        logger.info("Email no enviado (EMAIL_ENABLED=False).")
        return False
    try:
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = app.config['SMTP_USER']
        msg['To'] = to_email

        server = smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT'], timeout=10)
        server.starttls()
        server.login(app.config['SMTP_USER'], app.config['SMTP_PASS'])
        server.send_message(msg)
        server.quit()
        logger.info(f"Email enviado a {to_email}")
        return True
    except Exception as e:
        logger.error(f"Error enviando email a {to_email}: {e}")
        return False

def get_messages(session_id, limit=50):
    """Recupera mensajes de la sesión. Devuelve lista de dicts [{'role':..., 'content': ...}, ...]"""
    try:
        with get_db() as db:
            rows = db.execute(
                'SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?',
                (session_id, limit)
            ).fetchall()
            return [{'role': r['role'], 'content': r['content']} for r in rows]
    except Exception as e:
        logger.error(f"Error obteniendo historial para session {session_id}: {e}")
        return []

# --- Lógica de streaming hacia OpenRouter (con SSE) ---
def stream_query_model(messages, model, temperature, max_tokens, timeout=None):
    """
    Llama a la API externa con stream=True y emite chunks en formato SSE.
    Si la API devuelve errores, se envía un chunk con 'error'.
    """
    timeout = timeout or app.config['OPENROUTER_TIMEOUT']
    if model not in app.config['SUPPORTED_MODELS']:
        yield f"data: {json.dumps({'error': 'Modelo no soportado.'})}\n\n"
        return

    headers = {
        "Authorization": f"Bearer {app.config['OPENROUTER_KEY']}",
        "Content-Type": "application/json",
        "User-Agent": "TecSoftAI/3.0 - Production"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": True
    }

    try:
        logger.info(f"STREAM REQ model={model} temp={temperature} max_tokens={max_tokens}")
        resp = requests.post(app.config['BASE_URL'], headers=headers, json=payload, stream=True, timeout=timeout)
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            # OpenRouter -> prefijo "data: " esperado; si no existe intentamos parsear
            line_str = line
            if line_str.startswith('data: '):
                data = line_str[6:].strip()
            else:
                data = line_str.strip()

            if data == '[DONE]':
                break

            try:
                chunk = json.loads(data)
                content_chunk = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if content_chunk:
                    yield f"data: {json.dumps({'text': content_chunk})}\n\n"
            except json.JSONDecodeError:
                # No JSON válido; enviar como raw para diagnóstico
                yield f"data: {json.dumps({'raw': data})}\n\n"

    except requests.exceptions.RequestException as e:
        logger.error(f"Error en la API (Stream): {e}")
        yield f"data: {json.dumps({'error': f'Error en la API: {str(e)}'})}\n\n"
    except Exception as e:
        logger.exception("Error desconocido en streaming")
        yield f"data: {json.dumps({'error': f'Error interno: {str(e)}'})}\n\n"

# --- Blueprints y rutas ---
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/register', methods=['POST'])
@rate_limited
def register():
    data = request.get_json(silent=True) or {}
    username = re.sub(r'\s+', '', (data.get('username') or '')).lower()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not username or not email or not password or '@' not in email:
        abort(400, description='Usuario, email válido y contraseña son requeridos.')

    if len(username) < 3 or len(password) < 6:
        abort(400, description='Nombre de usuario mínimo 3 caracteres y contraseña mínimo 6 caracteres.')

    password_hash = generate_password_hash(password)
    try:
        with get_db() as db:
            db.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)', (username, email, password_hash))
            db.commit()
    except sqlite3.IntegrityError:
        abort(400, description='Usuario o email ya existe.')
    except Exception as e:
        logger.error(f"Error registrando usuario: {e}")
        abort(500, description='Error registrando usuario.')

    # Enviar email en un hilo (no bloqueante)
    Thread(target=send_email, args=(email, "Bienvenido a TecSoft AI", "Tu cuenta ha sido creada exitosamente."), daemon=True).start()
    return jsonify({'message': 'Usuario registrado exitosamente. Por favor, inicia sesión.'}), 201

@api_bp.route('/login', methods=['POST'])
@rate_limited
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip().lower()
    password = data.get('password') or ''

    if not username or not password:
        abort(400, description='Usuario y contraseña son requeridos.')

    with get_db() as db:
        user = db.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,)).fetchone()

    if user and check_password_hash(user['password_hash'], password):
        token = generate_auth_token(user['id'])
        if token:
            return jsonify({'token': token}), 200
        abort(500, description='Error generando token.')
    abort(401, description='Credenciales inválidas.')

@api_bp.route('/text/stream', methods=['POST'])
@login_required
@rate_limited
def api_text_stream():
    data = request.get_json(silent=True)
    if not data:
        abort(400, description='Datos JSON requeridos.')

    user_id = g.user_id
    session_id = data.get('session_id') or generate_session_id()
    if len(session_id) > app.config['SESSION_ID_MAX_LEN']:
        abort(400, description='session_id demasiado largo.')

    prompt = (data.get('prompt') or '').strip()
    model_name = data.get('model') or app.config['DEFAULT_MODEL']
    try:
        temperature = float(data.get('temperature', 0.7))
    except Exception:
        temperature = 0.7
    try:
        max_tokens = int(data.get('max_tokens', 2048))
    except Exception:
        max_tokens = 2048

    if model_name not in app.config['SUPPORTED_MODELS']:
        abort(400, description=f'Modelo "{model_name}" no soportado.')

    if not prompt:
        abort(400, description='El campo "prompt" es requerido.')

    # Saneamiento: permitir caracteres normales y signos de puntuación básicos
    sanitized_prompt = re.sub(r'[^\w\s\.\,\!\?\-\$\{\}\:\;\'\"\_@#%&\(\)]', '', prompt).strip()
    if len(sanitized_prompt) > app.config['MAX_MESSAGE_LENGTH']:
        abort(400, description='Mensaje demasiado largo.')

    # Guardar sesión y recuperar historial
    save_session(session_id, user_id)
    history = get_messages(session_id)

    total_length = sum(len(msg['content']) for msg in history) + len(sanitized_prompt)
    if total_length > app.config['MAX_HISTORY_LENGTH']:
        abort(400, description=f'Historial demasiado largo (máx. {app.config["MAX_HISTORY_LENGTH"]} caracteres).')

    # Guardar mensaje del usuario (asíncrono) antes del stream
    save_message_async(session_id, 'user', sanitized_prompt)

    # Construir payload para la API
    messages = history + [{'role': 'user', 'content': sanitized_prompt}]

    response_generator = stream_query_model(messages, model_name, temperature, max_tokens, timeout=app.config['OPENROUTER_TIMEOUT'])

    @stream_with_context
    def generate_stream():
        full_response = ""
        for chunk in response_generator:
            # chunk viene en formato SSE "data: {...}\n\n"
            if chunk.startswith("data: "):
                try:
                    data_json = json.loads(chunk[6:].strip())
                except Exception:
                    data_json = {}
                text_content = data_json.get('text') or ''
                error_content = data_json.get('error') or ''
                if text_content:
                    full_response += text_content
                if error_content:
                    logger.error(f"Stream terminó con error: {error_content}")
                    yield chunk
                    return
            yield chunk

        # Guardar respuesta completa (si existe)
        if full_response:
            save_message_async(session_id, 'assistant', full_response)

        # Señal de fin
        yield f"data: {json.dumps({'session_id': session_id, 'end_stream': True})}\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')

@api_bp.route('/vision', methods=['POST'])
@login_required
@rate_limited
def api_vision():
    # Endpoint multimodal por ahora no implementado
    abort(501, description="Endpoint multimodal no implementado en esta versión.")

# Registrar blueprint
app.register_blueprint(api_bp)

# --- Factory / Servir ---
def create_app():
    logger.info("Aplicación TecSoft AI V3.0 inicializada.")
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Iniciando aplicación en puerto {port}...")
    if os.getenv("FLASK_ENV") == "development":
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        # Waitress en producción
        serve(app, host='0.0.0.0', port=port)

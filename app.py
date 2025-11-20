from flask import Flask, request, jsonify, render_template_string, Response, Blueprint, g, abort, stream_with_context
import requests
import json
import os
import logging
from dotenv import load_dotenv
from functools import wraps
import time
from collections import defaultdict
import re
import sqlite3
from contextlib import contextmanager
import smtplib
from email.mime.text import MIMEText
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta, timezone
from threading import Lock, Thread # 💡 Usar Thread para operaciones asíncronas (guardado)
from flask_cors import CORS # 🔒 CORS para entornos de producción/frontend
from waitress import serve # 💡 Servidor WSGI de producción (alternativa a Gunicorn)

# --- Configuración y Entorno ---
load_dotenv()

# Clase de Configuración Detallada
class Config:
    # Seguridad
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    JWT_SECRET = os.getenv("JWT_SECRET", SECRET_KEY) 
    JWT_EXPIRATION_DAYS = int(os.getenv("JWT_EXPIRATION_DAYS", 7))
    CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(',') # Especificar dominios
    
    # Base de Datos
    DATABASE = os.getenv("DATABASE_PATH", "tecsoft_ai.db")
    
    # Límites de Usuario/Sistema
    MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", 5000))
    MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", 50000))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 10))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))
    
    # API de IA
    OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    SUPPORTED_MODELS = os.getenv("SUPPORTED_MODELS", "x-ai/grok-4.1-fast,mistralai/mixtral-8x7b-instruct").split(",")
    DEFAULT_MODEL = SUPPORTED_MODELS[0]
    
    # Correo Electrónico
    EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")

# --- Inicialización de Flask y Componentes de Arquitectura ---
app = Flask(__name__)
app.config.from_object(Config)

# 🔒 Habilitar CORS para permitir llamadas desde un frontend separado
CORS(app, resources={r"/api/*": {"origins": app.config['CORS_ALLOWED_ORIGINS']}})

# ⚠️ Validación Crítica
if not app.config['OPENROUTER_KEY']:
    raise ValueError("La variable de entorno OPENROUTER_KEY es obligatoria.")

# --- Logging Estructurado (Mejorado) ---
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Base de Datos SQLite (Manejador de Contexto) ---
@contextmanager
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        db.row_factory = sqlite3.Row
    yield db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.executescript('''
            -- Roles: 'user' o 'admin'
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user' NOT NULL, -- 💡 Campo de Rol
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

# Inicializar DB al iniciar
with app.app_context():
    init_db()

# --- Concurrencia y Rate Limiting Thread-Safe ---
rate_limit_store = defaultdict(list)
rate_limit_lock = Lock() 

def check_rate_limit(ip):
    with rate_limit_lock:
        now = time.time()
        rate_limit_store[ip] = [t for t in rate_limit_store[ip] if now - t < app.config['RATE_LIMIT_WINDOW']]
        if len(rate_limit_store[ip]) >= app.config['RATE_LIMIT_REQUESTS']:
            return False
        rate_limit_store[ip].append(now)
        return True

# --- Funciones de Seguridad y Persistencia ---
def generate_session_id():
    return secrets.token_urlsafe(16)

def save_session(session_id, user_id):
    # Usar INSERT OR IGNORE para que no falle si la sesión ya existe (manejo de sesiones continuas)
    with get_db() as db:
        db.execute('INSERT OR IGNORE INTO sessions (id, user_id) VALUES (?, ?)', (session_id, user_id))
        db.commit()

# 💡 Función Asíncrona para guardar mensajes (no bloquea el stream)
def save_message_async(session_id, role, content):
    def run_save():
        try:
            # Reabrir una conexión dentro del thread
            temp_db = sqlite3.connect(app.config['DATABASE'])
            temp_db.row_factory = sqlite3.Row
            temp_db.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)', (session_id, role, content))
            temp_db.commit()
            temp_db.close()
            logger.info(f"Mensaje guardado de forma asíncrona para sesión: {session_id}")
        except Exception as e:
            logger.error(f"Error asíncrono guardando mensaje: {str(e)}")

    Thread(target=run_save).start()
    
# --- JWT y Auth ---
def generate_auth_token(user_id):
    try:
        payload = {
            'exp': datetime.now(timezone.utc) + timedelta(days=app.config['JWT_EXPIRATION_DAYS']),
            'iat': datetime.now(timezone.utc),
            'sub': user_id
        }
        return jwt.encode(payload, app.config['JWT_SECRET'], algorithm='HS256')
    except Exception as e:
        logger.error(f"Error generando JWT: {e}")
        return None

def verify_auth_token(token):
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
        return payload['sub']
    except jwt.ExpiredSignatureError:
        return 'expired'
    except jwt.InvalidTokenError:
        return None

# --- Decoradores ---
def rate_limited(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        if not check_rate_limit(ip):
            logger.warning(f"Rate limit exceeded for IP: {ip}")
            abort(429)
        return f(*args, **kwargs)
    return decorated_function

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            abort(401, description='Autenticación requerida. Token Bearer faltante.')
        
        token = auth_header.split(' ')[1]
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
        user_id = g.user_id
        with get_db() as db:
            user = db.execute('SELECT role FROM users WHERE id = ?', (user_id,)).fetchone()
            if not user or user['role'] != 'admin':
                abort(403, description='Permiso de administrador requerido.')
        return f(*args, **kwargs)
    return decorated_function

# --- Manejadores de Error Centralizados ---
# 💡 Usar abort() en vez de return jsonify para activar estos handlers
@app.errorhandler(400)
@app.errorhandler(401)
@app.errorhandler(403)
@app.errorhandler(404)
@app.errorhandler(429)
def handle_http_error(e):
    # Captura el mensaje de error personalizado de abort(status_code, description=...)
    status_code = getattr(e, 'code', 500)
    description = getattr(e, 'description', 'Error interno del servidor')
    
    logger.error(f"HTTP Error {status_code}: {description}")
    return jsonify({'error': description}), status_code

@app.errorhandler(500)
def internal_error(e):
    logger.exception('Error interno del servidor') 
    return jsonify({'error': 'Error interno del servidor. Por favor, revisa el log.'}), 500

# --- Lógica de IA (Streaming) ---
def stream_query_model(messages, model, temperature, max_tokens, timeout=60):
    if model not in app.config['SUPPORTED_MODELS']:
        yield f"data: {json.dumps({'error': 'Modelo no soportado.'})}\n\n"
        return
        
    headers = {
        "Authorization": f"Bearer {app.config['OPENROUTER_KEY']}", # 💡 Usar config
        "Content-Type": "application/json",
        "User-Agent": "TecSoftAI/3.0 - Production"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens, # 💡 Parámetro dinámico
        "temperature": temperature, # 💡 Parámetro dinámico
        "stream": True
    }

    try:
        logger.info(f"Solicitud STREAM: Modelo={model}, Temp={temperature}")
        response = requests.post(app.config['BASE_URL'], headers=headers, json=payload, stream=True, timeout=timeout)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data = line_str[6:]
                    if data.strip() == '[DONE]':
                        break
                    
                    try:
                        chunk = json.loads(data)
                        content_chunk = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                        
                        if content_chunk:
                            # 💡 Ceder el chunk envuelto en formato SSE (Server-Sent Events)
                            yield f"data: {json.dumps({'text': content_chunk})}\n\n"
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error decodificando chunk JSON: {e}")
                        continue
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error en la API (Stream): {str(e)}")
        # Enviar error en formato SSE
        yield f"data: {json.dumps({'error': f'Error en la API: {str(e)}'})}\n\n"
    except Exception as e:
        logger.error(f"Error desconocido en streaming: {str(e)}")
        yield f"data: {json.dumps({'error': f'Error interno: {str(e)}'})}\n\n"

# --- Blueprints (Rutas) ---
api_bp = Blueprint('api', __name__)

# --- Rutas de Auth ---
@api_bp.route('/register', methods=['POST'])
@rate_limited
def register():
    data = request.get_json()
    username = re.sub(r'\s+', '', data.get('username', '')) # 💡 Limpiar espacios
    email = data.get('email', '').strip()
    password = data.get('password', '')
    
    if not (username and email and password and '@' in email):
        abort(400, description='Usuario, email válido y contraseña son requeridos')
    
    password_hash = generate_password_hash(password)
    try:
        with get_db() as db:
            db.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)', (username, email, password_hash))
            db.commit()
        
        # Uso de threading para enviar email (no bloquea la solicitud HTTP)
        Thread(target=send_email, args=(email, "Bienvenido a TecSoft AI", "Tu cuenta ha sido creada exitosamente.")).start()
        
        return jsonify({'message': 'Usuario registrado exitosamente. Por favor, inicia sesión.'}), 201
    except sqlite3.IntegrityError:
        abort(400, description='Usuario o email ya existe')

@api_bp.route('/login', methods=['POST'])
@rate_limited
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    with get_db() as db:
        user = db.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,)).fetchone()
    
    if user and check_password_hash(user['password_hash'], password):
        token = generate_auth_token(user['id'])
        if token:
            return jsonify({'token': token}), 200
        abort(500, description='Error generando token')
    
    abort(401, description='Credenciales inválidas')

# --- Ruta de Chat (Máximo Nivel de Flexibilidad) ---
@api_bp.route('/text/stream', methods=['POST'])
@login_required
@rate_limited
def api_text_stream():
    data = request.get_json()
    if not data:
        abort(400, description='Datos JSON requeridos')
    
    user_id = g.user_id
    session_id = data.get('session_id') or generate_session_id()
    prompt = data.get('prompt', '').strip()
    
    # 💡 Parámetros de IA dinámicos (para flexibilidad OpenRouter)
    model_name = data.get('model', app.config['DEFAULT_MODEL'])
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 2048)
    
    if model_name not in app.config['SUPPORTED_MODELS']:
        abort(400, description=f'Modelo "{model_name}" no soportado.')
    
    if not prompt:
        abort(400, description='El campo "prompt" es requerido.')

    # Validación y Sanitización
    sanitized_prompt = re.sub(r'[^\w\s\.\,\!\?\-\$\$\$\$\{\}\:\;\'\"\_]', '', prompt).strip()
    if len(sanitized_prompt) > app.config['MAX_MESSAGE_LENGTH']:
        abort(400, description='Mensaje demasiado largo.')

    # Preparación de Historial
    save_session(session_id, user_id) 
    history = get_messages(session_id)
    
    # Validar longitud total
    total_length = sum(len(msg['content']) for msg in history) + len(sanitized_prompt)
    if total_length > app.config['MAX_HISTORY_LENGTH']:
        # 💡 En producción, se debería truncar el historial antiguo aquí
        abort(400, description=f'Historial demasiado largo (máx. {app.config["MAX_HISTORY_LENGTH"]} caracteres).')

    # Guardar mensaje de usuario ANTES de la llamada a la API (asíncrono)
    save_message_async(session_id, 'user', sanitized_prompt)
    
    # Construir lista final para la API
    messages = history + [{'role': 'user', 'content': sanitized_prompt}]
    
    # Llamar a la API con streaming (Server-Sent Events)
    response_generator = stream_query_model(messages, model_name, temperature, max_tokens)
    
    # 💡 Usar stream_with_context para que el generador tenga acceso a 'app.app_context'
    @stream_with_context
    def generate_stream():
        full_response = ""
        for chunk in response_generator:
            if chunk.startswith("data: "):
                try:
                    # Desempaquetar el JSON del chunk para revisar el contenido de texto
                    data = json.loads(chunk[6:].strip())
                    text_content = data.get('text', '')
                    error_content = data.get('error', '')
                    
                    if text_content:
                        full_response += text_content
                    
                    # Si hay error, detenemos el stream y no guardamos
                    if error_content:
                        logger.error(f"Stream terminó con error: {error_content}")
                        yield chunk # Rendir el error al cliente
                        return
                    
                except json.JSONDecodeError:
                    pass # Ignorar líneas que no son JSON válido
            
            yield chunk # Rendir el chunk raw (formato SSE)
            
        # 💡 Guardar respuesta completa DESPUÉS de que el stream termine (asíncrono)
        if full_response:
            save_message_async(session_id, 'assistant', full_response)
        
        # Enviar el mensaje final para que el cliente sepa que terminó
        yield f"data: {json.dumps({'session_id': session_id, 'end_stream': True})}\n\n"

    # Retornar respuesta en formato SSE
    return Response(generate_stream(), mimetype='text/event-stream')


# --- Ruta de Imagen/Multimodal (síncrona) ---
@api_bp.route('/vision', methods=['POST']) # 💡 Renombrado a 'vision'
@login_required
@rate_limited
def api_vision():
    # Código síncrono para manejar multimodalidad, usando query_model_no_stream si fuera necesario
    # o una implementación directa si la latencia es aceptable
    abort(501, description="Endpoint multimodal no implementado completamente en esta versión.")

# ... (Otras rutas de historial y modelos, usar `abort` en lugar de `handle_error`)

# Registrar blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# --- Servidor de Producción ---
def create_app():
    logger.info("Aplicación TecSoft AI V3.0 (PROD-READY) inicializada.")
    return app

if __name__ == '__main__':
    # 💡 Usar Waitress, un servidor WSGI de producción ligero y robusto
    app = create_app()
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Iniciando servidor Waitress en el puerto {port}...")
    # 🔒 Añadir HSTS (Strict-Transport-Security) en un entorno de producción con HTTPS
    # Aquí se simula, el despliegue real requeriría un proxy (Nginx) para gestionar HTTPS y HSTS.
    
    # Para el desarrollo local, aún podemos usar app.run
    if os.getenv("FLASK_ENV") == "development":
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        # Usar Waitress para simular entorno de producción
        serve(app, host='0.0.0.0', port=port)

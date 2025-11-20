from flask import Flask, request, jsonify, render_template_string, Response, Blueprint, g, session
import requests
import json
import os
import logging
from dotenv import load_dotenv
from functools import wraps
import time
from collections import defaultdict
import hashlib
import re
import sqlite3
from contextlib import contextmanager
import threading
import smtplib
from email.mime.text import MIMEText
import secrets
import cachetools
from cachetools import TTLCache

# Cargar variables de entorno desde un archivo .env si existe
load_dotenv()

# Inicializar la aplicación Flask con configuración
class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(16))
    DATABASE = os.getenv("DATABASE", "tecsoft_ai.db")
    MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", 5000))
    MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", 50000))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 10))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))
    SUPPORTED_MODELS = os.getenv("SUPPORTED_MODELS", "x-ai/grok-4.1-fast").split(",")
    CACHE_TTL = int(os.getenv("CACHE_TTL", 300))  # 5 minutos
    EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@tecsoft.ai")

app = Flask(__name__)
app.config.from_object(Config)

# Configurar logging avanzado para debugging y monitoreo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ⚠️ Cargar clave API desde variable de entorno con validación
API_KEY = os.getenv("OPENROUTER_KEY")
if not API_KEY:
    logger.error("La variable de entorno OPENROUTER_KEY no está configurada.")
    raise ValueError("La variable de entorno OPENROUTER_KEY no está configurada. Por favor, configúrala con tu clave API de OpenRouter.")

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Cache para respuestas de API
response_cache = TTLCache(maxsize=100, ttl=app.config['CACHE_TTL'])

# Estructuras para rate limiting simple (en producción, usar Redis o similar)
rate_limit_store = defaultdict(list)

# Base de datos SQLite para persistencia
@contextmanager
def get_db():
    db = sqlite3.connect(app.config['DATABASE'])
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()

def init_db():
    with get_db() as db:
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        db.commit()

# Inicializar DB al iniciar
init_db()

# Función para hash de contraseñas (simple, en producción usar bcrypt)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Función para verificar contraseña
def verify_password(password, hash):
    return hash_password(password) == hash

# Función para enviar email (simulado si no configurado)
def send_email(to, subject, body):
    if not app.config['EMAIL_ENABLED']:
        logger.info(f"Email simulado enviado a {to}: {subject}")
        return True
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = app.config['SMTP_USER']
        msg['To'] = to

        server = smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT'])
        server.starttls()
        server.login(app.config['SMTP_USER'], app.config['SMTP_PASS'])
        server.sendmail(app.config['SMTP_USER'], to, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        logger.error(f"Error enviando email: {str(e)}")
        return False

# Función para verificar rate limiting
def check_rate_limit(ip):
    now = time.time()
    rate_limit_store[ip] = [t for t in rate_limit_store[ip] if now - t < app.config['RATE_LIMIT_WINDOW']]
    if len(rate_limit_store[ip]) >= app.config['RATE_LIMIT_REQUESTS']:
        return False
    rate_limit_store[ip].append(now)
    return True

# Decorador para rate limiting
def rate_limited(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        if not check_rate_limit(ip):
            logger.warning(f"Rate limit exceeded for IP: {ip}")
            return jsonify({'error': 'Demasiadas solicitudes. Inténtalo más tarde.'}), 429
        return f(*args, **kwargs)
    return decorated_function

# Decorador para requerir autenticación
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not verify_token(token):
            return jsonify({'error': 'Autenticación requerida'}), 401
        g.user_id = get_user_from_token(token)
        return f(*args, **kwargs)
    return decorated_function

# Función para generar token (simple, en producción usar JWT)
def generate_token(user_id):
    return hashlib.sha256(f"{user_id}{app.config['SECRET_KEY']}{time.time()}".encode()).hexdigest()

# Función para verificar token
def verify_token(token):
    # En producción, decodificar JWT
    return True  # Simplificado

# Función para obtener user_id de token
def get_user_from_token(token):
    # Simplificado, en producción buscar en DB
    return 1

# Función para sanitizar entrada de texto
def sanitize_text(text):
    # Remover caracteres potencialmente peligrosos
    text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\[\]\{\}\:\;\'\"]', '', text)
    return text.strip()

# Función para validar URL de imagen
def validate_image_url(url):
    if not url.startswith(('http://', 'https://')):
        return False
    # Verificar longitud y formato básico
    if len(url) > 2000:
        return False
    # Podrías agregar más validaciones, como verificar si es una imagen real
    return True

# Función para generar un hash único para sesiones
def generate_session_id():
    return hashlib.md5(str(time.time()).encode()).hexdigest()

# Función para almacenar historial de chat en DB
def save_message(session_id, role, content):
    with get_db() as db:
        db.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)', (session_id, role, content))
        db.commit()

def get_messages(session_id):
    with get_db() as db:
        rows = db.execute('SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp', (session_id,)).fetchall()
        return [{'role': row['role'], 'content': row['content']} for row in rows]

# Función auxiliar para comunicarse con OpenRouter con mejoras y logging detallado
def query_model(model, messages, timeout=30):
    if model not in app.config['SUPPORTED_MODELS']:
        logger.error(f"Modelo no soportado: {model}")
        return "Error: Modelo no soportado."
    
    # Verificar cache
    cache_key = hashlib.md5(json.dumps(messages).encode()).hexdigest()
    if cache_key in response_cache:
        logger.info("Respuesta obtenida del cache.")
        return response_cache[cache_key]
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "TecSoftAI/1.0"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000,  # Limitar tokens para control de costos
        "temperature": 0.7
    }

    try:
        logger.info(f"Enviando solicitud a {model} con {len(messages)} mensajes.")
        response = requests.post(BASE_URL, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "Sin respuesta")
        logger.info(f"Respuesta exitosa de {model}: {content[:100]}...")
        # Guardar en cache
        response_cache[cache_key] = content
        return content
    except requests.exceptions.Timeout:
        logger.error("Tiempo de espera agotado en la API.")
        return "Error: Tiempo de espera agotado en la API."
    except requests.exceptions.RequestException as e:
        logger.error(f"Error en la API: {str(e)}")
        return f"Error en la API: {str(e)}"
    except json.JSONDecodeError:
        logger.error("Error al decodificar respuesta JSON de la API.")
        return "Error: Respuesta inválida de la API."

# Función para manejar errores y devolver JSON consistente
def handle_error(error_message, status_code=500):
    logger.error(error_message)
    return jsonify({'error': error_message}), status_code

# Manejadores de error globales para devolver JSON siempre
@app.errorhandler(404)
def not_found(error):
    return handle_error('Ruta no encontrada', 404)

@app.errorhandler(500)
def internal_error(error):
    return handle_error('Error interno del servidor', 500)

@app.errorhandler(429)
def too_many_requests(error):
    return handle_error('Demasiadas solicitudes. Inténtalo más tarde.', 429)

# Blueprints para organizar rutas
api_bp = Blueprint('api', __name__)

# Ruta principal que renderiza el HTML
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

# Endpoint para registro de usuario
@api_bp.route('/register', methods=['POST'])
@rate_limited
def register():
    try:
        data = request.get_json()
        username = sanitize_text(data.get('username', ''))
        email = sanitize_text(data.get('email', ''))
        password = data.get('password', '')
        
        if not username or not email or not password:
            return handle_error('Todos los campos son requeridos', 400)
        
        password_hash = hash_password(password)
        with get_db() as db:
            db.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)', (username, email, password_hash))
            db.commit()
        
        send_email(email, "Bienvenido a TecSoft AI", "Tu cuenta ha sido creada exitosamente.")
        return jsonify({'message': 'Usuario registrado exitosamente'})
    except sqlite3.IntegrityError:
        return handle_error('Usuario o email ya existe', 400)
    except Exception as e:
        logger.error(f"Error en registro: {str(e)}")
        return handle_error('Error interno del servidor', 500)

# Endpoint para login
@api_bp.route('/login', methods=['POST'])
@rate_limited
def login():
    try:
        data = request.get_json()
        username = sanitize_text(data.get('username', ''))
        password = data.get('password', '')
        
        with get_db() as db:
            user = db.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,)).fetchone()
        
        if user and verify_password(password, user['password_hash']):
            token = generate_token(user['id'])
            return jsonify({'token': token})
        return handle_error('Credenciales inválidas', 401)
    except Exception as e:
        logger.error(f"Error en login: {str(e)}")
        return handle_error('Error interno del servidor', 500)

# Endpoint para chat de texto con mejoras
@api_bp.route('/text', methods=['POST'])
@login_required
@rate_limited
def api_text():
    try:
        data = request.get_json()
        if not data:
            return handle_error('Datos JSON requeridos', 400)
        
        session_id = data.get('session_id', generate_session_id())
        messages = data.get('messages', [])
        
        if not messages or not isinstance(messages, list):
            return handle_error('Lista de mensajes requerida', 400)
        
        # Sanitizar y validar mensajes
        sanitized_messages = []
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                return handle_error('Formato de mensaje inválido', 400)
            sanitized_content = sanitize_text(msg['content'])
            if len(sanitized_content) > app.config['MAX_MESSAGE_LENGTH']:
                return handle_error(f'Mensaje demasiado largo (máx. {app.config["MAX_MESSAGE_LENGTH"]} caracteres)', 400)
            sanitized_messages.append({'role': msg['role'], 'content': sanitized_content})
        
        # Verificar longitud total del historial
        total_length = sum(len(msg['content']) for msg in sanitized_messages)
        if total_length > app.config['MAX_HISTORY_LENGTH']:
            return handle_error(f'Historial demasiado largo (máx. {app.config["MAX_HISTORY_LENGTH"]} caracteres)', 400)
        
        # Guardar mensajes en DB
        for msg in sanitized_messages:
            save_message(session_id, msg['role'], msg['content'])
        
        reply = query_model("x-ai/grok-4.1-fast", sanitized_messages)
        
        # Guardar respuesta
        save_message(session_id, 'assistant', reply)
        
        return jsonify({'reply': reply, 'session_id': session_id})
    except Exception as e:
        logger.error(f"Error en /api/text: {str(e)}")
        return handle_error('Error interno del servidor', 500)

# Endpoint para análisis de imagen con mejoras
@api_bp.route('/image', methods=['POST'])
@login_required
@rate_limited
def api_image():
    try:
        data = request.get_json()
        if not data:
            return handle_error('Datos JSON requeridos', 400)
        
        text = data.get('text', '').strip()
        image_url = data.get('image_url', '').strip()
        
        if not text or not image_url:
            return handle_error('Texto e imagen requeridos', 400)
        
        # Sanitizar y validar
        text = sanitize_text(text)
        if len(text) > app.config['MAX_MESSAGE_LENGTH']:
            return handle_error(f'Texto demasiado largo (máx. {app.config["MAX_MESSAGE_LENGTH"]} caracteres)', 400)
        
        if not validate_image_url(image_url):
            return handle_error('URL de imagen inválida', 400)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
        
        reply = query_model("x-ai/grok-4.1-fast", messages)
        return jsonify({'reply': reply})
    except Exception as e:
        logger.error(f"Error en /api/image: {str(e)}")
        return handle_error('Error interno del servidor', 500)

# Endpoint para obtener historial de chat
@api_bp.route('/history/<session_id>', methods=['GET'])
@login_required
@rate_limited
def get_history(session_id):
    try:
        history = get_messages(session_id)
        if not history:
            return handle_error('Sesión no encontrada', 404)
        return jsonify({'history': history})
    except Exception as e:
        logger.error(f"Error obteniendo historial: {str(e)}")
        return handle_error('Error interno del servidor', 500)

# Endpoint para limpiar historial de una sesión
@api_bp.route('/clear/<session_id>', methods=['DELETE'])
@login_required
@rate_limited
def clear_history(session_id):
    try:
        with get_db() as db:
            db.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
            db.commit()
        return jsonify({'message': 'Historial limpiado'})
    except Exception as e:
        logger.error(f"Error limpiando historial: {str(e)}")
        return handle_error('Error interno del servidor', 500)

# Endpoint para listar modelos soportados
@api_bp.route('/models', methods=['GET'])
@login_required
def list_models():
    return jsonify({'models': app.config['SUPPORTED_MODELS']})

# Endpoint para estadísticas (solo admin, simplificado)
@api_bp.route('/stats', methods=['GET'])
@login_required
def get_stats():
    try:
        with get_db() as db:
            user_count = db.execute('SELECT COUNT(*) FROM users').fetchone()[0]
            message_count = db.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
        return jsonify({'users': user_count, 'messages': message_count})
    except Exception as e:
        logger.error(f"Error obteniendo stats: {str(e)}")
        return handle_error('Error interno del servidor', 500)

# Endpoint para configuración de usuario
@api_bp.route('/user/config', methods=['GET', 'PUT'])
@login_required
def user_config():
    if request.method == 'GET':
        # Simplificado, devolver config básica
        return jsonify({'max_messages': app.config['MAX_MESSAGE_LENGTH']})
    elif request.method == 'PUT':
        # Simplificado, no cambiar nada
        return jsonify({'message': 'Configuración actualizada'})

# Registrar blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# Función para inicializar la app con configuraciones adicionales
def create_app():
    # Aquí podrías agregar más inicializaciones, como conectar a DB externa
    logger.info("Aplicación TecSoft AI inicializada.")
    return app

# Ejecutar la aplicación si se llama directamente
if __name__ == '__main__':
    app = create_app()
    # En producción, usar un servidor WSGI como Gunicorn
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv("PORT", 5000)))

from flask import Flask, request, jsonify, render_template_string, session, send_from_directory
import requests
import json
import os
import logging
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import time
import re
from typing import List, Dict, Any, Optional
import markdown

# --- 1. CONFIGURACIÓN INICIAL Y AMBIENTE ---

# Cargar variables de entorno
load_dotenv()

# Inicialización de Flask
app = Flask(__name__, static_folder='static')

# Configurar clave secreta para sesiones
app.secret_key = os.getenv("FLASK_SECRET_KEY", "SUPER_SECRETO_DEBES_CAMBIAR_EN_PROD_1234567890")

# Habilitar CORS
CORS(app, supports_credentials=True, origins=["*"])

# Configurar logging avanzado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurar rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "30 per hour", "15 per minute"],
    storage_uri="memory://",
    headers_enabled=True
)

# --- 2. CONSTANTES DE LA API Y MODELOS ---

API_KEY = os.getenv("OPENROUTER_KEY")
if not API_KEY:
    logger.error("La variable de entorno OPENROUTER_KEY no está configurada.")
    raise ValueError("La variable de entorno OPENROUTER_KEY no está configurada. Por favor, configúrala.")

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Modelos configurables: Usando x-ai/grok-4.1-fast para ambos
TEXT_MODEL = "x-ai/grok-4.1-fast"
IMAGE_MODEL = "x-ai/grok-4.1-fast"

# Configuración de sistema para dar contexto y personalidad
SYSTEM_MESSAGE = (
    "Eres **TecSoft AI**, un asistente de IA avanzado para proyectos universitarios. "
    "Tu objetivo es ser un experto en programación, ciencia de datos, y tecnología, respondiendo de "
    "manera precisa, concisa, y utilizando el formato **Markdown** (negritas, listas, bloques de código) "
    "para una mejor legibilidad. Mantén un tono profesional y futurista."
)

# --- 3. FUNCIONES AUXILIARES ROBUSTAS ---

def query_model(model: str, messages: List[Dict[str, Any]], max_retries: int = 3) -> str:
    """Función robusta para comunicarse con OpenRouter con retries y backoff exponencial."""
    
    # CRÍTICO: El system message debe ir al inicio, independientemente de lo que envíe el front.
    full_messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + messages
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/tu_usuario/tu_repo_proyecto_uni",
        "X-Title": "TecSoft AI - Proyecto Universitario"
    }
    
    payload = {
        "model": model,
        "messages": full_messages,
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=45)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                content = result["choices"][0]["message"].get("content", "Sin respuesta útil del modelo.")
                logger.info(f"Respuesta exitosa del modelo {model} en intento {attempt + 1}. Uso: {result.get('usage', {})}")
                return content
            else:
                error_msg = f"Estructura de respuesta inválida de OpenRouter: {json.dumps(result)}"
                logger.error(error_msg)
                # No reintentar con respuesta 200 malformada
                return "Error interno del modelo: Respuesta malformada."
                
        except requests.exceptions.Timeout:
            logger.warning(f"Tiempo de espera agotado en intento {attempt + 1}.")
            if attempt == max_retries - 1:
                return "Error: Tiempo de espera agotado después de varios intentos."
            time.sleep(2 ** attempt + 1)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en la API en intento {attempt + 1}: {str(e)}. Código: {response.status_code if 'response' in locals() else 'N/A'}")
            if attempt == max_retries - 1:
                try:
                    error_json = response.json()
                    error_detail = error_json.get('error', {}).get('message', str(e))
                except:
                    error_detail = str(e)
                return f"Error en la API: {error_detail}"
            time.sleep(2 ** attempt + 1)
            
    return "Error fatal: No se pudo obtener respuesta después de varios intentos."


def get_chat_history_from_session() -> List[Dict[str, str]]:
    """Obtiene el historial de chat de la sesión, asegurando un formato válido."""
    history = session.get('chat_history')
    if history is None or not isinstance(history, list):
        session['chat_history'] = []
        return []
    return [msg for msg in history if isinstance(msg, dict) and 'role' in msg and 'content' in msg]

def update_chat_history_in_session(role: str, content: str):
    """Añade un mensaje al historial de sesión y lo limita."""
    history = get_chat_history_from_session()
    history.append({"role": role, "content": content})
    # Limitar el historial (ej. conservar los 10 últimos mensajes)
    session['chat_history'] = history[-10:]
    logger.debug(f"Historial actualizado. Total: {len(session['chat_history'])} mensajes.")

# --- 4. MANEJADORES DE ERROR Y RUTAS DE FLASK ---

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': f'Solicitud Inválida (400): {error.description}'}), 400

@app.errorhandler(429)
def rate_limit_exceeded(e):
    logger.warning(f"Rate limit excedido para IP: {get_remote_address()}")
    return jsonify({'error': '❌ Límite de solicitudes excedido. Intenta de nuevo en unos momentos.'}), 429

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Ruta no encontrada'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.exception("Error 500 capturado.")
    return jsonify({'error': 'Error interno del servidor. Revisa los logs.'}), 500

@app.route('/')
def home():
    """Ruta principal que sirve la interfaz web."""
    return render_template_string(HTML_TEMPLATE_MEJORADO)

# --- 5. RUTAS API PARA LA LÓGICA DE NEGOCIO ---

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Ruta para resetear el historial de chat de la sesión."""
    session['chat_history'] = []
    return jsonify({'message': 'Historial de chat reseteado con éxito.'}), 200

@app.route('/api/text', methods=['POST'])
@limiter.limit("10 per minute")
def api_text():
    """Maneja el chat de solo texto con historial de sesión."""
    try:
        data = request.get_json()
        user_message_content: Optional[str] = data.get('message')
        
        # Validación estricta del input
        if not user_message_content or not isinstance(user_message_content, str) or len(user_message_content.strip()) < 1:
            return jsonify({'error': 'Mensaje de texto es obligatorio.'}), 400
            
        user_message_content = user_message_content.strip()

        # 1. Agregar el nuevo mensaje del usuario al historial de sesión
        update_chat_history_in_session("user", user_message_content)
        
        # 2. Obtener el historial COMPLETO y actualizado para enviar al modelo
        updated_messages = get_chat_history_from_session()
        
        # 3. Llamar al modelo con el historial COMPLETO
        reply_content = query_model(TEXT_MODEL, updated_messages)

        # 4. Procesar el markdown A HTML en el backend
        html_reply_content = markdown.markdown(reply_content)

        # 5. Agregar la respuesta del asistente (solo si no es un error de API)
        if not (reply_content.startswith("Error:") or reply_content.startswith("Error interno del modelo")):
            update_chat_history_in_session("assistant", reply_content)

        # 6. Respuesta final (Enviar el HTML)
        return jsonify({'reply': html_reply_content}), 200

    except Exception as e:
        logger.exception(f"Error en /api/text: {e}")
        return jsonify({'error': f'Error en el procesamiento de la solicitud: {str(e)}'}), 500


@app.route('/api/image', methods=['POST'])
@limiter.limit("5 per hour")
def api_image():
    """Maneja el análisis de texto + imagen."""
    try:
        data = request.get_json()
        image_url: Optional[str] = data.get('image_url')
        text_prompt: Optional[str] = data.get('text')

        # Validación estricta
        if not image_url or not image_url.strip():
            return jsonify({'error': 'URL de imagen es obligatoria.'}), 400
        if not text_prompt or not text_prompt.strip():
            return jsonify({'error': 'El prompt de texto es obligatorio para el análisis.'}), 400
            
        image_url = image_url.strip()
        text_prompt = text_prompt.strip()

        # Validar formato URL básico
        if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', image_url):
            return jsonify({'error': 'Formato de URL inválido o inseguro.'}), 400

        # La estructura de mensajes para multimodal en OpenRouter
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        
        # Llamar al modelo multimodal
        reply_content = query_model(IMAGE_MODEL, messages)
        
        # Procesar el markdown A HTML en el backend
        html_reply_content = markdown.markdown(reply_content)

        # Respuesta final (Enviar el HTML)
        return jsonify({'reply': html_reply_content}), 200

    except Exception as e:
        logger.exception(f"Error en /api/image: {e}")
        return jsonify({'error': f'Error en el procesamiento de la solicitud: {str(e)}'}), 500


# --- 6. TEMPLATE HTML MEJORADO ---
# La línea "---" que causó el SyntaxError ha sido eliminada o comentada.

HTML_TEMPLATE_MEJORADO = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TecSoft AI - Mejorado</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600&display=swap" rel="stylesheet">
    <style>
        /* BASE Y FONDO */
        body {
            font-family: 'Rajdhani', sans-serif;
            /* Degradado de fondo con centro más claro */
            background: radial-gradient(circle at center, #0a0a1a, #000010 80%);
            color: #ffffff;
            margin: 0;
            padding: 0;
            overflow-x: hidden;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            animation: fadeIn 1.5s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* TÍTULO Y GLOW */
        h1 {
            margin-top: 40px;
            text-align: center;
            font-size: 3.5em; /* Aumento de tamaño */
            color: #00ffff;
            text-shadow: 0 0 40px #00ffff, 0 0 80px #ff00ff; /* Más intensidad */
            animation: glow 3s infinite alternate;
            letter-spacing: 3px; /* Más espaciado */
            font-family: 'Orbitron', sans-serif; /* Título en fuente más impactante */
        }

        @keyframes glow {
            from { text-shadow: 0 0 15px #00ffff, 0 0 30px #ff00ff; }
            to { text-shadow: 0 0 50px #00ffff, 0 0 100px #ff00ff; }
        }

        /* SECCIONES (Contenedores) */
        .section {
            width: 90%;
            max-width: 800px; /* Un poco más ancho */
            background: rgba(0, 0, 30, 0.95); /* Fondo más oscuro */
            border: 3px solid #00ffff; /* Borde más grueso */
            border-radius: 20px; /* Bordes más redondeados */
            padding: 30px;
            margin: 25px 0;
            box-shadow: 0 0 50px rgba(0, 255, 255, 0.5);
            backdrop-filter: blur(8px); /* Más blur */
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }

        .section:hover {
            transform: scale(1.01); /* Menor escala al hacer hover para menos distracción */
            box-shadow: 0 0 80px rgba(255, 0, 255, 0.8), 0 0 10px rgba(0, 255, 255, 0.8);
        }

        h2 {
            color: #ff00ff;
            text-shadow: 0 0 15px #ff00ff;
            margin-bottom: 20px;
            font-size: 1.8em; /* Un poco más grande */
            border-bottom: 2px dashed rgba(255, 0, 255, 0.3);
            padding-bottom: 10px;
        }

        /* INPUTS Y TEXTAREAS */
        textarea, input {
            width: 100%;
            padding: 15px;
            margin: 10px 0;
            border: 2px solid #00ffff;
            border-radius: 12px;
            background: rgba(255,255,255,0.05); /* Fondo más sutil */
            color: #fff;
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.2em;
            outline: none;
            transition: border-color 0.3s, box-shadow 0.3s;
            box-sizing: border-box; /* Asegura que padding no afecte el ancho total */
        }

        textarea:focus, input:focus {
            border-color: #ff00ff;
            box-shadow: 0 0 20px #ff00ff;
            background: rgba(255,255,255,0.1);
        }
        
        /* CONTENEDOR DE INPUT+BOTÓN */
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 10px;
        }
        
        /* BOTÓN */
        button {
            padding: 15px 30px;
            background: linear-gradient(45deg, #00ffff, #ff00ff);
            color: #111; /* Color más oscuro para mejor contraste */
            border: 0;
            border-radius: 12px;
            cursor: pointer;
            font-weight: bold;
            font-size: 1.2em;
            transition: 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        button:hover:not(:disabled) {
            background: linear-gradient(45deg, #ff00ff, #00ffff);
            box-shadow: 0 0 30px #ff00ff, 0 0 15px #00ffff;
            transform: scale(1.05);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: #444;
            transform: none;
            box-shadow: none;
        }
        
        /* CHAT Y MENSAJES */
        .chat-container {
            margin-top: 20px;
            height: 400px; /* Altura fija para el contenedor de chat */
            overflow-y: auto;
            background: rgba(0, 255, 255, 0.08); /* Fondo más claro para visibilidad */
            border-radius: 15px;
            border: 1px solid #00ffff;
            padding: 20px;
            box-shadow: inset 0 0 15px rgba(0,255,255,0.3);
            display: flex;
            flex-direction: column;
        }
        /* Estilo de la barra de desplazamiento */
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        .chat-container::-webkit-scrollbar-thumb {
            background: linear-gradient(to bottom, #00ffff, #ff00ff);
            border-radius: 10px;
        }
        .chat-container::-webkit-scrollbar-track {
            background: #0a0a1a;
        }

        .message {
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 15px;
            white-space: pre-wrap; /* Mantiene el formato de espacios y saltos de línea */
            font-size: 1.1em;
            max-width: 90%;
            line-height: 1.5;
            transition: opacity 0.5s ease-in-out;
            text-align: left; /* Asegura que el contenido HTML interno se alinee correctamente */
        }
        
        /* Estilos de código y markdown dentro del chat (¡Mejora clave!) */
        .message pre {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 10px;
            border-radius: 8px;
            overflow-x: auto;
            margin-top: 10px;
        }
        .message code {
            font-family: 'Consolas', 'Courier New', monospace;
            color: #00ffaa; /* Color de código */
            font-size: 0.95em;
        }

        .user-message {
            background: rgba(0, 255, 255, 0.15);
            align-self: flex-end; /* Alineado a la derecha */
            color: #00ffff;
            border-right: 4px solid #00ffff;
            border-bottom-right-radius: 0;
        }

        .assistant-message {
            background: rgba(255, 0, 255, 0.15);
            align-self: flex-start; /* Alineado a la izquierda */
            color: #ff00ff;
            border-left: 4px solid #ff00ff;
            border-bottom-left-radius: 0;
        }
        
        .loading {
            color: #00ffaa; /* Nuevo color para loading */
            font-style: italic;
            text-shadow: 0 0 10px #00ffaa;
            animation: pulse 1.5s infinite alternate;
        }

        @keyframes pulse {
            from { opacity: 0.7; }
            to { opacity: 1; }
        }

        .error {
            color: #ff4444;
            text-shadow: 0 0 10px #ff4444;
            font-weight: bold;
        }

        /* IMAGEN MULTIMODAL */
        .image-preview {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            margin-top: 15px;
            border: 2px solid #00ffff;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        }
        
        /* FOOTER */
        footer {
            margin: 50px 0 20px 0; /* Más margen abajo */
            color: #aaa;
            font-size: 1em;
            text-align: center;
            padding: 10px;
            border-top: 1px dashed rgba(255, 255, 255, 0.1);
        }

        /* MEDIA QUERIES (Responsivo) */
        @media (max-width: 768px) {
            h1 { font-size: 2.8em; margin-top: 30px; }
            .section { padding: 20px; margin: 15px 0; }
            .chat-container { height: 300px; }
            button { font-size: 1.1em; padding: 12px 20px; }
        }

        @media (max-width: 480px) {
            h1 { font-size: 2em; letter-spacing: 1px; }
            .section { border-radius: 15px; padding: 15px; }
            .user-message, .assistant-message { max-width: 100%; }
        }

        /* PARTÍCULAS (EFECTO ESPACIAL) */
        canvas#particles {
            position: fixed;
            top: 0; left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: transparent;
        }
    </style>
</head>
<body>
    <canvas id="particles"></canvas>

    <h1>🚀 TecSoft AI</h1>

    <div class="section">
        <h2>🧠 Chat de Texto - Conversación</h2>
        <div class="chat-container" id="textChat"></div>
        <div class="input-group">
            <textarea id="textInput" rows="4" placeholder="Escribe tu mensaje aquí..."></textarea>
            <button id="textButton" onclick="sendText()">Enviar Mensaje</button>
        </div>
    </div>

    <div class="section">
        <h2>🖼️ Imagen + Texto - Multimodal</h2>
        <div class="input-group">
            <input type="url" id="imageUrl" placeholder="URL de la imagen (ej: https://ejemplo.com/foto.jpg)" oninput="updateImagePreview()">
            <img id="imagePreview" class="image-preview" src="" style="display: none;" alt="Previsualización de imagen">
            <textarea id="imageText" rows="4" placeholder="¿Qué deseas saber o hacer con la imagen? (ej: Descríbela)"></textarea>
            <button id="imageButton" onclick="sendImage()">Analizar Imagen</button>
        </div>
        <div class="chat-container" id="imageChatResponse">
              <div id="imageResponse" class="message assistant-message">Esperando análisis...</div>
        </div>
    </div>

    <footer>✨ Desarrollado por <b>TecSoft AI</b> | Con tecnología futurista ⚙️</footer>

    <script>
        // --- Lógica de Partículas (Efecto Espacial) ---
        const canvas = document.getElementById('particles');
        const ctx = canvas.getContext('2d');
        let particles = [];

        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        // Generar partículas
        for (let i = 0; i < 50; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                size: Math.random() * 2 + 1,
                speedX: (Math.random() - 0.5) * 0.5,
                speedY: (Math.random() - 0.5) * 0.5
            });
        }

        function drawParticles() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            // Configuración de glow y color para el efecto espacial
            ctx.shadowColor = '#00ffff';
            ctx.shadowBlur = 10;
            ctx.fillStyle = '#00ffff'; 
            
            particles.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
                p.x += p.speedX;
                p.y += p.speedY;
                // Rebote en bordes
                if (p.x < 0 || p.x > canvas.width) p.speedX *= -1;
                if (p.y < 0 || p.y > canvas.height) p.speedY *= -1;
            });
            requestAnimationFrame(drawParticles);
        }
        drawParticles();

        // --- Funciones de Utilidad de Chat ---
        
        function addMessage(role, content_html, is_markdown=false) {
            const chatContainer = document.getElementById('textChat');
            const messageDiv = document.createElement('div');
            messageDiv.className = role === 'user' ? 'message user-message' : 'message assistant-message';
            
            // Si es un mensaje del usuario (texto plano) o si no es HTML, usamos textContent
            if (role === 'user' || !is_markdown) {
                messageDiv.textContent = content_html;
            } else {
                // Si es la respuesta del asistente (ya convertida a HTML), usamos innerHTML
                messageDiv.innerHTML = content_html;
            }
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // --- Chat de Texto (sendText) ---
        async function sendText() {
            const textInput = document.getElementById('textInput');
            const text = textInput.value.trim();
            const button = document.getElementById('textButton');
            
            if (!text) {
                textInput.focus();
                return; 
            }
            
            button.disabled = true;
            textInput.value = '';

            // Agregamos el mensaje del usuario como texto plano (no HTML)
            addMessage('user', text, false); 

            // Agregar mensaje de loading
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message assistant-message loading';
            loadingDiv.textContent = '⏳ Procesando...';
            document.getElementById('textChat').appendChild(loadingDiv);
            document.getElementById('textChat').scrollTop = document.getElementById('textChat').scrollHeight;

            try {
                const res = await fetch('/api/text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text }) // Solo enviamos el mensaje del usuario
                });
                
                // Remover loading
                const chatContainer = document.getElementById('textChat');
                if (chatContainer.contains(loadingDiv)) chatContainer.removeChild(loadingDiv);
                
                let data;
                if (res.ok) {
                    data = await res.json();
                    // data.reply ya es HTML renderizado por el servidor
                    addMessage('assistant', data.reply, true); 
                } else {
                    try {
                        data = await res.json();
                        addMessage('assistant', `❌ Error (${res.status}): ${data.error || 'Error desconocido'}`, false);
                    } catch (e) {
                        addMessage('assistant', `❌ Error en la respuesta del servidor (${res.status}).`, false);
                    }
                }
            } catch (e) {
                // Remover loading en caso de error de red
                const chatContainer = document.getElementById('textChat');
                if (chatContainer.contains(loadingDiv)) chatContainer.removeChild(loadingDiv);
                addMessage('assistant', `⚠️ Error de red: ${e.message}`, false);
            } finally {
                button.disabled = false;
                textInput.focus();
            }
        }
        
        // Permite enviar al presionar Enter en el textarea
        document.getElementById('textInput').addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendText();
            }
        });

        // --- Chat Multimodal (sendImage) ---
        
        // Función para previsualizar la imagen
        function updateImagePreview() {
            const imageUrl = document.getElementById('imageUrl').value.trim();
            const imgElement = document.getElementById('imagePreview');
            
            if (imageUrl) {
                imgElement.src = imageUrl;
                imgElement.style.display = 'block';
                // Opcional: Manejar error de carga de imagen
                imgElement.onerror = () => { imgElement.style.display = 'none'; };
            } else {
                imgElement.style.display = 'none';
                imgElement.src = '';
            }
        }

        async function sendImage() {
            const image = document.getElementById('imageUrl').value.trim();
            const text = document.getElementById('imageText').value.trim();
            const outputDiv = document.getElementById('imageResponse');
            const button = document.getElementById('imageButton');
            
            if (!image || !text) {
                alert("Proporciona una URL de imagen válida y la pregunta a analizar.");
                return;
            }
            
            button.disabled = true;
            outputDiv.className = 'message assistant-message loading';
            outputDiv.textContent = '🖼️ Analizando imagen...';
            
            try {
                const res = await fetch('/api/image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, image_url: image })
                });
                
                let data;
                outputDiv.className = 'message assistant-message'; // Cambiar clase de vuelta
                
                if (res.ok) {
                    data = await res.json();
                    // data.reply ya es HTML renderizado por el servidor
                    outputDiv.innerHTML = data.reply; 
                } else {
                    try {
                        data = await res.json();
                        outputDiv.className += ' error';
                        outputDiv.innerHTML = `❌ Error (${res.status}): ${data.error || "Error desconocido"}`;
                    } catch (e) {
                        outputDiv.className += ' error';
                        outputDiv.innerHTML = "❌ Error al comunicarse con el servidor de IA.";
                    }
                }
            } catch (e) {
                outputDiv.className = 'message assistant-message error';
                outputDiv.innerHTML = `⚠️ Error de conexión: ${e.message}`;
            } finally {
                button.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

---

## 💡 Próximo Paso

**Reemplaza** el contenido completo de tu archivo `app.py` con este código.

**Asegúrate de que no haya líneas que contengan solo `---` fuera de las cadenas de texto (`"""..."""`) o comentarios (`#`)** en tu código de Python. Esto debería resolver tu `SyntaxError` y permitir que Gunicorn inicie la aplicación.

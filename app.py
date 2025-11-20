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
app = Flask(__name__, static_folder='static') # Habilitar carpeta 'static' (opcional, pero buena práctica)

# Configurar clave secreta para sesiones (CRÍTICO para seguridad)
# Usa un secreto largo y complejo en .env
app.secret_key = os.getenv("FLASK_SECRET_KEY", "SUPER_SECRETO_DEBES_CAMBIAR_EN_PROD_1234567890")

# Habilitar CORS para solicitudes desde el frontend (Configuración más segura)
# Para un proyecto universitario, 'origins' puede ser '*' o la URL específica del frontend.
CORS(app, supports_credentials=True, origins=["*"])

# Configurar logging avanzado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", mode='a'), # Modo append
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurar rate limiting para evitar abuso (CRÍTICO)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "30 per hour", "15 per minute"],
    storage_uri="memory://", # Usar memoria para simplicidad, o Redis/Memcached en producción
    headers_enabled=True # Considerar headers X-Forwarded-For si se usa proxy
)

# --- 2. CONSTANTES DE LA API Y MODELOS ---

API_KEY = os.getenv("OPENROUTER_KEY")
if not API_KEY:
    logger.error("La variable de entorno OPENROUTER_KEY no está configurada.")
    raise ValueError("La variable de entorno OPENROUTER_KEY no está configurada. Por favor, configúrala.")

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Modelos configurables
TEXT_MODEL = "kwaipilot/kat-coder-pro:free"  # Modelo para texto (Mantener el original)
IMAGE_MODEL = "x-ai/grok-4.1-fast"  # Modelo para imagen + texto (Mantener el original)

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
    
    # 1. Agregar el System Message al inicio del historial (si no está)
    # Esto asegura que el modelo mantenga el contexto y la personalidad.
    full_messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + messages
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/tu_usuario/tu_repo_proyecto_uni", # BUENA PRÁCTICA DE OPENROUTER
        "X-Title": "TecSoft AI - Proyecto Universitario"
    }
    
    # payload: se ajusta max_tokens y se añade temperature para control creativo
    payload = {
        "model": model,
        "messages": full_messages,
        "max_tokens": 2048, # Aumentado para respuestas universitarias detalladas
        "temperature": 0.7,  # Un poco creativo
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=45) # Aumentado timeout
            response.raise_for_status()
            result = response.json()
            
            # Validación de la respuesta de la API
            if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                content = result["choices"][0]["message"].get("content", "Sin respuesta útil del modelo.")
                logger.info(f"Respuesta exitosa del modelo {model} en intento {attempt + 1}. Uso: {result.get('usage', {})}")
                return content
            else:
                # La API respondió 200, pero la estructura es incorrecta (Error de OpenRouter)
                error_msg = f"Estructura de respuesta inválida de OpenRouter: {json.dumps(result)}"
                logger.error(error_msg)
                return "Error interno del modelo: Respuesta malformada."
                
        except requests.exceptions.Timeout:
            logger.warning(f"Tiempo de espera agotado en intento {attempt + 1}.")
            if attempt == max_retries - 1:
                return "Error: Tiempo de espera agotado después de varios intentos."
            time.sleep(2 ** attempt + 1) # Exponential backoff con jitter
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en la API en intento {attempt + 1}: {str(e)}. Código: {response.status_code if 'response' in locals() else 'N/A'}")
            if attempt == max_retries - 1:
                # Intentar parsear el error si es JSON (ej. Rate Limit de OpenRouter)
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
        # Inicializar o limpiar historial corrupto
        session['chat_history'] = []
        return []
    # Validación simple de formato (debe ser [{role: str, content: str}, ...])
    return [msg for msg in history if isinstance(msg, dict) and 'role' in msg and 'content' in msg]

def update_chat_history_in_session(role: str, content: str):
    """Añade un mensaje al historial de sesión."""
    history = get_chat_history_from_session()
    history.append({"role": role, "content": content})
    # Limitar el historial para evitar payloads gigantes y costos excesivos
    # Conserva los 10 últimos mensajes (5 pares)
    session['chat_history'] = history[-10:]
    logger.debug(f"Historial actualizado. Total: {len(session['chat_history'])} mensajes.")

# --- 4. MANEJADORES DE ERROR Y RUTAS DE FLASK ---

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': f'Solicitud Inválida (400): {error.description}'}), 400

@app.errorhandler(429)
def rate_limit_exceeded(e):
    # Error específico de Rate Limiter
    logger.warning(f"Rate limit excedido para IP: {get_remote_address()}")
    return jsonify({'error': '❌ Límite de solicitudes excedido. Intenta de nuevo en unos momentos.'}), 429

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Ruta no encontrada'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.exception("Error 500 capturado.") # Captura la traza completa
    return jsonify({'error': 'Error interno del servidor. Revisa los logs.'}), 500

@app.route('/')
def home():
    """Ruta principal que sirve la interfaz web."""
    # Usar el template mejorado con manejo de estado de sesión
    return render_template_string(HTML_TEMPLATE_MEJORADO)

# --- 5. RUTAS API PARA LA LÓGICA DE NEGOCIO ---

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Ruta para resetear el historial de chat de la sesión."""
    session['chat_history'] = []
    return jsonify({'message': 'Historial de chat reseteado con éxito.'}), 200

@app.route('/api/text', methods=['POST'])
@limiter.limit("10 per minute") # Límite estricto para chat
def api_text():
    """Maneja el chat de solo texto con historial de sesión."""
    try:
        data = request.get_json()
        user_message_content: Optional[str] = data.get('message')
        
        # Validación estricta del input
        if not user_message_content or not isinstance(user_message_content, str) or len(user_message_content.strip()) < 1:
            return jsonify({'error': 'Mensaje de texto es obligatorio.'}), 400
            
        user_message_content = user_message_content.strip()

        # 1. Obtener el historial (sin el system message)
        messages = get_chat_history_from_session()
        
        # 2. Agregar el nuevo mensaje del usuario
        update_chat_history_in_session("user", user_message_content)

        # 3. Llamar al modelo
        reply_content = query_model(TEXT_MODEL, messages + [{"role": "user", "content": user_message_content}])

        # 4. Agregar la respuesta del asistente (solo si no es un error de API)
        if not (reply_content.startswith("Error:") or reply_content.startswith("Error interno del modelo")):
            update_chat_history_in_session("assistant", reply_content)

        # 5. Respuesta final
        return jsonify({'reply': reply_content}), 200

    except Exception as e:
        logger.exception(f"Error en /api/text: {e}")
        return jsonify({'error': f'Error en el procesamiento de la solicitud: {str(e)}'}), 500


@app.route('/api/image', methods=['POST'])
@limiter.limit("5 per hour") # Límite más estricto para análisis de imagen/multimodal (más costoso)
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

        # La estructura de mensajes para multimodal en OpenRouter debe ser:
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

        # Respuesta final (el historial de multimodal no se persiste por simplicidad,
        # pero se podría hacer si el modelo lo permite y es necesario)
        return jsonify({'reply': reply_content}), 200

    except Exception as e:
        logger.exception(f"Error en /api/image: {e}")
        return jsonify({'error': f'Error en el procesamiento de la solicitud: {str(e)}'}), 500


# --- 6. TEMPLATE HTML MEJORADO (A LA MEDIDA DEL PROYECTO) ---

# Nota: El HTML_TEMPLATE_MEJORADO reemplaza al original.

HTML_TEMPLATE_MEJORADO = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TecSoft AI | Asistente de Proyecto</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600&display=swap" rel="stylesheet">
    <style>
        /* Estilos Mejorados (Variables CSS y diseño futurista/cyberpunk) */
        :root {
            --primary-color: #00ff7f; /* Neón verde */
            --secondary-color: #ff33cc; /* Neón magenta */
            --bg-color: #00000a; /* Fondo oscuro casi negro */
            --text-color: #e6e6e6;
            --code-bg: #1a1a33;
            --shadow-glow: 0 0 10px rgba(0, 255, 127, 0.6);
            --error-color: #ff4444;
        }

        body {
            font-family: 'Rajdhani', sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            animation: backgroundFade 5s ease-in-out;
        }

        @keyframes backgroundFade { from { opacity: 0; } to { opacity: 1; } }

        h1 {
            margin-top: 40px;
            font-family: 'Orbitron', sans-serif;
            color: var(--primary-color);
            text-shadow: var(--shadow-glow), 0 0 20px var(--secondary-color);
            animation: neonGlow 1.5s infinite alternate;
        }

        @keyframes neonGlow {
            to { text-shadow: 0 0 20px var(--primary-color), 0 0 40px var(--secondary-color); }
        }

        .section {
            width: 90%;
            max-width: 800px;
            background: rgba(10, 10, 30, 0.95);
            border: 1px solid var(--primary-color);
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 0 15px rgba(0, 255, 127, 0.3);
        }

        h2 { color: var(--secondary-color); text-shadow: 0 0 5px var(--secondary-color); border-bottom: 1px dashed var(--secondary-color); padding-bottom: 5px; }

        textarea, input[type="url"], input[type="text"] {
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            border: 1px solid var(--primary-color);
            border-radius: 8px;
            background: var(--code-bg);
            color: var(--text-color);
            box-shadow: inset 0 0 5px rgba(0, 255, 127, 0.3);
        }

        textarea:focus, input:focus { border-color: var(--secondary-color); box-shadow: 0 0 10px var(--secondary-color); }

        button {
            padding: 10px 20px;
            background: var(--primary-color);
            color: #000;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 700;
            transition: 0.3s;
            margin-right: 10px;
        }

        button:hover { background: var(--secondary-color); box-shadow: 0 0 15px var(--secondary-color); transform: translateY(-2px); }
        button:disabled { opacity: 0.4; cursor: not-allowed; }

        .chat-container {
            max-height: 450px;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 8px;
            border: 1px dashed var(--primary-color);
            padding: 15px;
            margin-bottom: 20px;
        }

        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
            white-space: pre-wrap;
            font-size: 1.05em;
            overflow-wrap: break-word;
        }

        .user-message {
            background: rgba(0, 255, 127, 0.1);
            text-align: right;
            color: var(--primary-color);
            border-left: 3px solid var(--primary-color);
        }

        .assistant-message {
            background: rgba(255, 51, 204, 0.1);
            color: var(--secondary-color);
            border-right: 3px solid var(--secondary-color);
        }

        .assistant-message p, .assistant-message ul, .assistant-message ol, .assistant-message pre {
            margin: 0 0 10px 0;
        }

        /* Estilos para el markdown renderizado */
        .assistant-message pre {
            background: var(--code-bg);
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            border: 1px dashed var(--primary-color);
            color: #ffffff;
        }
        .assistant-message code {
            background: rgba(255, 51, 204, 0.2);
            padding: 2px 4px;
            border-radius: 3px;
        }

        .loading { color: var(--primary-color); font-style: italic; }
        .error { color: var(--error-color); font-weight: bold; }
        .success { color: var(--primary-color); }
        
        .multimodal-preview {
            max-width: 100%;
            height: auto;
            border: 2px solid var(--secondary-color);
            border-radius: 5px;
            margin-top: 10px;
        }

        footer { margin-top: 50px; margin-bottom: 20px; color: #555; font-size: 0.9em; text-align: center; }

        /* Pequeña animación de fondo (simplificada sin canvas) */
        .background-line {
            position: absolute;
            height: 100vh;
            width: 1px;
            background: linear-gradient(to bottom, var(--primary-color), var(--secondary-color));
            opacity: 0.1;
            z-index: -1;
            animation: scanLine 10s linear infinite;
        }

        .line-1 { left: 10%; animation-delay: 0s; }
        .line-2 { left: 30%; animation-delay: 3s; }
        .line-3 { left: 70%; animation-delay: 6s; }
        
        @keyframes scanLine {
            0% { opacity: 0.1; transform: scaleY(0.1); }
            50% { opacity: 0.4; transform: scaleY(1); }
            100% { opacity: 0.1; transform: scaleY(0.1); }
        }

    </style>
</head>
<body>
    <div class="background-line line-1"></div>
    <div class="background-line line-2"></div>
    <div class="background-line line-3"></div>

    <h1>🚀 TecSoft AI</h1>
    
    <div class="section">
        <h2>🧠 Chat de Texto (Persistente)</h2>
        <div class="chat-container" id="textChat" aria-live="polite"></div>
        <textarea id="textInput" rows="4" placeholder="Escribe tu pregunta tecnológica o de código aquí..." aria-label="Mensaje de texto"></textarea>
        <div style="display: flex; justify-content: flex-start; margin-top: 10px;">
            <button id="textButton" onclick="sendText()" aria-label="Enviar mensaje de texto">Enviar [↵]</button>
            <button id="resetButton" onclick="resetChat()" aria-label="Reiniciar chat">Reiniciar Chat</button>
        </div>
    </div>

    <div class="section">
        <h2>🖼️ Análisis Multimodal (Imagen + Texto)</h2>
        <input type="url" id="imageUrl" placeholder="URL de la imagen (ej: https://...)" aria-label="URL de imagen">
        <textarea id="imageText" rows="3" placeholder="¿Qué deseas saber o analizar sobre esta imagen? (No tiene historial)" aria-label="Pregunta sobre imagen"></textarea>
        <button id="imageButton" onclick="sendImage()" aria-label="Enviar con imagen">Analizar Imagen</button>
        <div id="imageResponse" class="message assistant-message" style="margin-top: 20px;"></div>
    </div>

    <footer>
        ✨ Desarrollado por <b>TecSoft AI</b> para Proyecto Universitario |
        ⚙️ Motorizado por Flask, OpenRouter (kwaipilot/kat-coder-pro:free & x-ai/grok-4.1-fast) |
        🔒 Sesiones: <span id="sessionStatus">Inactiva</span>
    </footer>

    <script>
        // --- Lógica de la interfaz mejorada ---

        const textInput = document.getElementById('textInput');
        const textChat = document.getElementById('textChat');
        const textButton = document.getElementById('textButton');
        const imageButton = document.getElementById('imageButton');

        // Escuchar Enter para enviar mensaje
        textInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendText();
            }
        });
        
        function updateSessionStatus(active) {
            const statusElement = document.getElementById('sessionStatus');
            statusElement.textContent = active ? 'Activa' : 'Inactiva';
            statusElement.style.color = active ? 'var(--primary-color)' : 'var(--error-color)';
        }
        updateSessionStatus(true); // Asumimos que la sesión está activa al cargar

        /**
         * Agrega un mensaje al contenedor de chat.
         * @param {string} role 'user' o 'assistant'
         * @param {string} content El contenido del mensaje (se renderiza Markdown si es asistente).
         * @param {HTMLElement | null} targetElement El elemento a reemplazar o null si se añade uno nuevo.
         */
        function addMessage(role, content, targetElement = null) {
            const chatContainer = document.getElementById('textChat');
            let messageDiv;

            if (targetElement && chatContainer.contains(targetElement)) {
                messageDiv = targetElement;
                messageDiv.className = role === 'user' ? 'message user-message' : 'message assistant-message';
            } else {
                messageDiv = document.createElement('div');
                messageDiv.className = role === 'user' ? 'message user-message' : 'message assistant-message';
                chatContainer.appendChild(messageDiv);
            }
            
            // Usar DOMPurify o similar en producción, pero aquí renderizaremos el markdown
            if (role === 'assistant') {
                // Renderizado de Markdown (Simplificado para el ejemplo)
                messageDiv.innerHTML = renderMarkdown(content);
            } else {
                messageDiv.textContent = content;
            }
            
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return messageDiv; // Retorna el elemento para posible uso posterior
        }
        
        function renderMarkdown(markdownText) {
            // Implementación de renderizado de Markdown (muy simple)
            // Para un proyecto universitario robusto, se recomienda usar una librería JS como 'marked.js'
            let html = markdownText
                .replace(/```([\s\S]*?)```/g, (match, code) => `<pre><code>${code.trim()}</code></pre>`)
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/^(#+)\s*(.*)$/gm, (match, hashes, content) => {
                    const level = hashes.length > 6 ? 6 : hashes.length;
                    return `<h${level}>${content}</h${level}>`;
                })
                .replace(/^- (.*)$/gm, '<li>$1</li>') // Lista simple
                ;
            
            // Envolver el texto restante en párrafos (mejorar el manejo de párrafos)
            html = html.split('\n\n').map(p => {
                if (p.startsWith('<h') || p.startsWith('<pre') || p.startsWith('<li')) return p;
                return `<p>${p.replace(/\n/g, '<br>')}</p>`;
            }).join('');
            return html;
        }

        // Función de tipeo simulado
        function typeResponse(element, fullText) {
            let i = 0;
            const speed = 20; // Velocidad en milisegundos (ajustable)
            element.innerHTML = ''; // Limpiar el contenido antes de empezar

            function type() {
                if (i < fullText.length) {
                    element.textContent += fullText.charAt(i);
                    textChat.scrollTop = textChat.scrollHeight; // Scroll automático
                    i++;
                    setTimeout(type, speed);
                } else {
                    // Al terminar, renderizar el markdown completo
                    element.innerHTML = renderMarkdown(fullText);
                    textButton.disabled = false;
                    imageButton.disabled = false; // Desbloquear otros botones si es necesario
                }
            }
            
            type();
        }


        // Cargar Historial (Asumimos que el historial lo maneja la sesión de Flask,
        // pero para demostrar la persistencia, se hace una llamada inicial)
        async function loadInitialHistory() {
            // En un caso real con Flask Sessions, esto requeriría una ruta /api/history
            // Por simplicidad, aquí cargamos un mensaje de bienvenida.
            addMessage('assistant', "Hola! Soy **TecSoft AI**, tu asistente para proyectos de universidad. ¿En qué puedo ayudarte hoy?", false);
        }
        loadInitialHistory();


        async function sendText() {
            const text = textInput.value.trim();
            if (!text) return alert("Escribe un mensaje para TecSoft AI");

            textButton.disabled = true;
            imageButton.disabled = true;
            textInput.value = '';

            // 1. Mostrar mensaje de usuario
            addMessage('user', text);

            // 2. Agregar elemento de loading
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message assistant-message loading';
            loadingDiv.textContent = '⏳ Procesando en el servidor...';
            textChat.appendChild(loadingDiv);
            textChat.scrollTop = textChat.scrollHeight;

            try {
                // La URL se modificó para usar el historial de sesión de Flask
                const res = await fetch('/api/text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    // Solo se envía el nuevo mensaje, el historial lo maneja la sesión
                    body: JSON.stringify({ message: text }) 
                });
                
                const data = await res.json();
                textChat.removeChild(loadingDiv); // Eliminar loading

                if (res.ok) {
                    const responseElement = addMessage('assistant', data.reply);
                    // Efecto de tipeo solo para la respuesta exitosa
                    typeResponse(responseElement, data.reply);
                } else {
                    // Mostrar error de servidor o rate limit
                    const errorMsg = '❌ Error del servidor: ' + (data.error || 'Desconocido');
                    addMessage('assistant', errorMsg);
                    textButton.disabled = false;
                    imageButton.disabled = false;
                }

            } catch (e) {
                console.error('Error de red:', e);
                const chatContainer = document.getElementById('textChat');
                if (chatContainer.contains(loadingDiv)) chatContainer.removeChild(loadingDiv);
                addMessage('assistant', '⚠️ Error de conexión. Intenta de nuevo.');
                textButton.disabled = false;
                imageButton.disabled = false;
            }
        }

        async function resetChat() {
            try {
                const res = await fetch('/api/reset', { method: 'POST' });
                if (res.ok) {
                    textChat.innerHTML = '';
                    loadInitialHistory();
                    alert("Chat reseteado. El historial de sesión ha sido limpiado.");
                } else {
                    const data = await res.json();
                    alert("Error al resetear el chat: " + (data.error || 'Desconocido'));
                }
            } catch (e) {
                alert("Error de conexión al resetear el chat.");
            }
        }

        async function sendImage() {
            const image = document.getElementById('imageUrl').value.trim();
            const text = document.getElementById('imageText').value.trim();
            const output = document.getElementById('imageResponse');
            
            if (!image || !text) return output.innerHTML = "<p class='error'>❌ Proporciona texto y una URL de imagen válida.</p>";
            
            imageButton.disabled = true;
            textButton.disabled = true;
            output.innerHTML = `<img src="${image}" alt="Imagen a analizar" class="multimodal-preview"><p class='loading' style="margin-top:10px;">🖼️ Analizando imagen: ${text}</p>`;

            try {
                const res = await fetch('/api/image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, image_url: image })
                });
                
                const data = await res.json();
                
                if (res.ok) {
                    output.innerHTML = `<img src="${image}" alt="Imagen analizada" class="multimodal-preview">` + 
                                         `<p class='success' style="margin-top:15px;">**Respuesta del Modelo:**</p>` + 
                                          renderMarkdown(data.reply);
                } else {
                    output.innerHTML = `<img src="${image}" alt="Imagen a analizar" class="multimodal-preview" style="opacity:0.5;">` +
                                         `<p class='error' style="margin-top:15px;">❌ Error al analizar: ${data.error || "Error desconocido"}</p>`;
                }
            } catch (e) {
                output.innerHTML = `<p class='error'>⚠️ Error de conexión/red: ${e.message}</p>`;
            } finally {
                imageButton.disabled = false;
                textButton.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

# Bloque de ejecución principal
if __name__ == '__main__':
    # Usar el puerto 5000 o el que esté configurado en la variable de entorno
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Iniciando TecSoft AI en puerto {port}")
    # Nota: En producción, usar un servidor WSGI como Gunicorn.
    app.run(debug=True, port=port, host='0.0.0.0')

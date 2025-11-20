from flask import Flask, request, jsonify, render_template_string, session
import requests
import json
import os
import logging
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import time

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)

# Configurar clave secreta para sesiones (usar variable de entorno en producción)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key_change_in_prod")

# Habilitar CORS para solicitudes desde el frontend
CORS(app)

# Configurar rate limiting para evitar abuso
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour", "10 per minute"]
)

# Configurar logging avanzado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Variables de entorno
API_KEY = os.getenv("OPENROUTER_KEY")
if not API_KEY:
    raise ValueError("La variable de entorno OPENROUTER_KEY no está configurada. Por favor, configúrala con tu clave API de OpenRouter.")

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Modelos configurables
TEXT_MODEL = "kwaipilot/kat-coder-pro:free"  # Modelo para texto
IMAGE_MODEL = "x-ai/grok-4.1-fast"  # Modelo para imagen + texto

# Función auxiliar mejorada para comunicarse con OpenRouter con retries y validación
def query_model(model, messages, max_retries=3):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 1000}  # Limitar tokens para eficiencia
    
    for attempt in range(max_retries):
        try:
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "Sin respuesta")
            logging.info(f"Respuesta exitosa del modelo {model} en intento {attempt + 1}: {content[:100]}...")
            return content
        except requests.exceptions.Timeout:
            logging.warning(f"Tiempo de espera agotado en intento {attempt + 1}")
            if attempt == max_retries - 1:
                return "Error: Tiempo de espera agotado en la API después de varios intentos."
            time.sleep(2 ** attempt)  # Exponential backoff
        except requests.exceptions.RequestException as e:
            logging.error(f"Error en la API en intento {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return f"Error en la API: {str(e)}"
            time.sleep(2 ** attempt)
    return "Error: No se pudo obtener respuesta después de varios intentos."

# HTML template mejorado con más funcionalidades, accesibilidad y optimizaciones
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TecSoft AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600&display=swap" rel="stylesheet">
    <style>
        /* Estilos mejorados con variables CSS para mantenibilidad */
        :root {
            --primary-color: #00ffff;
            --secondary-color: #ff00ff;
            --bg-color: radial-gradient(circle at center, #0a0a1a, #000010 80%);
            --text-color: #ffffff;
            --error-color: #ff4444;
            --success-color: #44ff44;
        }

        body {
            font-family: 'Rajdhani', sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
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

        h1 {
            margin-top: 40px;
            text-align: center;
            font-size: clamp(2em, 5vw, 3.2em);
            color: var(--primary-color);
            text-shadow: 0 0 30px var(--primary-color), 0 0 60px var(--secondary-color);
            animation: glow 2.5s infinite alternate;
            letter-spacing: 2px;
        }

        @keyframes glow {
            from { text-shadow: 0 0 15px var(--primary-color), 0 0 30px var(--secondary-color); }
            to { text-shadow: 0 0 40px var(--primary-color), 0 0 80px var(--secondary-color); }
        }

        .section {
            width: 90%;
            max-width: 750px;
            background: rgba(0, 0, 25, 0.9);
            border: 2px solid var(--primary-color);
            border-radius: 15px;
            padding: 30px;
            margin: 25px 0;
            box-shadow: 0 0 40px rgba(0, 255, 255, 0.4);
            backdrop-filter: blur(5px);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }

        .section:hover {
            transform: scale(1.03);
            box-shadow: 0 0 60px rgba(255, 0, 255, 0.6);
        }

        h2 {
            color: var(--secondary-color);
            text-shadow: 0 0 15px var(--secondary-color);
            margin-bottom: 15px;
            font-size: 1.5em;
        }

        textarea, input {
            width: 100%;
            padding: 14px;
            margin: 10px 0;
            border: 2px solid var(--primary-color);
            border-radius: 10px;
            background: rgba(255,255,255,0.07);
            color: var(--text-color);
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.1em;
            outline: none;
            transition: border-color 0.3s, box-shadow 0.3s;
        }

        textarea:focus, input:focus {
            border-color: var(--secondary-color);
            box-shadow: 0 0 15px var(--secondary-color);
        }

        button {
            padding: 12px 25px;
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            color: #000;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            font-size: 1.1em;
            transition: 0.3s;
            margin-top: 10px;
        }

        button:hover {
            background: linear-gradient(45deg, var(--secondary-color), var(--primary-color));
            box-shadow: 0 0 25px var(--secondary-color);
            transform: scale(1.07);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .chat-container {
            margin-top: 20px;
            max-height: 400px;
            overflow-y: auto;
            background: rgba(0, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px solid var(--primary-color);
            padding: 15px;
            box-shadow: inset 0 0 10px rgba(0,255,255,0.2);
        }

        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
            white-space: pre-wrap;
            font-size: 1em;
            word-wrap: break-word;
        }

        .user-message {
            background: rgba(0, 255, 255, 0.1);
            text-align: right;
            color: var(--primary-color);
        }

        .assistant-message {
            background: rgba(255, 0, 255, 0.1);
            color: var(--secondary-color);
        }

        .loading {
            color: var(--primary-color);
            font-style: italic;
        }

        .error {
            color: var(--error-color);
        }

        .success {
            color: var(--success-color);
        }

        footer {
            margin-top: 50px;
            color: #aaa;
            font-size: 0.9em;
            text-align: center;
        }

        @media (max-width: 600px) {
            h1 { font-size: 2.2em; }
            .section { padding: 20px; }
        }

        /* Animación de partículas suaves en el fondo */
        canvas#particles {
            position: fixed;
            top: 0; left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: transparent;
        }

        /* Accesibilidad: foco visible */
        button:focus, textarea:focus, input:focus {
            outline: 2px solid var(--secondary-color);
        }
    </style>
</head>
<body>
    <canvas id="particles"></canvas>

    <audio autoplay loop volume="0.2" aria-label="Música de fondo lofi">
        <source src="https://cdn.pixabay.com/download/audio/2022/03/15/audio_72a1cdb55e.mp3?filename=lofi-study-112191.mp3" type="audio/mpeg">
        Tu navegador no soporta audio.
    </audio>

    <h1>🚀 TecSoft AI</h1>

    <div class="section">
        <h2>🧠 Chat de Texto</h2>
        <div class="chat-container" id="textChat"></div>
        <textarea id="textInput" rows="4" placeholder="Escribe tu mensaje aquí..." aria-label="Mensaje de texto"></textarea>
        <button id="textButton" onclick="sendText()" aria-label="Enviar mensaje de texto">Enviar</button>
        <button id="resetButton" onclick="resetChat()" aria-label="Reiniciar chat">Reiniciar Chat</button>
    </div>

    <div class="section">
        <h2>🖼️ Imagen + Texto</h2>
        <input type="url" id="imageUrl" placeholder="URL de la imagen..." aria-label="URL de imagen">
        <textarea id="imageText" rows="4" placeholder="¿Qué deseas saber sobre la imagen?" aria-label="Pregunta sobre imagen"></textarea>
        <button id="imageButton" onclick="sendImage()" aria-label="Enviar con imagen">Enviar con Imagen</button>
        <div id="imageResponse" class="message assistant-message" aria-live="polite"></div>
    </div>

    <footer>✨ Desarrollado por <b>TecSoft AI</b> | Con tecnología futurista ⚙️</footer>

    <script>
        // Partículas suaves
        const canvas = document.getElementById('particles');
        const ctx = canvas.getContext('2d');
        let particles = [];

        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

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
            ctx.fillStyle = 'rgba(0,255,255,0.6)';
            particles.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
                p.x += p.speedX;
                p.y += p.speedY;
                if (p.x < 0 || p.x > canvas.width) p.speedX *= -1;
                if (p.y < 0 || p.y > canvas.height) p.speedY *= -1;
            });
            requestAnimationFrame(drawParticles);
        }
        drawParticles();

        // Cargar historial de chat desde localStorage para persistencia básica
        let chatHistory = JSON.parse(localStorage.getItem('chatHistory')) || [];

        function saveChatHistory() {
            localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        }

        function loadChatHistory() {
            const chatContainer = document.getElementById('textChat');
            chatContainer.innerHTML = '';
            chatHistory.forEach(msg => addMessage(msg.role, msg.content, false));
        }

        function addMessage(role, content, save = true) {
            const chatContainer = document.getElementById('textChat');
            const messageDiv = document.createElement('div');
            messageDiv.className = role === 'user' ? 'message user-message' : 'message assistant-message';
            messageDiv.textContent = content;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            if (save) {
                chatHistory.push({ role, content });
                saveChatHistory();
            }
        }

        // Cargar historial al inicio
        loadChatHistory();

        async function sendText() {
            const text = document.getElementById('textInput').value.trim();
            const button = document.getElementById('textButton');
            if (!text) return alert("Escribe algo primero");
            button.disabled = true;
            document.getElementById('textInput').value = '';

            addMessage('user', text);

            // Agregar mensaje de loading
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message assistant-message loading';
            loadingDiv.textContent = '⏳ Procesando...';
            document.getElementById('textChat').appendChild(loadingDiv);

            try {
                const res = await fetch('/api/text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: chatHistory })
                });
                let data;
                if (res.ok) {
                    data = await res.json();
                    document.getElementById('textChat').removeChild(loadingDiv);
                    addMessage('assistant', data.reply);
                } else {
                    try {
                        data = await res.json();
                        document.getElementById('textChat').removeChild(loadingDiv);
                        addMessage('assistant', '❌ ' + (data.error || 'Error desconocido'));
                    } catch (e) {
                        document.getElementById('textChat').removeChild(loadingDiv);
                        addMessage('assistant', '❌ Error en la respuesta del servidor');
                    }
                }
            } catch (e) {
                const chatContainer = document.getElementById('textChat');
                if (chatContainer.contains(loadingDiv)) chatContainer.removeChild(loadingDiv);
                addMessage('assistant', '⚠️ ' + e.message);
            } finally {
                button.disabled = false;
            }
        }

        function resetChat() {
            chatHistory = [];
            saveChatHistory();
            loadChatHistory();
        }

        async function sendImage() {
            const image = document.getElementById('imageUrl').value.trim();
            const text = document.getElementById('imageText').value.trim();
            const output = document.getElementById('imageResponse');
            const button = document.getElementById('imageButton');
            if (!image || !text) return alert("Proporciona texto y una URL de imagen");
            button.disabled = true;
            output.innerHTML = "<p class='loading'>🖼️ Analizando imagen...</p>";

            try {
                const res = await fetch('/api/image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, image_url: image })
                });
                let data;
                if (res.ok) {
                    data = await res.json();
                    output.innerHTML = "<p class='success'>" + data.reply + "</p>";
                } else {
                    try {
                        data = await res.json();
                        output.innerHTML = "<p class='error'>❌ " + (data.error || "Error desconocido") + "</p>";
                    } catch (e) {
                        output.innerHTML = "<p class='error'>❌ Error en la respuesta del servidor</p>";
                    }
                }
            } catch (e) {
                output.innerHTML = "<p class='error'>⚠️ " + e.message + "</p>";
            } finally {
                button.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

# Manejadores de error mejorados
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Ruta no encontrada'}), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Error 500: {str(error)}")
    return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/text', methods=['POST'])
@limiter.limit("10 per minute")  # Limitar solicitudes de texto
def api_text():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Datos JSON requeridos'}), 400
        messages = data.get('messages', [])
        if not messages or not isinstance(messages, list):
            return jsonify({'error': 'Lista de mensajes requerida'}), 400
        # Validar cada mensaje
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                return jsonify({'error': 'Formato de mensaje inválido'}), 400
            if len(str(msg

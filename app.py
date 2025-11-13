from flask import Flask, request, jsonify, render_template_string
import requests
import json
import os
import logging
from dotenv import load_dotenv  # 👈 IMPORTANTE: agrega esta línea

# Cargar variables de entorno
load_dotenv()  # 👈 Y esta línea también

app = Flask(__name__)

# Configurar logging para debugging
logging.basicConfig(level=logging.INFO)

# ⚠️ Usa variable de entorno
API_KEY = os.getenv("OPENROUTER_KEY")
if not API_KEY:
    raise ValueError("La variable de entorno OPENROUTER_KEY no está configurada. Por favor, configúrala con tu clave API de OpenRouter.")

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# 🧩 HTML futurista mejorado con correcciones y mejoras
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TecSoft AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Rajdhani', sans-serif;
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

        h1 {
            margin-top: 40px;
            text-align: center;
            font-size: 3.2em;
            color: #00ffff;
            text-shadow: 0 0 30px #00ffff, 0 0 60px #ff00ff;
            animation: glow 2.5s infinite alternate;
            letter-spacing: 2px;
        }

        @keyframes glow {
            from { text-shadow: 0 0 15px #00ffff, 0 0 30px #ff00ff; }
            to { text-shadow: 0 0 40px #00ffff, 0 0 80px #ff00ff; }
        }

        .section {
            width: 90%;
            max-width: 750px;
            background: rgba(0, 0, 25, 0.9);
            border: 2px solid #00ffff;
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
            color: #ff00ff;
            text-shadow: 0 0 15px #ff00ff;
            margin-bottom: 15px;
            font-size: 1.5em;
        }

        textarea, input {
            width: 100%;
            padding: 14px;
            margin: 10px 0;
            border: 2px solid #00ffff;
            border-radius: 10px;
            background: rgba(255,255,255,0.07);
            color: #fff;
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.1em;
            outline: none;
            transition: border-color 0.3s, box-shadow 0.3s;
        }

        textarea:focus, input:focus {
            border-color: #ff00ff;
            box-shadow: 0 0 15px #ff00ff;
        }

        button {
            padding: 12px 25px;
            background: linear-gradient(45deg, #00ffff, #ff00ff);
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
            background: linear-gradient(45deg, #ff00ff, #00ffff);
            box-shadow: 0 0 25px #ff00ff;
            transform: scale(1.07);
        }

        .response {
            margin-top: 20px;
            padding: 15px;
            background: rgba(0, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px solid #00ffff;
            white-space: pre-wrap;
            box-shadow: inset 0 0 10px rgba(0,255,255,0.2);
            font-size: 1em;
            min-height: 60px;
        }

        .loading {
            color: #00ffff;
            font-style: italic;
        }

        .error {
            color: #ff4444;
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
    </style>
</head>
<body>
    <canvas id="particles"></canvas>

    <audio autoplay loop volume="0.2">
        <source src="https://cdn.pixabay.com/download/audio/2022/03/15/audio_72a1cdb55e.mp3?filename=lofi-study-112191.mp3" type="audio/mpeg">
        Tu navegador no soporta audio.
    </audio>

    <h1>🚀 TecSoft AI</h1>

    <div class="section">
        <h2>🧠 Chat de Texto</h2>
        <textarea id="textInput" rows="4" placeholder="Escribe tu mensaje aquí..."></textarea>
        <button id="textButton" onclick="sendText()">Enviar</button>
        <div id="textResponse" class="response"></div>
    </div>

    <div class="section">
        <h2>🖼️ Imagen + Texto</h2>
        <input type="url" id="imageUrl" placeholder="URL de la imagen...">
        <textarea id="imageText" rows="4" placeholder="¿Qué deseas saber sobre la imagen?"></textarea>
        <button id="imageButton" onclick="sendImage()">Enviar con Imagen</button>
        <div id="imageResponse" class="response"></div>
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

        async function sendText() {
            const text = document.getElementById('textInput').value.trim();
            const output = document.getElementById('textResponse');
            const button = document.getElementById('textButton');
            if (!text) return alert("Escribe algo primero");
            button.disabled = true;
            output.innerHTML = "<p class='loading'>⏳ Procesando...</p>";

            try {
                const res = await fetch('/api/text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await res.json();
                if (res.ok) output.textContent = data.reply;
                else output.innerHTML = "<p class='error'>❌ " + (data.error || "Error desconocido") + "</p>";
            } catch (e) {
                output.innerHTML = "<p class='error'>⚠️ " + e.message + "</p>";
            } finally {
                button.disabled = false;
            }
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
                const data = await res.json();
                if (res.ok) output.textContent = data.reply;
                else output.innerHTML = "<p class='error'>❌ " + (data.error || "Error desconocido") + "</p>";
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


# 🧩 Función auxiliar para comunicarse con OpenRouter con mejoras
def query_model(model, messages):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload, timeout=30)  # Agregar timeout
        response.raise_for_status()
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "Sin respuesta")
        logging.info(f"Respuesta del modelo {model}: {content[:100]}...")  # Log parcial para debugging
        return content
    except requests.exceptions.Timeout:
        return "Error: Tiempo de espera agotado en la API."
    except requests.exceptions.RequestException as e:
        logging.error(f"Error en la API: {str(e)}")
        return f"Error en la API: {str(e)}"

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/text', methods=['POST'])
def api_text():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Datos JSON requeridos'}), 400
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Texto requerido'}), 400
        if len(text) > 10000:  # Limitar longitud para evitar abuso
            return jsonify({'error': 'Texto demasiado largo (máx. 10000 caracteres)'}), 400

        reply = query_model("z-ai/glm-4.5-air:free", [{"role": "user", "content": text}])
        return jsonify({'reply': reply})
    except Exception as e:
        logging.error(f"Error en /api/text: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/image', methods=['POST'])
def api_image():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Datos JSON requeridos'}), 400
        text = data.get('text', '').strip()
        image_url = data.get('image_url', '').strip()
        if not text or not image_url:
            return jsonify({'error': 'Texto e imagen requeridos'}), 400
        if len(text) > 5000 or len(image_url) > 2000:  # Limitar longitud
            return jsonify({'error': 'Texto o URL demasiado largos'}), 400

        # Validar URL básica
        if not image_url.startswith(('http://', 'https://')):
            return jsonify({'error': 'URL de imagen inválida'}), 400

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
        reply = query_model("google/gemma-3-12b-it:free", messages)
        return jsonify({'reply': reply})
    except Exception as e:
        logging.error(f"Error en /api/image: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

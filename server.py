from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import asyncio
from typing import Dict
from contextlib import asynccontextmanager

# 假设的TTS工厂类
class TTSFactory:
    @staticmethod
    def get_tts_engine(engine_name: str, language: str, device: str):
        # 这里应该是你的实际TTS引擎实现
        class TTSEngine:
            def __init__(self, language: str, device: str):
                print(f"Initializing TTS Engine - Language: {language}, Device: {device}")
                # 模拟耗时的初始化过程
                import time
                time.sleep(2)
                self.language = language
                self.device = device
                print("TTS Engine initialized")
            
            def generate_audio(self, text: str) -> str:
                audio_dir = "temp_audio"
                os.makedirs(audio_dir, exist_ok=True)
                filename = f"{uuid.uuid4().hex}.wav"
                filepath = os.path.join(audio_dir, filename)
                
                print(f"Generating audio for: {text}")
                
                # 模拟生成文件
                with open(filepath, "wb") as f:
                    f.write(b"fake_audio_data")  # 替换为实际音频数据
                
                return filepath
        
        return TTSEngine(language, device)

# 全局TTS引擎实例
tts_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化TTS引擎
    global tts_engine
    print("Initializing TTS engine on startup...")
    tts_engine = TTSFactory.get_tts_engine("melo_tts", "ZH", "AX650")
    yield
    # 关闭时清理资源
    print("Shutting down TTS service...")
    # 这里可以添加引擎的清理代码

app = FastAPI(
    title="WebSocket TTS Service",
    description="Optimized FastAPI WebSocket TTS service with pre-initialized engine",
    version="1.0.0",
    lifespan=lifespan
)

# 静态文件服务（用于测试页面）
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_audio(self, client_id: str, audio_path: str):
        if client_id not in self.active_connections:
            return False
        
        websocket = self.active_connections[client_id]
        
        try:
            # 发送音频文件
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
                await websocket.send_bytes(audio_data)
            
            # 发送完成后删除临时文件
            try:
                os.remove(audio_path)
            except:
                pass
                
            return True
        except Exception as e:
            print(f"Error sending audio: {e}")
            return False

manager = ConnectionManager()

@app.websocket("/ws/tts/{client_id}")
async def websocket_tts_endpoint(websocket: WebSocket, client_id: str):
    global tts_engine
    
    if tts_engine is None:
        await websocket.close(code=1008, reason="TTS engine not initialized")
        return
    
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # 接收客户端发送的文本
            text = await websocket.receive_text()
            print(f"Received text from {client_id}: {text}")
            
            # 使用预初始化的引擎生成音频
            audio_path = tts_engine.generate_audio(text)
            
            # 通过WebSocket发送音频
            success = await manager.send_audio(client_id, audio_path)
            
            if not success:
                break
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        print(f"Client {client_id} disconnected")
    except Exception as e:
        manager.disconnect(client_id)
        print(f"Error with client {client_id}: {e}")

@app.get("/")
async def get():
    return HTMLResponse("""
    <html>
        <head>
            <title>WebSocket TTS Test</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                #status { margin: 10px 0; padding: 10px; border-radius: 5px; }
                .connected { background-color: #d4edda; color: #155724; }
                .disconnected { background-color: #f8d7da; color: #721c24; }
            </style>
        </head>
        <body>
            <h1>WebSocket TTS Test</h1>
            <div id="status" class="disconnected">Disconnected</div>
            <form action="" onsubmit="sendMessage(event)">
                <input type="text" id="messageText" autocomplete="off" placeholder="Enter text to convert"/>
                <button>Convert to Speech</button>
            </form>
            <audio id="audioPlayer" controls></audio>
            <script>
                const clientId = Math.random().toString(36).substring(2);
                let socket = null;
                
                function connectWebSocket() {
                    socket = new WebSocket(`ws://${location.host}/ws/tts/${clientId}`);
                    
                    socket.onopen = function() {
                        document.getElementById("status").className = "connected";
                        document.getElementById("status").textContent = "Connected (Client ID: " + clientId + ")";
                    };
                    
                    socket.onclose = function() {
                        document.getElementById("status").className = "disconnected";
                        document.getElementById("status").textContent = "Disconnected";
                        setTimeout(connectWebSocket, 3000); // 尝试重新连接
                    };
                    
                    socket.onmessage = function(event) {
                        if (event.data instanceof Blob) {
                            const audioUrl = URL.createObjectURL(event.data);
                            const audioPlayer = document.getElementById("audioPlayer");
                            audioPlayer.src = audioUrl;
                            audioPlayer.play();
                        }
                    };
                }
                
                function sendMessage(event) {
                    event.preventDefault();
                    const input = document.getElementById("messageText");
                    if (socket && socket.readyState === WebSocket.OPEN) {
                        socket.send(input.value);
                        input.value = '';
                    } else {
                        alert("WebSocket connection is not ready");
                    }
                }
                
                // 初始连接
                connectWebSocket();
            </script>
        </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
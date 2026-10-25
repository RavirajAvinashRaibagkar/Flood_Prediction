from typing import List, Dict
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        # active_connections maps channel names (e.g., 'alerts', 'predictions') to lists of WebSockets
        self.active_connections: Dict[str, List[WebSocket]] = {
            "alerts": [],
            "predictions": []
        }

    async def connect(self, websocket: WebSocket, channel: str = "alerts"):
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = []
        self.active_connections[channel].append(websocket)

    def disconnect(self, websocket: WebSocket, channel: str = "alerts"):
        if channel in self.active_connections:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict, channel: str = "alerts"):
        if channel in self.active_connections:
            for connection in self.active_connections[channel]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"Error broadcasting to {channel}: {e}")
                    pass

manager = ConnectionManager()

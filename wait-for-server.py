# wait-for-server.py
import socket
import time
import os

server_host = os.environ.get("FLOWER_SERVER", "server")
server_port = int(os.environ.get("FLOWER_PORT", 8080))

def wait_for_server(host, port):
    while True:
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"✅ Server is up: {host}:{port}")
                return
        except OSError:
            print(f"❌ Waiting for server {host}:{port}...")
            time.sleep(2)

wait_for_server(server_host, server_port)

@echo off
echo Starting DeployMate AI...

echo Starting Ollama...
start cmd /k "ollama serve"

echo Waiting for Ollama to be ready (15 seconds)...
timeout /t 15 /nobreak

echo Starting Docker PostgreSQL + App...
docker compose down
docker compose up -d

echo Waiting for services (10 seconds)...
timeout /t 10 /nobreak

echo Starting Cloudflare Tunnel...
start cmd /k "cloudflared tunnel --url http://localhost:8501"

echo DeployMate AI is starting up!
pause
```


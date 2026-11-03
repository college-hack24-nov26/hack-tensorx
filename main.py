from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ECG X")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html", context={"active_page": "home"})

@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"active_page": "dashboard"})

@app.get("/about")
def about(request: Request):
    return templates.TemplateResponse(request=request, name="about.html", context={"active_page": "about"})

if __name__ == "__main__":
    import uvicorn
    import socket

    # Get IP Adress
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    port = 8000

    print(f"\n" + "="*50)
    print(f" TENSORX CLINICAL AI IS STARTING...")
    print(f" Local Access:   http://localhost:{port}")
    print(f" Network Access: http://{local_ip}:{port}")
    print("="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
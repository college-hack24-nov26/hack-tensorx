from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes.predict import router as predict_router
from routes.chatbot import router as chatbot_router
from routes.simulator_stream import router as sim_router
from database import init_db, get_history, clear_history
import os
from dotenv import load_dotenv

load_dotenv()

init_db()

app = FastAPI(title="ECG X")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(predict_router)
app.include_router(chatbot_router)
app.include_router(sim_router)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html", context={"active_page": "home"})

@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"active_page": "dashboard"})

@app.get("/about")
def about(request: Request):
    return templates.TemplateResponse(request=request, name="about.html", context={"active_page": "about"})

@app.get("/chatbot")
def chatbot_page(request: Request):
    return templates.TemplateResponse(request=request, name="chatbot.html", context={"active_page": "chatbot"})

@app.get("/live")
def live_monitor(request: Request):
    return templates.TemplateResponse(request=request, name="live.html", context={"active_page": "live"})

@app.get("/history")
def history(request: Request):
    data = get_history()
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"data": data, "active_page": "history"}
    )

@app.post("/history/clear")
def history_clear():
    clear_history()
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/history", status_code=303)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join(STATIC_DIR, "favicon.ico")) if os.path.exists(os.path.join(STATIC_DIR, "favicon.ico")) else None

if __name__ == "__main__":
    import uvicorn
    import socket

    # Get IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    port = 8000

    print(f"\n" + "="*50)
    print(f" TENSORX CLINICAL AI IS STARTING...")
    print(f" Local Access:   http://localhost:{port}")
    print(f" Network Access: http://{local_ip}:{port}")
    print("="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port)

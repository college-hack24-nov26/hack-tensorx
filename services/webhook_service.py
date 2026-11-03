import requests
import json
import traceback
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
ESP32_IP = os.getenv("ESP32_IP")

CRITICAL_CONDITIONS = ["VTCC", "AFib"]

def trigger_buzzer(on: bool = True):
    if not ESP32_IP:
        print("ESP32_IP not set in .env")
        return
    action = "on" if on else "off"
    try:
        response = requests.get(f"http://{ESP32_IP}/{action}", timeout=3)
        print(f"Buzzer {action.upper()}: {response.text}")
    except Exception as e:
        print(f"Buzzer trigger failed: {str(e)}")

def trigger_emergency_webhook(patient_data: dict, prediction: str, probabilities: dict, vitals: dict, analysis: dict):
    if prediction not in CRITICAL_CONDITIONS:
        return None
    confidence_key = prediction
    payload = {
        "event": f"{prediction}_EMERGENCY",
        "patient": {
            "name": patient_data.get("name", "Unknown"),
            "age": patient_data.get("age", "N/A"),
            "blood_group": patient_data.get("blood_group", "N/A"),
            "email": patient_data.get("email", "N/A"),
            "doctor_email": patient_data.get("doctor_email", "N/A"),
            "icu_ward": patient_data.get("icu_ward", "N/A"),
            "icu_room": patient_data.get("icu_room", "N/A")
        },
        "vitals": {
            "heart_rate": vitals.get("heart_rate", "N/A"),
            "bp": f"{vitals.get('bp_sys', '--')}/{vitals.get('bp_dia', '--')} mmHg",
            "spo2": vitals.get("spo2", "N/A")
        },
        "diagnosis": {
            "prediction": prediction,
            "confidence": f"{probabilities.get(confidence_key, 0)*100:.1f}%",
            "all_probabilities": {
                "NSR": f"{probabilities.get('NSR', 0)*100:.1f}%",
                "AFib": f"{probabilities.get('AFib', 0)*100:.1f}%",
                "VTCC": f"{probabilities.get('VTCC', 0)*100:.1f}%"
            }
        },
        "clinical": {
            "summary": analysis.get("interpretation", "No summary available."),
            "quick_actions": analysis.get("quick_steps", [])
        }
    }
    try:
        trigger_buzzer(on=True)
        response = requests.post(
            WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        print(f"Webhook triggered: Status {response.status_code}")
        return response.status_code
    except Exception as e:
        print(f"Error triggering webhook: {str(e)}")
        traceback.print_exc()
        return None


async def async_trigger_buzzer(on: bool = True):
    """Non-blocking ESP32 buzzer trigger for use inside async bridge loop."""
    if not ESP32_IP:
        print("ESP32_IP not set in .env")
        return
    action = "on" if on else "off"
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(f"http://{ESP32_IP}/{action}", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                text = await resp.text()
                print(f"[ESP32] Buzzer {action.upper()}: {text}")
    except Exception as e:
        print(f"[ESP32] Buzzer trigger failed: {e}")


async def async_trigger_emergency_webhook(
    prediction: str,
    probabilities: dict,
    # file-upload path
    patient_data: dict = None,
    vitals: dict = None,
    analysis: dict = None,
    # live-stream path
    rhythm: str = None,
    source: str = None,
    session_id: str = None
):
    """Non-blocking n8n webhook trigger for both file-upload and live stream detections."""
    if prediction not in CRITICAL_CONDITIONS:
        return
    if not WEBHOOK_URL:
        print("WEBHOOK_URL not set in .env")
        return

    if session_id is not None:
        # live stream payload
        payload = {
            "event": f"{prediction}_LIVE_EMERGENCY",
            "source": "live_stream",
            "session_id": session_id,
            "simulator_rhythm": rhythm,
            "simulator_source": source,
            "diagnosis": {
                "prediction": prediction,
                "confidence": f"{probabilities.get(prediction, 0)*100:.1f}%",
                "all_probabilities": {
                    "NSR":  f"{probabilities.get('NSR',  0)*100:.1f}%",
                    "AFib": f"{probabilities.get('AFib', 0)*100:.1f}%",
                    "VTCC": f"{probabilities.get('VTCC', 0)*100:.1f}%"
                }
            }
        }
    else:
        # file-upload payload
        pd_ = patient_data or {}
        v   = vitals or {}
        a   = analysis or {}
        payload = {
            "event": f"{prediction}_EMERGENCY",
            "patient": {
                "name":         pd_.get("name", "Unknown"),
                "age":          pd_.get("age", "N/A"),
                "blood_group":  pd_.get("blood_group", "N/A"),
                "email":        pd_.get("email", "N/A"),
                "doctor_email": pd_.get("doctor_email", "N/A"),
                "icu_ward":     pd_.get("icu_ward", "N/A"),
                "icu_room":     pd_.get("icu_room", "N/A")
            },
            "vitals": {
                "heart_rate": v.get("heart_rate", "N/A"),
                "bp": f"{v.get('bp_sys', '--')}/{v.get('bp_dia', '--')} mmHg",
                "spo2": v.get("spo2", "N/A")
            },
            "diagnosis": {
                "prediction": prediction,
                "confidence": f"{probabilities.get(prediction, 0)*100:.1f}%",
                "all_probabilities": {
                    "NSR":  f"{probabilities.get('NSR',  0)*100:.1f}%",
                    "AFib": f"{probabilities.get('AFib', 0)*100:.1f}%",
                    "VTCC": f"{probabilities.get('VTCC', 0)*100:.1f}%"
                }
            },
            "clinical": {
                "summary":       a.get("interpretation", "No summary available."),
                "quick_actions": a.get("quick_steps", [])
            }
        }
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                WEBHOOK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                print(f"[n8n] Live webhook triggered: {resp.status} — {prediction}")
    except Exception as e:
        print(f"[n8n] Live webhook failed: {e}")

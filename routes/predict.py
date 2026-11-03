from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
import numpy as np
import json
import ast
import pandas as pd
import traceback
import datetime
from io import BytesIO
from pydantic import BaseModel

from services.model_loader import predict_signal
from services.preprocess import preprocess_signal, clean_ecg_signal
from services.groq_analysis import get_llm_analysis
from services.webhook_service import async_trigger_buzzer, async_trigger_emergency_webhook
from services.report_generator import generate_pdf_report
from services.input_validator import load_and_validate_content
from database import save_prediction
from fastapi.responses import Response

router = APIRouter()

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    heart_rate: Optional[int] = Form(None),
    bp_sys: Optional[int] = Form(None),
    bp_dia: Optional[int] = Form(None),
    spo2: Optional[int] = Form(None),
    patient_name: Optional[str] = Form("John Doe"),
    patient_email: Optional[str] = Form("N/A"),
    patient_age: Optional[int] = Form(None),
    patient_blood_group: Optional[str] = Form(None),
    icu_ward: Optional[str] = Form(None),
    icu_room: Optional[str] = Form(None),
    doctor_email: Optional[str] = Form(None)
):
    try:
        content = await file.read()
        filename = file.filename.lower()

        vitals = {
            "heart_rate": heart_rate,
            "bp_sys": bp_sys,
            "bp_dia": bp_dia,
            "spo2": spo2
        }
        
        patient_data = {
            "name": patient_name,
            "email": patient_email,
            "age": patient_age,
            "blood_group": patient_blood_group,
            "icu_ward": icu_ward,
            "icu_room": icu_room,
            "doctor_email": doctor_email
        }

        # ── ROBUST VALIDATION PIPELINE ─────────────────────────────────────────
        results = load_and_validate_content(content, filename)
        
        valid_results = [r for r in results if r["signal"] is not None]
        if not valid_results:
            errors = [f"Row {r['row']}: {r['error']}" for r in results if r["error"]]
            raise HTTPException(status_code=400, detail="No valid signals found in input. ERRORS: " + "; ".join(errors))

        # ── MULTI-ROW / BATCH PROCESSING ─────────────────────────────────────
        if len(results) > 1 or filename.endswith(".csv"):
            batch_results = []
            for r in results:
                if r["signal"] is not None:
                    # Valid signal found
                    sig = r["signal"]
                    filtered_for_ui = clean_ecg_signal(sig)
                    processed_signal = preprocess_signal(sig)
                    result = predict_signal(processed_signal)
                    
                    # Save to DB (optional: only save valid ones)
                    save_prediction(result["prediction"], result["probabilities"], vitals)
                    
                    batch_results.append({
                        "row": int(r["row"]),
                        "prediction": result["prediction"],
                        "probabilities": result["probabilities"],
                        "filtered_signal": filtered_for_ui.tolist(),
                    })
                else:
                    # Invalid row in a batch
                    batch_results.append({
                        "row": int(r["row"]),
                        "error": r["error"]
                    })
            
            # Sort by original row index
            batch_results.sort(key=lambda x: x["row"])
            return {"batch": True, "results": batch_results, "vitals": vitals}

        # ── SINGLE SIGNAL PIPELINE ────────────────────────────────────────────
        # (Already verified valid_results has at least one item)
        first_valid = valid_results[0]
        signal = first_valid["signal"]

        filtered_for_ui = clean_ecg_signal(signal)
        processed_signal = preprocess_signal(signal)
        result = predict_signal(processed_signal)
        save_prediction(result["prediction"], result["probabilities"], vitals)

        llm = get_llm_analysis(
            prediction=result["prediction"],
            probabilities=result["probabilities"],
            vitals=vitals,
            filtered_signal=filtered_for_ui.tolist(),
        )

        result["vitals"] = vitals
        result["filtered_signal"] = filtered_for_ui.tolist()
        result["input_length"] = int(signal.size)
        result["llm_analysis"] = llm
        result["batch"] = False

        # Trigger Webhook for VTCC and AFib
        if result["prediction"] in ["VTCC", "AFib"]:
            import asyncio
            asyncio.create_task(async_trigger_buzzer(on=True))
            asyncio.create_task(async_trigger_emergency_webhook(
                prediction=result["prediction"],
                probabilities=result["probabilities"],
                patient_data=patient_data,
                vitals=vitals,
                analysis=llm
            ))

        return result

    except Exception as e:
        if isinstance(e, HTTPException): raise e
        print(f"Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

class ReportRequest(BaseModel):
    patient_data: dict
    prediction: str
    probabilities: dict
    vitals: dict
    analysis: dict
    filtered_signal: list

@router.post("/download-report")
async def download_report(request: dict): # Simpler to use dict and parse manually or use proper pydantic
    try:
        pdf_content = generate_pdf_report(
            patient_data=request.get("patient_data"),
            prediction=request.get("prediction"),
            probabilities=request.get("probabilities"),
            vitals=request.get("vitals"),
            analysis=request.get("analysis"),
            filtered_signal=request.get("filtered_signal")
        )
        
        return Response(
            content=bytes(pdf_content),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=TensorX_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
    except Exception as e:
        print(f"Error generating PDF: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

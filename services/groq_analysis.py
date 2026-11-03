import os
import json
import numpy as np
from groq import Groq

MODEL = "openai/gpt-oss-120b"

def _ecg_stats(signal: list) -> dict:
    arr = np.array(signal, dtype=np.float32)
    return {
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "samples": len(arr),
        "duration_sec": round(len(arr) / 250, 1),
    }

def get_llm_analysis(prediction: str, probabilities: dict, vitals: dict, filtered_signal: list) -> dict:
    stats = _ecg_stats(filtered_signal)

    prompt = f"""You are a clinical ECG assistant. Analyze the data below and respond ONLY in the exact JSON format shown. Be brief.

ECG: {stats['duration_sec']}s | {stats['samples']} samples | 250Hz | Lead II | 0.5-40Hz filtered
Amplitude: min={stats['min']} max={stats['max']} mean={stats['mean']} std={stats['std']} mV

AI Result: {prediction} (NSR={round(probabilities.get('NSR',0)*100,1)}% AFib={round(probabilities.get('AFib',0)*100,1)}% VTCC={round(probabilities.get('VTCC',0)*100,1)}%)

Vitals: HR={vitals.get('heart_rate') or 'N/A'} BPM | BP={vitals.get('bp_sys') or '--'}/{vitals.get('bp_dia') or '--'} mmHg | SpO2={vitals.get('spo2') or 'N/A'}%

Respond ONLY with this JSON, no extra text:
{{"interpretation": "2-3 sentence clinical summary.", "quick_steps": ["Step 1", "Step 2", "Step 3"]}}"""

    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        result = json.loads(raw)
        print("\n" + "="*50)
        print("✅ GROQ API CALL SUCCESSFUL")
        print(f"   Model  : {MODEL}")
        print(f"  no of Tokens used for this query  : {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
        print(f"   Response returned is : {raw}")
        print("="*50 + "\n")
        return result
    except Exception as e:
        print("\n" + "="*50)
        print(f"❌ GROQ API CALL FAILED, check API key and Groq model Configs")
        print(f"   Error: {str(e)}")
        print("="*50 + "\n")
        return {
            "interpretation": f"LLM analysis unavailable: {str(e)}",
            "quick_steps": []
        }

import pandas as pd
import numpy as np
import ast
import json
import io

TARGET_LEN = 2500

def parse_signal(signal):
    """Converts input (list, numpy array, or stringified list) into a numpy array."""
    if isinstance(signal, np.ndarray):
        return signal
    if isinstance(signal, list):
        return np.array(signal)
    if isinstance(signal, str):
        try:
            # Handle cases where strings might have extra whitespace or newlines
            parsed = ast.literal_eval(signal.strip())
            return np.array(parsed)
        except:
            # Try to parse if it's just comma separated values
            try:
                parsed = [float(x) for x in signal.split(',') if x.strip()]
                return np.array(parsed)
            except:
                raise ValueError("Invalid signal format (could not parse string)")
    raise ValueError(f"Unsupported signal type: {type(signal)}")

def validate_signal(signal):
    """Flattens, validates numeric type, and enforces strict length."""
    try:
        signal = parse_signal(signal)
        # Flatten nested structures if any
        signal = signal.flatten()
        
        # Enforce numeric
        if not np.issubdtype(signal.dtype, np.number):
            raise ValueError("Non-numeric values found in signal")
            
        # Strict length check
        if signal.size != TARGET_LEN:
            raise ValueError(f"Expected {TARGET_LEN} values, got {signal.size}")
            
        return signal.astype(np.float32)
    except Exception as e:
        raise ValueError(str(e))

def load_and_validate_content(content: bytes, filename: str):
    """
    Parses raw bytes content based on filename extension.
    Returns a list of dicts: {"row": int, "signal": np.array, "error": str}
    """
    raw_signals = []
    results = []

    try:
        if filename.endswith(".csv") or filename.endswith(".txt"):
            # Try reading with headers first to find "signal" column
            try:
                df = pd.read_csv(io.BytesIO(content))
                if "signal" in df.columns:
                    raw_signals = df["signal"].tolist()
                else:
                    # Re-read without header for raw data formats
                    df = pd.read_csv(io.BytesIO(content), header=None)
                    if df.shape[1] == 1:
                        # Single column → every row is a sample? (User's Case 2)
                        # Actually, Case 2 usually means one signal spread across many rows
                        # or one signal per row. Let's follow the provided logic.
                        raw_signals = df.iloc[:, 0].tolist()
                    elif df.shape[1] == TARGET_LEN:
                        # Case 3: Each row is a full signal
                        raw_signals = df.values.tolist()
                    else:
                        raise ValueError(f"Unsupported CSV shape: {df.shape}")
            except Exception as e:
                # Fallback for very raw text files
                text = content.decode('utf-8')
                raw_signals = [text] # Try to parse the whole block as one signal

        elif filename.endswith(".json"):
            data = json.loads(content)
            if isinstance(data, list):
                if all(isinstance(x, (int, float)) for x in data):
                    raw_signals = [data]
                elif all(isinstance(x, list) for x in data):
                    raw_signals = data
                elif all(isinstance(x, dict) for x in data):
                    raw_signals = [item["signal"] for item in data if "signal" in item]
            elif isinstance(data, dict):
                if "signal" in data:
                    raw_signals = [data["signal"]]
            else:
                raise ValueError("Unsupported JSON structure")
        else:
            raise ValueError("Only CSV and JSON supported")

        # Validate each extracted signal
        for i, raw_sig in enumerate(raw_signals):
            try:
                valid_sig = validate_signal(raw_sig)
                results.append({"row": i, "signal": valid_sig, "error": None})
            except Exception as e:
                results.append({"row": i, "signal": None, "error": str(e)})

    except Exception as e:
        raise ValueError(f"File parsing failed: {str(e)}")

    return results

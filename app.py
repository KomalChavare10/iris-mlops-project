from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
M = ROOT / "models"
required = [M / "iris_nn.keras", M / "scaler.joblib", M / "label_encoder.joblib"]
if not all(p.exists() for p in required):
    raise RuntimeError("Model files missing. Run `python train.py` first.")
model = tf.keras.models.load_model(required[0])
scaler = joblib.load(required[1])
encoder = joblib.load(required[2])
app = FastAPI(title="Iris Flower Classifier", version="1.0")

class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Iris Classifier</title><style>
    body{font-family:Arial;background:#f2f5fb;margin:0;color:#19243b}
    main{max-width:680px;margin:40px auto;padding:20px}.card{background:white;padding:28px;border-radius:14px;box-shadow:0 6px 24px #0001}
    h1{color:#294d9b}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    label{display:block;font-weight:bold;font-size:14px;margin-bottom:6px}
    input{width:100%;box-sizing:border-box;padding:12px;border:1px solid #ccd2df;border-radius:7px}
    button{margin-top:20px;padding:12px 18px;border:0;border-radius:7px;background:#3157b7;color:white;font-size:16px;cursor:pointer}
    #result{margin-top:18px;padding:14px;background:#edf3ff;border-radius:7px;display:none}
    @media(max-width:520px){.grid{grid-template-columns:1fr}}
    </style></head><body><main><div class="card"><h1>Iris Flower Classifier</h1>
    <p>Enter flower measurements in centimeters.</p><form id="f"><div class="grid">
    <div><label>Sepal length (cm)</label><input id="sl" type="number" step="any" value="5.1" min="0.1" required></div>
    <div><label>Sepal width (cm)</label><input id="sw" type="number" step="any" value="3.5" min="0.1" required></div>
    <div><label>Petal length (cm)</label><input id="pl" type="number" step="any" value="1.4" min="0.1" required></div>
    <div><label>Petal width (cm)</label><input id="pw" type="number" step="any" value="0.2" min="0.1" required></div>
    </div><button>Predict species</button></form><div id="result"></div>
    <p><small>Basic MLOps demo: saved model, preprocessing, MLflow tracking, and API predictions.</small></p>
    </div></main><script>
    document.getElementById('f').addEventListener('submit',async e=>{
      e.preventDefault();let r=document.getElementById('result');r.style.display='block';r.textContent='Predicting...';
      let body={sepal_length:+sl.value,sepal_width:+sw.value,petal_length:+pl.value,petal_width:+pw.value};
      try{let res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      let d=await res.json();if(!res.ok)throw Error(d.detail||'Prediction failed');
      r.innerHTML='<b>Predicted species: '+d.species+'</b><br>Confidence: '+d.confidence_percent+'%';}
      catch(err){r.textContent='Error: '+err.message;}
    });
    </script></body></html>"""

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(f: FlowerInput):
    x = np.array([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]], dtype=np.float32)
    probs = model.predict(scaler.transform(x), verbose=0)[0]
    i = int(np.argmax(probs))
    return {"species": str(encoder.inverse_transform([i])[0]),
            "confidence_percent": round(float(probs[i]) * 100, 2),
            "probabilities": {str(c): round(float(p), 4) for c, p in zip(encoder.classes_, probs)}}

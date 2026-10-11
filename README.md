# Iris Flower Neural Network with Basic MLOps

## What this project demonstrates
- CSV dataset loading and validation
- Neural-network training with TensorFlow/Keras
- Train/test split and feature scaling
- MLflow experiment tracking (parameters, accuracy, and saved artifacts)
- Saved model and preprocessing files
- Local prediction web app using FastAPI
- Health endpoint at `/health`

## 1. Open this folder in VS Code
Extract the ZIP. In VS Code select **File > Open Folder** and open `iris_mlops_project`. Keep `iris_dataset.csv`, `train.py`, and `app.py` in the same folder.

## 2. Create a virtual environment (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```
If `py` is not recognized, try `python -m venv .venv`.

## 3. Train and track the model
```powershell
python train.py
```
This creates model files in `models` and logs an MLflow run locally. Accuracy can vary; do not force 100%.

## 4. View MLflow
Open a second terminal in the project folder, activate `.venv`, then run:
```powershell
mlflow ui
```
Open http://127.0.0.1:5000. Keep that terminal open while viewing the dashboard.

## 5. Run the local web app
Open another terminal in the project folder, activate `.venv`, and run:
```powershell
uvicorn app:app --reload
```
Open http://127.0.0.1:8000. Enter the four flower measurements and click **Predict species**.
API docs: http://127.0.0.1:8000/docs
Health check: http://127.0.0.1:8000/health

## 6. Compare experiments
Change a setting in `train.py` (for example neurons, epochs, or learning rate), run `python train.py` again, then compare runs in MLflow. Restart the app to load the newly saved model.

This is a basic local college-project pipeline, not a production cloud system. It does not include CI/CD or drift detection.

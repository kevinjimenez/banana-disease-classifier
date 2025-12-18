"""
API REST con FastAPI + PostgreSQL para el clasificador de enfermedades de banano.

Endpoints:
    - POST /predict - Predicción de una imagen (guarda en DB)
    - GET /history - Historial de predicciones
    - GET /history/{id} - Predicción específica
    - POST /feedback - Agregar feedback
    - GET /stats - Estadísticas del sistema
    - GET /health - Health check
    - GET /model/info - Información del modelo
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
import io
import hashlib
import time
from PIL import Image
from datetime import datetime, timedelta
from dl.predictor import BananaClassifierInference
from database import db_models
from database.connection import get_db, init_db, engine
from database import schemas
from utils import compute_image_hash

from dotenv import load_dotenv

load_dotenv()

import os

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
if not CHECKPOINT_PATH:
    raise ValueError("La variable de entorno CHECKPOINT_PATH no está definida")

DEVICE = os.getenv("DEVICE")
if not DEVICE:
    raise ValueError("La variable de entorno DEVICE no está definida")

API_HOST = os.getenv("API_HOST")
if not API_HOST:
    raise ValueError("La variable de entorno API_HOST no está definida")

API_PORT = os.getenv("API_PORT")
if not API_PORT:
    raise ValueError("La variable de entorno API_PORT no está definida")
API_PORT = int(API_PORT)

# Variables globales
predictor: Optional[BananaClassifierInference] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Maneja el ciclo de vida de la aplicación (startup y shutdown)."""
    global predictor

    # Startup: Inicializar base de datos y modelo
    try:
        print(" **** Inicializando base de datos...  ****")
        db_models.Base.metadata.create_all(bind=engine)
        print(" **** Base de datos inicializada  ****")
    except Exception as e:
        print(f"[ERROR] Error inicializando base de datos: {e}")

    try:
        print(f"**** Cargando modelo desde: {CHECKPOINT_PATH} ****")
        predictor = BananaClassifierInference(CHECKPOINT_PATH, device=DEVICE)
        print(f"**** Modelo cargado en device: {predictor.device} ****")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        raise

    yield

    # Shutdown: Limpieza si es necesaria
    print(" **** Cerrando aplicación  ****")


app = FastAPI(
    title="Banana Disease Classifier API",
    description="API REST para clasificación de enfermedades de banano",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz."""
    return {
        "message": "Banana Disease Classifier API v2.0",
        "features": ["predictions", "history", "feedback", "stats"],
        "docs": "/docs",
    }


@app.get("/health", response_model=schemas.HealthResponse, tags=["General"])
async def health_check(db: Session = Depends(get_db)):
    """Health check del servicio + database."""
    db_connected = False
    try:
        db.execute(func.now())
        db_connected = True
    except:
        pass

    return {
        "status": "healthy" if (predictor and db_connected) else "unhealthy",
        "model_loaded": predictor is not None,
        "device": predictor.device if predictor else "N/A",
        "database_connected": db_connected,
    }


@app.get("/model/info", response_model=schemas.ModelInfo, tags=["Model"])
async def get_model_info():
    """Información del modelo."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return predictor.get_model_info()


@app.post("/predict", response_model=schemas.PredictionResponse, tags=["Prediction"])
async def predict_image(
    file: UploadFile = File(...),
    return_all_probs: bool = False,
    save_to_db: bool = True,
    db: Session = Depends(get_db),
):
    """
    Realiza predicción y GUARDA en base de datos.

    Args:
        file: Imagen
        return_all_probs: Retornar todas las probabilidades
        save_to_db: Guardar en base de datos (default: True)
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Debe ser una imagen")

    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Calcular hash
        image_hash = compute_image_hash(contents)

        # Predicción
        start_time = time.time()
        result = predictor.predict(image, return_all_probs=True)
        inference_time_ms = (time.time() - start_time) * 1000

        # Guardar en base de datos
        if save_to_db:
            prediction_record = db_models.Prediction(
                image_name=file.filename,
                image_hash=image_hash,
                predicted_class=result["class"],
                confidence=result["confidence"],
                all_probabilities=result.get("all_probabilities"),
                device_used=predictor.device,
                inference_time_ms=inference_time_ms,
            )
            db.add(prediction_record)
            db.commit()
            db.refresh(prediction_record)

        return {
            "class_name": result["class"],
            "confidence": result["confidence"],
            "class_id": result["class_id"],
            "all_probabilities": (
                result.get("all_probabilities") if return_all_probs else None
            ),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get(
    "/history", response_model=List[schemas.PredictionHistoryResponse], tags=["History"]
)
async def get_history(
    skip: int = 0,
    limit: int = 100,
    class_filter: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Obtiene historial de predicciones.

    Args:
        skip: Offset
        limit: Límite de resultados
        class_filter: Filtrar por clase
    """
    query = db.query(db_models.Prediction).order_by(
        db_models.Prediction.created_at.desc()
    )

    if class_filter:
        query = query.filter(db_models.Prediction.predicted_class == class_filter)

    predictions = query.offset(skip).limit(limit).all()
    return predictions


@app.get(
    "/history/{prediction_id}",
    response_model=schemas.PredictionHistoryResponse,
    tags=["History"],
)
async def get_prediction_by_id(prediction_id: int, db: Session = Depends(get_db)):
    """Obtiene una predicción específica por ID."""
    prediction = (
        db.query(db_models.Prediction)
        .filter(db_models.Prediction.id == prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")

    return prediction


@app.post(
    "/feedback", response_model=schemas.PredictionFeedbackResponse, tags=["Feedback"]
)
async def add_feedback(
    feedback: schemas.PredictionFeedbackCreate, db: Session = Depends(get_db)
):
    """
    Agrega feedback de un usuario sobre una predicción.

    Útil para reentrenamiento y mejora del modelo.
    """
    # Verificar que la predicción existe
    prediction = (
        db.query(db_models.Prediction)
        .filter(db_models.Prediction.id == feedback.prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")

    # Crear feedback
    feedback_record = db_models.PredictionFeedback(
        prediction_id=feedback.prediction_id,
        correct_class=feedback.correct_class,
        is_correct=feedback.is_correct,
        comment=feedback.comment,
    )

    db.add(feedback_record)
    db.commit()
    db.refresh(feedback_record)

    return feedback_record


@app.get("/stats", response_model=schemas.StatsResponse, tags=["Statistics"])
async def get_stats(db: Session = Depends(get_db)):
    """Obtiene estadísticas del sistema."""

    # Total de predicciones
    total = db.query(func.count(db_models.Prediction.id)).scalar()

    # Predicciones de hoy
    today = datetime.utcnow().date()
    predictions_today = (
        db.query(func.count(db_models.Prediction.id))
        .filter(func.date(db_models.Prediction.created_at) == today)
        .scalar()
    )

    # Confianza promedio
    avg_confidence = db.query(func.avg(db_models.Prediction.confidence)).scalar() or 0.0

    # Distribución por clase
    class_dist = (
        db.query(
            db_models.Prediction.predicted_class, func.count(db_models.Prediction.id)
        )
        .group_by(db_models.Prediction.predicted_class)
        .all()
    )

    class_distribution = {cls: count for cls, count in class_dist}

    # Clase más común
    most_common = (
        max(class_distribution.items(), key=lambda x: x[1])[0]
        if class_distribution
        else "N/A"
    )

    return {
        "total_predictions": total or 0,
        "predictions_today": predictions_today or 0,
        "avg_confidence": float(avg_confidence),
        "most_common_class": most_common,
        "class_distribution": class_distribution,
    }


@app.delete("/history/{prediction_id}", tags=["History"])
async def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Elimina una predicción del historial."""
    prediction = (
        db.query(db_models.Prediction)
        .filter(db_models.Prediction.id == prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")

    db.delete(prediction)
    db.commit()

    return {"message": "Predicción eliminada", "id": prediction_id}


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handler genérico de excepciones."""
    return JSONResponse(
        status_code=500, content={"detail": f"Error interno: {str(exc)}"}
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # print("=" * 70)
    # print("Banana Disease Classifier API v2.0 - With PostgreSQL")
    # print("=" * 70)
    # print(f"Checkpoint: {CHECKPOINT_PATH}")
    # print(f"Device: {DEVICE}")
    # print("\nIniciando servidor...")
    # print("Documentación: http://localhost:8000/docs")
    # print("=" * 70)

    uvicorn.run("api.api:app", host=API_HOST, port=API_PORT, reload=True)

# 🍌 Banana Disease Classifier

Sistema de clasificación de enfermedades de banano usando Deep Learning con PyTorch y FastAPI.

## 📋 Tabla de Contenidos

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Ejecución del Backend (API)](#ejecución-del-backend-api)
- [Ejecución del Frontend](#ejecución-del-frontend)
- [Uso de la API](#uso-de-la-api)
- [Estructura del Proyecto](#estructura-del-proyecto)

## ✨ Características

- **Clasificación de 4 enfermedades**: Cordana, Healthy, Pestalotiopsis, Sigatoka
- **Arquitecturas SOTA**: ConvNeXt, EfficientNetV2, Swin Transformer, Vision Transformer
- **API REST con FastAPI**: Endpoints para predicción, historial y estadísticas
- **Base de datos PostgreSQL**: Almacenamiento de predicciones y feedback
- **Soporte Multi-GPU**: CUDA, MPS (Apple Silicon), CPU
- **Tracking de experimentos**: Weights & Biases, TensorBoard
- **Interpretabilidad**: Grad-CAM y visualizaciones

## 🔧 Requisitos

- Python 3.10+
- PostgreSQL 15+
- Docker & Docker Compose (opcional)
- GPU con CUDA 11.8+ o Apple Silicon (MPS)

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/kevinjimenez/banana-disease-classifier.git
cd banana-disease-classifier
```

### 2. Crear entorno virtual

```bash
python3 -m venv .
source bin/activate  # En Windows: Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar base de datos

#### Opción A: Con Docker (Recomendado)

```bash
docker compose up -d
```

Esto levantará PostgreSQL en el puerto 5432.

#### Opción B: PostgreSQL local

Instala PostgreSQL y crea la base de datos:

```bash
createdb banana_classifier
```

## ⚙️ Configuración

### 1. Copiar archivo de configuración

```bash
cp .env.example .env
```

### 2. Editar variables de entorno

Abre `.env` y configura:

```bash
# Base de datos
DATABASE_URL=postgresql://banana_user:banana_password@localhost:5432/banana_classifier

# Modelo
CHECKPOINT_PATH=./experiments/banana_experiment/checkpoints/best_model.pth
DEVICE=mps  # Opciones: cuda, mps, cpu

# API
API_HOST=0.0.0.0
API_PORT=8000

# Sistema
NUM_WORKERS=4
SEED=42
```

## 🎓 Entrenamiento del Modelo

### 1. Preparar el dataset

Organiza tus imágenes en la siguiente estructura:

```
dataset/
├── train/
│   ├── cordana/
│   ├── healthy/
│   ├── pestalotiopsis/
│   └── sigatoka/
├── val/
│   ├── cordana/
│   ├── healthy/
│   ├── pestalotiopsis/
│   └── sigatoka/
└── test/
    ├── cordana/
    ├── healthy/
    ├── pestalotiopsis/
    └── sigatoka/
```

### 2. Configurar el entrenamiento

Edita `deep/configs/train_config.yaml` para ajustar hiperparámetros:

```yaml
model:
  name: convnext_base
  pretrained: true
  drop_rate: 0.2

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  
data:
  train_dir: ./dataset/train
  val_dir: ./dataset/val
  test_dir: ./dataset/test
```

### 3. Ejecutar entrenamiento

```bash
# Entrenamiento básico
python deep/train.py

# Con configuración personalizada
python deep/train.py --config deep/configs/train_config.yaml

# Con Weights & Biases
python deep/train.py --use-wandb --project banana-classifier
```

### 4. Monitorear entrenamiento

#### Con TensorBoard:
```bash
tensorboard --logdir experiments/
```

#### Con Weights & Biases:
Visita: https://wandb.ai/your-username/banana-classifier

### 5. Evaluar modelo

```bash
python deep/evaluate.py \
  --checkpoint experiments/banana_experiment/checkpoints/best_model.pth \
  --test-dir dataset/test
```

## 🚀 Ejecución del Backend (API)

### 1. Verificar configuración

Asegúrate de que:
- PostgreSQL está corriendo
- El archivo `.env` está configurado
- El checkpoint del modelo existe en la ruta especificada

### 2. Iniciar el servidor

```bash
cd backend
python3 api/api.py
```

O con uvicorn directamente:

```bash
uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Verificar que el servidor está corriendo

```bash
curl http://localhost:8000/health
```

Deberías ver:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps",
  "database_connected": true
}
```

## 📡 Uso de la API

### Documentación interactiva

Visita: http://localhost:8000/docs

### Endpoints principales

#### 1. Health Check
```bash
GET http://localhost:8000/health
```

#### 2. Información del modelo
```bash
GET http://localhost:8000/model/info
```

#### 3. Realizar predicción
```bash
curl -X POST "http://localhost:8000/predict?return_all_probs=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

Respuesta:
```json
{
  "class_name": "sigatoka",
  "confidence": 0.95,
  "class_id": 3,
  "all_probabilities": {
    "cordana": 0.01,
    "healthy": 0.02,
    "pestalotiopsis": 0.02,
    "sigatoka": 0.95
  }
}
```

#### 4. Historial de predicciones
```bash
# Todas las predicciones
GET http://localhost:8000/history

# Con filtros
GET http://localhost:8000/history?skip=0&limit=10&class_filter=sigatoka
```

#### 5. Agregar feedback
```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": 1,
    "is_correct": false,
    "correct_class": "healthy",
    "comment": "La imagen estaba borrosa"
  }'
```

#### 6. Estadísticas
```bash
GET http://localhost:8000/stats
```

Respuesta:
```json
{
  "total_predictions": 150,
  "predictions_today": 25,
  "avg_confidence": 0.87,
  "most_common_class": "sigatoka",
  "class_distribution": {
    "cordana": 30,
    "healthy": 45,
    "pestalotiopsis": 25,
    "sigatoka": 50
  }
}
```

## 🖥️ Ejecución del Frontend

El frontend es una aplicación web estática que se conecta a la API del backend.

### Capturas de Pantalla

**Interfaz Principal**

![Frontend Screenshot](./frontend/screenshot.png)

**Resultado de Predicción**

![Prediction Result](./frontend/screenshot-result.png)

### 1. Asegúrate de que el backend esté corriendo

El frontend necesita que el backend esté activo en `http://localhost:8000`:

```bash
# En una terminal
cd backend
python3 api/api.py
```

### 2. Iniciar el servidor frontend

Abre una nueva terminal y ejecuta:

#### Opción A: Con Python (Recomendado)

```bash
cd frontend
python3 -m http.server 3000
```

#### Opción B: Con Node.js

```bash
cd frontend
npx http-server -p 3000
```

#### Opción C: Con Live Server (VS Code)

Si usas VS Code, instala la extensión "Live Server" y haz clic derecho en `index.html` → "Open with Live Server"

### 3. Abrir en el navegador

Visita: http://localhost:3000

### Funcionalidades del Frontend

- **Subir imagen**: Arrastra y suelta o selecciona una imagen de hoja de banano
- **Predicción en tiempo real**: Visualiza la clase predicha y el nivel de confianza
- **Métricas de modelos**: Compara ConvNeXt, ViT y YOLOv8
- **Interfaz moderna**: Diseño responsivo con tema oscuro

## 📁 Estructura del Proyecto

```
banana-disease-classifier/
├── backend/                    # API Backend
│   ├── api/
│   │   └── api.py             # FastAPI application
│   ├── database/              # Modelos y schemas de DB
│   │   ├── connection.py
│   │   ├── db_models.py
│   │   └── schemas.py
│   ├── dl/                    # Deep Learning
│   │   └── predictor/
│   │       └── banana_classifier_inference.py
│   └── utils/                 # Utilidades
│
├── frontend/                  # Interfaz Web
│   ├── index.html            # Página principal
│   ├── app.js                # Lógica de la aplicación
│   ├── styles.css            # Estilos
│   └── screenshot.png        # Captura de pantalla
│
├── deep/                      # Entrenamiento y modelos
│   ├── configs/              # Configuraciones
│   ├── core/                 # Arquitecturas y utilidades
│   │   ├── inference.py      # Modelo BananaClassifier
│   │   ├── trainer.py
│   │   └── metrics.py
│   ├── data/                 # Datasets y augmentations
│   ├── train.py              # Script de entrenamiento
│   └── evaluate.py           # Script de evaluación
│
├── dataset/                   # Datos
│   ├── train/
│   ├── val/
│   └── test/
│
├── experiments/               # Experimentos y checkpoints
│   └── banana_experiment/
│       └── checkpoints/
│           └── best_model.pth
│
├── scripts/                   # Scripts auxiliares
├── docker-compose.yml         # Docker para PostgreSQL
├── requirements.txt           # Dependencias Python
├── .env.example              # Ejemplo de configuración
└── README.md                 # Este archivo
```

## 🐛 Troubleshooting

### Error: "No module named 'fastapi'"
```bash
pip install -r requirements.txt
```

### Error: "Cannot connect to PostgreSQL"
```bash
# Verificar que PostgreSQL está corriendo
docker compose ps

# Reiniciar contenedor
docker compose restart
```

### Error: "FileNotFoundError: checkpoint not found"
Verifica que `CHECKPOINT_PATH` en `.env` apunta al archivo correcto:
```bash
ls -la experiments/banana_experiment/checkpoints/best_model.pth
```

### Error: "CUDA out of memory"
Reduce el `batch_size` en la configuración de entrenamiento.

## 📝 Licencia

MIT License

## 👥 Contribuciones

- Carolina Ortiz
- Alisson Lopez
- Kevin Jiménez
- Cristian Cobos


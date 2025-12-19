# 🍌 Banana Disease Classifier

Sistema de clasificación de enfermedades de banano usando Deep Learning con PyTorch, YOLOv8 y FastAPI.

## 📋 Tabla de Contenidos

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Entrenamiento CNN](#entrenamiento-cnn)
- [Entrenamiento YOLO](#entrenamiento-yolo)
- [Benchmark CNN vs YOLO](#benchmark-cnn-vs-yolo)
- [Evaluación de Modelos](#evaluación-de-modelos)
- [Ejecución del Backend (API)](#ejecución-del-backend-api)
- [Ejecución del Frontend](#ejecución-del-frontend)
- [Uso de la API](#uso-de-la-api)

## ✨ Características

- **Clasificación de 6 enfermedades**: Cordana, Fusarium R4T, Healthy, Moko, Pestalotiopsis, Sigatoka
- **Dos arquitecturas disponibles**:
  - **CNN**: ConvNeXt, EfficientNet (alta precisión)
  - **YOLO**: YOLOv8 Classification (rápido y ligero)
- **API REST con FastAPI**: Endpoints para predicción, historial y estadísticas
- **Base de datos PostgreSQL**: Almacenamiento de predicciones y feedback
- **Soporte Multi-GPU**: CUDA, MPS (Apple Silicon), CPU
- **Pipeline automatizado**: Análisis, limpieza y entrenamiento
- **Benchmark integrado**: Comparación automática CNN vs YOLO
- **Interpretabilidad**: Grad-CAM y visualizaciones

## 🔧 Requisitos

- Python 3.10+
- PostgreSQL 15+ (opcional, para backend)
- Docker & Docker Compose (opcional)
- GPU con CUDA 11.8+ o Apple Silicon (MPS) o CPU

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/banana-disease-classifier.git
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

### 4. Configurar base de datos (opcional, solo para backend)

#### Opción A: Con Docker (Recomendado)

```bash
docker compose up -d
```

#### Opción B: PostgreSQL local

```bash
createdb banana_classifier
```

## ⚙️ Configuración

### 1. Copiar archivo de configuración

```bash
cp .env.example .env
```

### 2. Editar variables de entorno

El usuario debe abrir el archivo \`.env\` y configurar según sus necesidades:

```bash
# Directorios
DATA_DIR=./dataset/train
TEST_DIR=./dataset/test
EXPERIMENTS_DIR=./cnn/experiments

# Entrenamiento CNN
MODEL_NAME=convnext_base
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.0001
AUGMENTATION_LEVEL=medium

# Entrenamiento YOLO
YOLO_MODEL_SIZE=s
YOLO_EPOCHS=50

# Hardware
DEVICE=auto  # auto, cuda, mps, cpu
NUM_WORKERS=4

# Pipeline
SKIP_CLEAN=false
SKIP_ANALYSIS=false
USE_TTA=true
AUTO_CONFIG=true
```

## 📂 Estructura del Proyecto

```
banana-disease-classifier/
│
├── cnn/                           # 🧠 CNN (ConvNeXt, EfficientNet)
│   ├── core/
│   │   ├── inference.py          # Modelo de inferencia
│   │   ├── trainer.py            # Entrenador
│   │   ├── dataset.py            # Dataset y dataloaders
│   │   └── transforms.py         # Augmentaciones
│   ├── config/
│   │   └── config.py             # Configuración
│   ├── train.py                  # Script de entrenamiento
│   ├── evaluate.py               # Script de evaluación
│   └── experiments/              # Resultados de experimentos
│
├── yolo/                          # ⚡ YOLO (YOLOv8 Classification)
│   ├── core/
│   │   ├── yolo_classifier.py    # Clasificador YOLO
│   │   └── yolo_utils.py         # Utilidades
│   ├── train.py                  # Script de entrenamiento
│   ├── evaluate.py               # Script de evaluación
│   └── runs/                     # Resultados de entrenamientos
│
├── backend/                       # 🌐 API REST
│   ├── api/
│   │   └── api.py                # FastAPI
│   └── dl/predictor/
│       └── banana_classifier_inference.py
│
├── frontend/                      # 💻 Interfaz Web
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── dataset/                       # 📊 Datos
│   ├── train/
│   ├── val/
│   └── test/
│
├── scripts/                       # 🛠️ Scripts generales
│   ├── benchmark.py              # Comparación CNN vs YOLO
│   ├── analyze_dataset.py        # Análisis de dataset
│   ├── clean_dataset.py          # Limpieza de dataset
│   └── check_system.py           # Verificación del sistema
│
├── train_cnn.sh                  # 🚀 Pipeline automático CNN
├── train_yolo.sh                 # 🚀 Script rápido YOLO
└── .env.example                  # Ejemplo de configuración
```

## 🎓 Entrenamiento CNN

### Opción 1: Pipeline Automático (Recomendado)

El pipeline automático ejecuta los siguientes pasos:
1. ✅ Verificación del sistema
2. 📊 Análisis del dataset
3. 🧹 Limpieza de imágenes corruptas y duplicados
4. ⚙️ Generación de configuración óptima
5. 🎯 Entrenamiento del modelo
6. 📈 Evaluación en test set

```bash
bash train_cnn.sh
```

El usuario puede personalizar el entrenamiento editando el archivo \`.env\`.

### Opción 2: Entrenamiento Manual

```bash
# Entrenamiento básico
python cnn/train.py \\
  --data_dir ./dataset/train \\
  --model convnext_base \\
  --epochs 50

# Con configuración avanzada
python cnn/train.py \\
  --data_dir ./dataset/train \\
  --model convnext_base \\
  --batch_size 64 \\
  --epochs 100 \\
  --device auto
```

### Modelos CNN Disponibles

| Modelo | Parámetros | Precisión | Velocidad | Uso |
|--------|-----------|-----------|-----------|-----|
| \`convnext_tiny\` | 28M | Alta | Rápido | **Recomendado** |
| \`convnext_base\` | 89M | Muy Alta | Lento | Máxima precisión |
| \`efficientnet_b0\` | 5M | Media-Alta | Muy rápido | Dispositivos móviles |

## ⚡ Entrenamiento YOLO

YOLOv8 ofrece un modelo más rápido y ligero, ideal para aplicaciones en tiempo real.

### Script Rápido

```bash
bash train_yolo.sh
```

### Entrenamiento Manual

```bash
python yolo/train.py --data_dir ./dataset --model_size s --epochs 50
```

### Tamaños de Modelo YOLO

| Modelo | Parámetros | Velocidad | Precisión | Uso |
|--------|-----------|-----------|-----------|-----|
| **n** (nano) | 1.4M | Muy rápido | Media | Dispositivos móviles |
| **s** (small) | 3.2M | Rápido | Media-Alta | **Recomendado** |
| **m** (medium) | 6.4M | Medio | Alta | Balance óptimo |

## 📊 Benchmark: CNN vs YOLO

El sistema permite comparar el rendimiento de diferentes arquitecturas.

```bash
python scripts/benchmark.py \\
  --data_dir ./dataset \\
  --output_dir ./benchmark_results
```

### Métricas Comparadas

- **Rendimiento**: Accuracy, Precision, Recall, F1-Score
- **Eficiencia**: Tiempo de inferencia, FPS, Memoria GPU
- **Tamaño**: Número de parámetros, Tamaño del modelo (MB)

## 📈 Evaluación de Modelos

### Evaluar CNN

```bash
python cnn/evaluate.py \\
  --checkpoint cnn/experiments/banana_experiment/checkpoints/best_model.pth \\
  --data_dir dataset/test
```

### Evaluar YOLO

```bash
python yolo/evaluate.py \\
  --model yolo/runs/yolov8s_*/weights/best.pt \\
  --data_dir dataset/test
```

## 🚀 Ejecución del Backend (API)

```bash
cd backend
python api/api.py
```

La documentación interactiva está disponible en: http://localhost:8000/docs

## 🌐 Ejecución del Frontend

```bash
cd frontend
python -m http.server 8080
```

El usuario puede visitar: http://localhost:8080

## 📡 Uso de la API

### Predicción

```bash
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@imagen.jpg"
```

## 🎯 Recomendaciones

### Para Apple Silicon (M1/M2/M3)

```bash
DEVICE=mps
NUM_WORKERS=8
```

### Para NVIDIA GPU

```bash
DEVICE=cuda
BATCH_SIZE=128
```

## 📝 Notas

- **CNN vs YOLO**: CNN ofrece mayor precisión, YOLO es más rápido y ligero
- **Pipeline Automático**: Ideal para entrenamientos completos sin intervención
- **Variables de entorno**: Toda la configuración se maneja desde el archivo \`.env\`

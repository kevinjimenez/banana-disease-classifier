#!/bin/bash
# Script para entrenar YOLOv8 para clasificación de enfermedades de banano
# Uso: bash train_yolo_quick.sh

set -e  # Salir si hay error

echo "========================================================================"
echo "ENTRENAMIENTO YOLOv8 - Clasificación de Enfermedades de Banano"
echo "========================================================================"
echo ""

# Configuración
DATA_DIR="./dataset"
MODEL_SIZE="s"  # Opciones: n, s, m, l, x
EPOCHS=50
BATCH_SIZE=32
IMAGE_SIZE=224
DEVICE="auto"  # auto, cuda, mps, cpu

# Verificar que existe el dataset
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: No se encontró el directorio $DATA_DIR"
    exit 1
fi

# Verificar que existe train/
if [ ! -d "$DATA_DIR/train" ]; then
    echo "❌ Error: No se encontró $DATA_DIR/train"
    exit 1
fi

echo "📁 Dataset: $DATA_DIR"
echo "🤖 Modelo: YOLOv8${MODEL_SIZE}-cls"
echo "📊 Épocas: $EPOCHS"
echo "📦 Batch size: $BATCH_SIZE"
echo "🖼️  Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "💻 Device: $DEVICE"
echo ""
echo "========================================================================"
echo ""

# Ejecutar entrenamiento
python scripts/train_yolo.py \
    --data_dir "$DATA_DIR" \
    --model_size "$MODEL_SIZE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --image_size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --optimizer AdamW \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --mixup 0.1 \
    --label_smoothing 0.1 \
    --patience 20

echo ""
echo "========================================================================"
echo "✅ Entrenamiento completado!"
echo "========================================================================"
echo ""
echo "📁 Resultados guardados en: runs/yolo_classify/"
echo "🔍 Para ver resultados: ls -la runs/yolo_classify/"
echo ""
echo "💡 Próximos pasos:"
echo "   1. Evaluar modelo: python scripts/evaluate_yolo.py"
echo "   2. Comparar con CNN: python scripts/benchmark.py"
echo "   3. Exportar modelo: python scripts/export_yolo.py"
echo ""

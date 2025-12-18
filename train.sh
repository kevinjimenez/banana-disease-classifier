#!/bin/bash

# ============================================================================
# PIPELINE AUTOMÁTICO COMPLETO - Clasificador de Enfermedades de Banano
# ============================================================================
#
# Este script ejecuta el pipeline completo de optimización:
# 1. Analiza el dataset
# 2. Limpia duplicados y corruptas
# 3. Genera configuración óptima
# 4. Entrena el modelo
# 5. Evalúa resultados
#
# Uso:
#   ./train.sh                      # Ejecuta todo automáticamente
#   ./train.sh --help               # Ver ayuda
#
# ============================================================================

set -e  # Salir si hay error

# Cargar variables de entorno si existe .env
if [ -f .env ]; then
    echo "📄 Cargando configuración desde .env"
    export $(grep -v '^#' .env | xargs)
fi

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir con color
print_step() {
    printf "${BLUE}========================================${NC}\n"
    printf "${BLUE}%s${NC}\n" "$1"
    printf "${BLUE}========================================${NC}\n"
}

print_success() {
    printf "${GREEN}✅ %s${NC}\n" "$1"
}

print_warning() {
    printf "${YELLOW}⚠️  %s${NC}\n" "$1"
}

print_error() {
    printf "${RED}❌ %s${NC}\n" "$1"
}

# Configuración por defecto (se puede sobrescribir con .env o argumentos)
DATA_DIR="${DATA_DIR:-./dataset/train}"
VAL_DIR="${VAL_DIR:-./dataset/val}"
TEST_DIR="${TEST_DIR:-./dataset/test}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pipeline_auto_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-./experiments}"
SKIP_CLEAN=false
DEVICE="${DEVICE:-auto}"  # auto, cuda, mps, cpu

# Crear estructura de directorios para el experimento
EXPERIMENT_DIR="$EXPERIMENTS_DIR/$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_DIR"/{checkpoints,logs,evaluation}

# Parsear argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --skip-clean)
            SKIP_CLEAN=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Uso: ./train.sh [opciones]"
            echo ""
            echo "Opciones:"
            echo "  --data_dir DIR           Directorio con datos de entrenamiento (default: ./dataset/train)"
            echo "  --experiment_name NAME   Nombre del experimento (default: pipeline_auto_TIMESTAMP)"
            echo "  --skip-clean             Saltar limpieza de dataset"
            echo "  --device DEVICE          Dispositivo: auto, cuda, mps, cpu (default: auto)"
            echo "  --help, -h               Mostrar esta ayuda"
            exit 0
            ;;
        *)
            print_error "Argumento desconocido: $1"
            echo "Usa --help para ver opciones"
            exit 1
            ;;
    esac
done

# Banner
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║       🍌 PIPELINE AUTOMÁTICO - CLASIFICADOR DE ENFERMEDADES       ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "check_system.py" ]; then
    print_error "No estás en el directorio del proyecto"
    echo "Navega a: banana_disease_classifier/"
    exit 1
fi

# Verificar que el venv está activado
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Entorno virtual no detectado"
    echo "Activando venv..."

    if [ -d "venv" ]; then
        source venv/bin/activate
        print_success "Entorno virtual activado"
    else
        print_error "No se encontró venv/"
        echo "Crea el entorno virtual primero:"
        echo "  python -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
fi

# Verificar sistema
print_step "PASO 0: Verificación del Sistema"
python3 check_system.py

if [ $? -ne 0 ]; then
    print_error "Verificación del sistema falló"
    echo "Instala las dependencias:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

print_success "Sistema verificado"
echo ""

# Verificar que existe el dataset
if [ ! -d "$DATA_DIR" ]; then
    print_error "No se encuentra el directorio: $DATA_DIR"
    echo "Crea la estructura de carpetas:"
    echo "  dataset/"
    echo "  ├── train/"
    echo "  │   ├── cordana/"
    echo "  │   ├── healthy/"
    echo "  │   ├── pestalotiopsis/"
    echo "  │   └── sigatoka/"
    echo "  ├── val/"
    echo "  └── test/"
    exit 1
fi

# ============================================================================
# PASO 1: Análisis de Dataset
# ============================================================================
print_step "PASO 1: Análisis de Dataset"
echo "Analizando: $DATA_DIR"
echo ""

python3 scripts/analyze_dataset.py \
    --data_dir "$DATA_DIR" \
    --output_dir ./dataset/analysis

if [ $? -ne 0 ]; then
    print_error "Error en análisis de dataset"
    exit 1
fi

print_success "Análisis completado"
echo ""
print_warning "REVISA las recomendaciones:"
echo "  cat ./dataset/analysis/dataset_report.txt"
echo ""

read -p "¿Continuar con la limpieza? (s/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    print_warning "Pipeline detenido por el usuario"
    exit 0
fi

# ============================================================================
# PASO 2: Limpieza de Dataset
# ============================================================================
if [ "$SKIP_CLEAN" = false ]; then
    print_step "PASO 2: Limpieza de Dataset"

    echo "Ejecutando dry-run primero..."
    python3 scripts/clean_dataset.py \
        --data_dir "$DATA_DIR" \
        --dry_run

    echo ""
    read -p "¿Ejecutar limpieza real? (s/n): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Ss]$ ]]; then
        python3 scripts/clean_dataset.py \
            --data_dir "$DATA_DIR"

        if [ $? -ne 0 ]; then
            print_error "Error en limpieza de dataset"
            exit 1
        fi

        print_success "Dataset limpiado"
        print_warning "Backup guardado en: ${DATA_DIR}_backup"
    else
        print_warning "Limpieza omitida"
    fi
else
    print_warning "PASO 2: Limpieza omitida (--skip-clean)"
fi

echo ""

# ============================================================================
# PASO 3: Generación de Configuración Óptima
# ============================================================================
print_step "PASO 3: Generación de Configuración de Entrenamiento"

# Detectar dispositivo si es auto
if [ "$DEVICE" = "auto" ]; then
    echo "🔍 Detectando dispositivo de cómputo..."

    # Verificar CUDA/MPS/CPU en un solo comando de Python
    DETECTED_DEVICE=$(python3 -c "
import torch
import sys

# Verificar en orden de preferencia
if torch.cuda.is_available():
    print('cuda', file=sys.stdout)
elif torch.backends.mps.is_available():
    print('mps', file=sys.stdout)
else:
    print('cpu', file=sys.stdout)
" 2>/dev/null)

    DEVICE="$DETECTED_DEVICE"

    # Mostrar resultado
    if [ "$DEVICE" = "cuda" ]; then
        print_success "GPU NVIDIA detectada (CUDA)"
    elif [ "$DEVICE" = "mps" ]; then
        print_success "Apple Silicon detectado (MPS)"
    else
        print_warning "Solo CPU disponible (será más lento)"
    fi
fi

echo ""
echo "Dispositivo seleccionado: $DEVICE"
echo ""

# Generar configuración de entrenamiento óptima basada en el dataset
echo "⚙️  Analizando dataset y generando configuración..."
python3 scripts/auto_config.py \
    --data_dir "$DATA_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --checkpoint_dir "$EXPERIMENT_DIR/checkpoints" \
    --no_wandb \
    --output train_config.sh

if [ $? -ne 0 ]; then
    print_error "Error generando configuración de entrenamiento"
    exit 1
fi

print_success "Configuración generada exitosamente"

# Copiar configuración al directorio del experimento
cp train_config.sh "$EXPERIMENT_DIR/train_config.sh"

echo ""
echo "📋 Configuración de entrenamiento:"
echo ""
cat train_config.sh
echo ""
echo "📁 Experimento guardado en: $EXPERIMENT_DIR"
echo ""


# ============================================================================
# PASO 4: Entrenamiento del Modelo
# ============================================================================
print_step "PASO 4: Entrenamiento del Modelo"

# Estimar tiempo
if [ "$DEVICE" = "cuda" ]; then
    ESTIMATED_TIME="35-45 minutos (50 epochs)"
elif [ "$DEVICE" = "mps" ]; then
    ESTIMATED_TIME="40-60 minutos (50 epochs)"
else
    ESTIMATED_TIME="3-4 horas (30 epochs)"
fi

echo "📊 Experimento: $EXPERIMENT_NAME"
echo "💻 Dispositivo: $DEVICE"
echo "⏱️  Tiempo estimado: $ESTIMATED_TIME"
echo ""
echo "🚀 Iniciando entrenamiento en 5 segundos..."
echo "   (Presiona Ctrl+C para cancelar)"
sleep 5

echo ""
echo "� Entrenamiento en progreso..."
echo ""

# Ejecutar el comando generado
bash train_config.sh

if [ $? -ne 0 ]; then
    print_error "Error durante entrenamiento"
    exit 1
fi

print_success "Entrenamiento completado"
echo ""

# ============================================================================
# PASO 5: Evaluación del Modelo
# ============================================================================
print_step "PASO 5: Evaluación del Modelo"

# Buscar el mejor checkpoint en el directorio del experimento
CHECKPOINT="$EXPERIMENT_DIR/checkpoints/best_model.pth"

if [ ! -f "$CHECKPOINT" ]; then
    print_error "No se encontró checkpoint del modelo en $CHECKPOINT"
    echo "Verifica que el entrenamiento haya completado correctamente"
    exit 1
fi

echo "📊 Evaluando modelo: $CHECKPOINT"
echo "📁 Test set: $TEST_DIR"
echo ""

python3 scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$TEST_DIR" \
    --device "$DEVICE" \
    --use_tta \
    --output_dir "$EXPERIMENT_DIR/evaluation"

if [ $? -ne 0 ]; then
    print_error "Error en evaluación"
    exit 1
fi

print_success "Evaluación completada exitosamente"
echo ""

# ============================================================================
# RESUMEN FINAL
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║                    ✅ PIPELINE COMPLETADO                          ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

print_success "Pipeline completado exitosamente"
echo ""
echo "📊 Resultados del experimento: $EXPERIMENT_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Directorio del experimento: $EXPERIMENT_DIR/"
echo ""
echo "Contenido:"
echo "  � Análisis del dataset:     ./dataset/analysis/"
echo "  ⚙️  Configuración:             $EXPERIMENT_DIR/train_config.sh"
echo "  🤖 Checkpoints:               $EXPERIMENT_DIR/checkpoints/"
echo "  📈 Evaluación:                $EXPERIMENT_DIR/evaluation/"
echo "  📉 Matriz de confusión:       $EXPERIMENT_DIR/evaluation/confusion_matrix.png"
echo ""
echo "💡 Revisar métricas:"
echo "  cat $EXPERIMENT_DIR/evaluation/evaluation_metrics.json"
echo ""
echo "🎯 Próximos pasos:"
echo "  • Accuracy <85%:  Revisar ./dataset/analysis/ y ajustar configuración"
echo "  • Accuracy 85-92%: Ajustar hiperparámetros o aumentar datos"
echo "  • Accuracy >92%:  ¡Modelo listo para producción!"
echo ""
echo "📦 Compartir experimento:"
echo "  tar -czf ${EXPERIMENT_NAME}.tar.gz $EXPERIMENT_DIR/"
echo ""
print_success "¡Clasificador de enfermedades de banano entrenado! 🍌✨"
echo ""

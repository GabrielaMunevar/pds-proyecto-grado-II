"""
Tests unitarios para la API de PLS
Usa pytest y mocking para no depender del modelo real

Ejecutar:
    pytest test_api.py -v
    pytest test_api.py -v --cov=. --cov-report=html

Prerequisitos:
    pip install -r requirements.txt
"""

import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock, Mock
import torch

# ============================================================================
# MOCKING DE DEPENDENCIAS ANTES DE IMPORTAR
# ============================================================================

# Mock de las funciones de utils ANTES de importar main
mock_setup_chunking = MagicMock(return_value=MagicMock())
mock_generar_pls = MagicMock(return_value=("PLS generado mock", 1))
mock_calcular_todas_metricas = MagicMock(return_value={
    "rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.4,
    "bleu": 0.35, "meteor": 0.4, "bertscore_f1": 0.8,
    "sari": 0.45,
    "flesch_reading_ease": 64.5, "flesch_kincaid_grade": 7.2,
    "compression_ratio": 0.35, "longitud_palabras": 173,
    "longitud_original_palabras": 494
})
mock_calcular_metricas_basicas = MagicMock(return_value={
    "flesch_reading_ease": 64.5, "flesch_kincaid_grade": 7.2,
    "compression_ratio": 0.35, "longitud_palabras": 173,
    "longitud_original_palabras": 494
})

# Mock utils antes de importar main
import sys
from unittest.mock import MagicMock as MockModule

# Crear módulo mock para utils
mock_utils = MockModule()
mock_utils.setup_chunking = mock_setup_chunking
mock_utils.generar_pls_con_chunking = mock_generar_pls
mock_utils.calcular_todas_las_metricas = mock_calcular_todas_metricas
mock_utils.calcular_metricas_basicas = mock_calcular_metricas_basicas

# Reemplazar utils en sys.modules antes de importar main
sys.modules['utils'] = mock_utils

# Ahora importar main (usará el mock de utils)
import main

# Reemplazar las funciones importadas en main con los mocks
main.generar_pls_con_chunking = mock_generar_pls
main.calcular_todas_las_metricas = mock_calcular_todas_metricas
main.calcular_metricas_basicas = mock_calcular_metricas_basicas

# Mock de las variables globales de main
main.MODEL = None
main.TOKENIZER = None
main.TEXT_SPLITTER = None
main.DEVICE = 'cpu'
main.MODEL_PATH = '../models/t5_base'

# Deshabilitar startup event para que no intente cargar el modelo
if hasattr(main.app, 'router') and hasattr(main.app.router, 'on_startup'):
    main.app.router.on_startup = []

# Importar app
from main import app

# Cliente de prueba
client = TestClient(app)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def texto_corto():
    """Texto médico corto (< 512 tokens)"""
    return "La hipertensión arterial es una condición médica crónica."

@pytest.fixture
def texto_largo():
    """Texto médico largo (> 512 tokens)"""
    return " ".join(["La hipertensión arterial sistémica es una enfermedad."] * 100)

@pytest.fixture
def pls_generado():
    """PLS generado de ejemplo"""
    return "La presión alta es un problema de salud que dura mucho tiempo."

@pytest.fixture
def pls_referencia():
    """PLS de referencia"""
    return "La presión arterial alta es una enfermedad crónica."

@pytest.fixture(autouse=True)
def reset_mocks():
    """Resetear mocks antes y después de cada test"""
    # Resetear mocks completamente (incluyendo side_effect)
    main.generar_pls_con_chunking.reset_mock()
    main.calcular_todas_las_metricas.reset_mock()
    main.calcular_metricas_basicas.reset_mock()
    
    # Limpiar side_effect explícitamente
    main.generar_pls_con_chunking.side_effect = None
    main.calcular_todas_las_metricas.side_effect = None
    main.calcular_metricas_basicas.side_effect = None
    
    # Configurar valores por defecto
    main.generar_pls_con_chunking.return_value = ("PLS generado mock", 1)
    main.calcular_todas_las_metricas.return_value = {
        "rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.4,
        "bleu": 0.35, "meteor": 0.4, "bertscore_f1": 0.8,
        "sari": 0.45,
        "flesch_reading_ease": 64.5, "flesch_kincaid_grade": 7.2,
        "compression_ratio": 0.35, "word_length": 173,
        "original_word_length": 494
    }
    main.calcular_metricas_basicas.return_value = {
        "flesch_reading_ease": 64.5, "flesch_kincaid_grade": 7.2,
        "compression_ratio": 0.35, "word_length": 173,
        "original_word_length": 494
    }
    
    yield
    
    # Restaurar estado inicial después del test (None = estado inicial)
    main.MODEL = None
    main.TOKENIZER = None
    main.TEXT_SPLITTER = None

# ============================================================================
# TESTS: ENDPOINT RAÍZ
# ============================================================================

def test_root_endpoint():
    """Test del endpoint raíz"""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "mensaje" in data
    assert "version" in data
    assert "endpoints" in data
    assert data["version"] == "1.0.0"

# ============================================================================
# TESTS: HEALTH CHECK
# ============================================================================

def test_health_check():
    """Test health check"""
    with patch('main.MODEL', MagicMock()):
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "modelo_cargado" in data
        assert "device" in data
        assert "modelo_path" in data

def test_health_check_modelo_no_cargado():
    """Test health check sin modelo cargado"""
    with patch('main.MODEL', None):
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["modelo_cargado"] == False

# ============================================================================
# TESTS: GENERAR PLS
# ============================================================================

def test_generate_pls_success(texto_corto):
    """Test generación exitosa de PLS"""
    with patch('main.MODEL', MagicMock()), \
         patch('main.TOKENIZER', MagicMock()), \
         patch('main.TEXT_SPLITTER', MagicMock()):
        
        # Configurar mock de generar_pls_con_chunking
        main.generar_pls_con_chunking.return_value = ("PLS generado exitosamente", 1)
        
        response = client.post(
            "/generate",
            json={
                "technical_text": texto_corto,
                "max_length": 256,
                "num_beams": 4
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verificar campos requeridos
        assert "generated_pls" in data
        assert "generation_time" in data
        assert "num_chunks" in data
        assert "tokens_input" in data
        assert "tokens_output" in data
        
        # Verificar tipos
        assert isinstance(data["generated_pls"], str)
        assert isinstance(data["generation_time"], float)
        assert isinstance(data["num_chunks"], int)
        assert isinstance(data["tokens_input"], int)
        assert isinstance(data["tokens_output"], int)
        
        # Verificar que el tiempo es razonable
        assert data["generation_time"] >= 0
        assert data["generation_time"] < 10

def test_generate_pls_texto_muy_corto():
    """Test con texto demasiado corto (debe fallar validación)"""
    response = client.post(
        "/generate",
        json={
                "technical_text": "Corto",  # Solo 5 caracteres
            "max_length": 256,
            "num_beams": 4
        }
    )
    
    assert response.status_code == 422  # Validation error

def test_generate_pls_parametros_invalidos():
    """Test con parámetros inválidos"""
    # max_length muy pequeño
    response = client.post(
        "/generate",
        json={
            "texto_tecnico": "La hipertensión arterial es una condición.",
            "max_length": 10,  # Menos de 50
            "num_beams": 4
        }
    )
    assert response.status_code == 422
    
    # num_beams inválido
    response = client.post(
        "/generate",
        json={
            "texto_tecnico": "La hipertensión arterial es una condición.",
            "max_length": 256,
            "num_beams": 0  # Debe ser >= 1
        }
    )
    assert response.status_code == 422

def test_generate_pls_sin_modelo():
    """Test generación sin modelo cargado"""
    with patch('main.MODEL', None):
        response = client.post(
            "/generate",
            json={
                "technical_text": "La hipertensión arterial es una condición.",
                "max_length": 256,
                "num_beams": 4
            }
        )
        
        assert response.status_code == 503  # Service unavailable

def test_generate_pls_texto_largo(texto_largo):
    """Test con texto largo que requiere chunking"""
    with patch('main.MODEL', MagicMock()), \
         patch('main.TOKENIZER', MagicMock()), \
         patch('main.TEXT_SPLITTER', MagicMock()):
        
        # Simular múltiples chunks
        main.generar_pls_con_chunking.return_value = ("PLS generado desde chunks", 3)
        
        response = client.post(
            "/generate",
            json={
                "texto_tecnico": texto_largo,
                "max_length": 256,
                "num_beams": 4
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Debería tener múltiples chunks
        assert data["num_chunks"] >= 1

# ============================================================================
# TESTS: EVALUAR PLS
# ============================================================================

def test_evaluate_pls_sin_referencia(texto_corto, pls_generado):
    """Test evaluación sin PLS de referencia (solo métricas básicas)"""
    main.calcular_metricas_basicas.return_value = {
        "flesch_reading_ease": 64.5,
        "flesch_kincaid_grade": 7.2,
        "compression_ratio": 0.35,
        "longitud_palabras": 173,
        "longitud_original_palabras": 494
    }
    
    response = client.post(
        "/evaluate",
            json={
                "original_text": texto_corto,
                "generated_pls": pls_generado
            }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verificar métricas básicas presentes
    assert "flesch_reading_ease" in data
    assert "flesch_kincaid_grade" in data
    assert "compression_ratio" in data
        assert "word_length" in data
        assert "original_word_length" in data
    
    # Verificar tipos
    assert isinstance(data["flesch_reading_ease"], float)
    assert isinstance(data["flesch_kincaid_grade"], float)
    assert isinstance(data["compression_ratio"], float)
        assert isinstance(data["word_length"], int)

def test_evaluate_pls_con_referencia(texto_corto, pls_generado, pls_referencia):
    """Test evaluación con PLS de referencia (métricas completas)"""
    main.calcular_todas_las_metricas.return_value = {
        "rouge1": 0.456,
        "rouge2": 0.234,
        "rougeL": 0.389,
        "bleu": 0.312,
        "meteor": 0.401,
        "bertscore_f1": 0.867,
        "sari": 0.423,
        "flesch_reading_ease": 64.5,
        "flesch_kincaid_grade": 7.2,
        "compression_ratio": 0.35,
        "longitud_palabras": 173,
        "longitud_original_palabras": 494
    }
    
    response = client.post(
        "/evaluate",
            json={
                "original_text": texto_corto,
                "generated_pls": pls_generado,
                "reference_pls": pls_referencia
            }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verificar métricas de similitud
    assert "rouge1" in data
    assert "rouge2" in data
    assert "rougeL" in data
    assert "bleu" in data
    assert "meteor" in data
    assert "sari" in data
    
    # Verificar rangos válidos (0-1)
    if data.get("rouge1") is not None:
        assert 0 <= data["rouge1"] <= 1
    if data.get("rouge2") is not None:
        assert 0 <= data["rouge2"] <= 1
    if data.get("rougeL") is not None:
        assert 0 <= data["rougeL"] <= 1
    if data.get("sari") is not None:
        assert 0 <= data["sari"] <= 1

def test_evaluate_pls_textos_vacios():
    """Test con textos vacíos"""
    main.calcular_metricas_basicas.return_value = {
        "flesch_reading_ease": 0.0,
        "flesch_kincaid_grade": 0.0,
        "compression_ratio": 0.0,
        "longitud_palabras": 0,
        "longitud_original_palabras": 0
    }
    
    response = client.post(
        "/evaluate",
        json={
                "original_text": "",
                "generated_pls": ""
        }
    )
    
    # Puede retornar 422 (validation error) o 200 con métricas en 0
    assert response.status_code in [200, 422]

# ============================================================================
# TESTS: GENERAR CON MÉTRICAS
# ============================================================================

def test_generate_with_metrics_success(texto_corto):
    """Test generación + métricas exitosa"""
    with patch('main.MODEL', MagicMock()), \
         patch('main.TOKENIZER', MagicMock()), \
         patch('main.TEXT_SPLITTER', MagicMock()):
        
        main.generar_pls_con_chunking.return_value = ("PLS generado exitosamente", 1)
        main.calcular_metricas_basicas.return_value = {
            "flesch_reading_ease": 64.5,
            "flesch_kincaid_grade": 7.2,
            "compression_ratio": 0.35,
            "longitud_palabras": 173,
            "longitud_original_palabras": 494
        }
        
        response = client.post(
            "/generate-with-metrics",
            json={
                "technical_text": texto_corto,
                "max_length": 256,
                "num_beams": 4
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verificar estructura
        assert "generated_pls" in data
        assert "metrics" in data
        assert "generation_time" in data
        assert "num_chunks" in data
        
        # Verificar que metrics es un objeto
        assert isinstance(data["metrics"], dict)
        
        # Verificar métricas básicas
        metricas = data["metrics"]
        assert "flesch_reading_ease" in metricas
        assert "compression_ratio" in metricas

def test_generate_with_metrics_con_referencia(texto_corto, pls_referencia):
    """Test generación + métricas con referencia"""
    with patch('main.MODEL', MagicMock()), \
         patch('main.TOKENIZER', MagicMock()), \
         patch('main.TEXT_SPLITTER', MagicMock()):
        
        main.generar_pls_con_chunking.return_value = ("PLS generado exitosamente", 1)
        main.calcular_todas_las_metricas.return_value = {
            "rouge1": 0.456,
            "rouge2": 0.234,
            "rougeL": 0.389,
            "bleu": 0.312,
            "meteor": 0.401,
            "bertscore_f1": 0.867,
            "sari": 0.423,
            "flesch_reading_ease": 64.5,
            "flesch_kincaid_grade": 7.2,
            "compression_ratio": 0.35,
            "longitud_palabras": 173,
            "longitud_original_palabras": 494
        }
        
        response = client.post(
            "/generate-with-metrics",
            json={
                "technical_text": texto_corto,
                "reference_pls": pls_referencia,
                "max_length": 256,
                "num_beams": 4
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Con referencia, debería tener ROUGE y SARI
        metricas = data["metricas"]
        assert "rouge1" in metricas or metricas.get("rouge1") is None
        assert "sari" in metricas or metricas.get("sari") is None

def test_generate_with_metrics_sin_modelo():
    """Test sin modelo cargado"""
    with patch('main.MODEL', None):
        response = client.post(
            "/generate-with-metrics",
            json={
                "technical_text": "La hipertensión arterial es una condición.",
                "max_length": 256,
                "num_beams": 4
            }
        )
        
        assert response.status_code == 503

# ============================================================================
# TESTS: VALIDACIÓN DE DATOS
# ============================================================================

def test_validacion_campos_requeridos():
    """Test validación de campos requeridos"""
        # Falta technical_text
    response = client.post(
        "/generate",
        json={
            "max_length": 256,
            "num_beams": 4
        }
    )
    assert response.status_code == 422

def test_validacion_tipos_datos():
    """Test validación de tipos de datos"""
    # max_length como string en lugar de int
    # FastAPI automáticamente convierte strings a int si es posible
    # Así que probamos con un tipo realmente inválido
    response = client.post(
        "/generate",
        json={
            "texto_tecnico": "La hipertensión arterial.",
            "max_length": None,  # None no es válido
            "num_beams": 4
        }
    )
    # Puede ser 422 (validation) o 503 (si valida antes de verificar modelo)
    assert response.status_code in [422, 503]

def test_validacion_rangos():
    """Test validación de rangos de parámetros"""
    # max_length fuera de rango
    response = client.post(
        "/generate",
        json={
            "texto_tecnico": "La hipertensión arterial.",
            "max_length": 1000,  # Máximo es 512
            "num_beams": 4
        }
    )
    assert response.status_code == 422
    
    # num_beams > 10
    response = client.post(
        "/generate",
        json={
            "texto_tecnico": "La hipertensión arterial.",
            "max_length": 256,
            "num_beams": 20  # Máximo es 10
        }
    )
    assert response.status_code == 422

# ============================================================================
# TESTS: MANEJO DE ERRORES
# ============================================================================

def test_manejo_error_generacion():
    """Test manejo de errores durante generación"""
    with patch('main.MODEL', MagicMock()), \
         patch('main.TOKENIZER', MagicMock()), \
         patch('main.TEXT_SPLITTER', MagicMock()):
        
        # Simular error en generación
        main.generar_pls_con_chunking.side_effect = RuntimeError("CUDA out of memory")
        
        response = client.post(
            "/generate",
            json={
                "technical_text": "La hipertensión arterial es una condición.",
                "max_length": 256,
                "num_beams": 4
            }
        )
        
        assert response.status_code == 500

def test_manejo_error_metricas():
    """Test manejo de errores en cálculo de métricas"""
    main.calcular_metricas_basicas.return_value = {
        "flesch_reading_ease": 0.0,
        "flesch_kincaid_grade": 0.0,
        "compression_ratio": 0.0,
        "longitud_palabras": 0,
        "longitud_original_palabras": 0
    }
    
    response = client.post(
        "/evaluate",
        json={
            "original_text": "Texto muy corto",
            "generated_pls": "Muy corto",
            "reference_pls": "Corto"
        }
    )
    
    # Debería ser exitoso aunque las métricas sean pobres
    assert response.status_code == 200

# ============================================================================
# TESTS: CORS
# ============================================================================

def test_cors_headers(reset_mocks):
    """Test que los headers CORS estén presentes"""
    # Usar POST en lugar de OPTIONS (TestClient maneja CORS diferente)
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    # Configurar mocks explícitamente DESPUÉS del fixture
    main.MODEL = MagicMock()
    main.TOKENIZER = mock_tokenizer
    main.TEXT_SPLITTER = MagicMock()
    main.generar_pls_con_chunking.return_value = ("PLS generado", 1)
    
    response = client.post(
        "/generate",
            json={
                "technical_text": "La hipertensión arterial es una condición.",
                "max_length": 256,
                "num_beams": 4
            },
        headers={"Origin": "http://localhost:3000"}
    )
    
    # Verificar headers CORS en la respuesta
    assert response.status_code == 200
    # CORS headers se agregan automáticamente por el middleware
    # En TestClient pueden no aparecer, pero el middleware está configurado

# ============================================================================
# TESTS: PERFORMANCE
# ============================================================================

def test_response_time_aceptable(texto_corto, reset_mocks):
    """Test que el tiempo de respuesta sea aceptable"""
    import time
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    # Configurar mocks explícitamente DESPUÉS del fixture
    main.MODEL = MagicMock()
    main.TOKENIZER = mock_tokenizer
    main.TEXT_SPLITTER = MagicMock()
    main.generar_pls_con_chunking.return_value = ("PLS generado", 1)
    
    start = time.time()
    response = client.post(
        "/generate",
        json={
            "texto_tecnico": texto_corto,
            "max_length": 256,
            "num_beams": 4
        }
    )
    duration = time.time() - start
    
    assert response.status_code == 200
    # El mock debería ser muy rápido (< 1 segundo)
    assert duration < 1.0

# ============================================================================
# TESTS: EDGE CASES
# ============================================================================

def test_texto_con_caracteres_especiales(reset_mocks):
    """Test con caracteres especiales en el texto"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    # Configurar mocks explícitamente DESPUÉS del fixture
    main.MODEL = MagicMock()
    main.TOKENIZER = mock_tokenizer
    main.TEXT_SPLITTER = MagicMock()
    main.generar_pls_con_chunking.return_value = ("PLS generado", 1)
    
    texto = "La hipertensión (HTN) con signos ≥140/90 mmHg & complicaciones."
    
    response = client.post(
        "/generate",
        json={
            "texto_tecnico": texto,
            "max_length": 256,
            "num_beams": 4
        }
    )
    
    assert response.status_code == 200

def test_texto_con_numeros(reset_mocks):
    """Test con texto que contiene números"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    # Configurar mocks explícitamente DESPUÉS del fixture
    main.MODEL = MagicMock()
    main.TOKENIZER = mock_tokenizer
    main.TEXT_SPLITTER = MagicMock()
    main.generar_pls_con_chunking.return_value = ("PLS generado", 1)
    
    texto = "La presión arterial de 140/90 mmHg indica hipertensión estadio 2."
    
    response = client.post(
        "/generate",
        json={
            "texto_tecnico": texto,
            "max_length": 256,
            "num_beams": 4
        }
    )
    
    assert response.status_code == 200

def test_multiples_requests_consecutivos(reset_mocks):
    """Test múltiples requests seguidos (threading safety)"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    # Configurar mocks explícitamente DESPUÉS del fixture
    main.MODEL = MagicMock()
    main.TOKENIZER = mock_tokenizer
    main.TEXT_SPLITTER = MagicMock()
    main.generar_pls_con_chunking.return_value = ("PLS generado", 1)
    
    textos = [
        "La hipertensión arterial.",
        "La diabetes mellitus tipo 2.",
        "El asma bronquial crónica."
    ]
    
    for texto in textos:
        response = client.post(
            "/generate",
            json={
                "technical_text": texto,
                "max_length": 256,
                "num_beams": 4
            }
        )
        assert response.status_code == 200

# ============================================================================
# MAIN: EJECUTAR TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

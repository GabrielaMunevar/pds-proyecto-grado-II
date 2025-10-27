# Scripts de Análisis

Scripts útiles para analizar y evaluar el proyecto mientras el modelo se entrena.

##  Análisis de Pares Sintéticos

**Script:** `analyze_synthetic_pairs.py`

**Propósito:** Analizar la calidad de los pares sintéticos antes del entrenamiento.

**Ejecución:**
```bash
python src/analysis/analyze_synthetic_pairs.py
```

**Qué analiza:**
-  Ratio de compresión (target: 0.3-0.8)
-  Overlap léxico (target: <0.7)
-  Diversidad de palabras
-  Cambios específicos de palabras
-  Muestra 5 ejemplos aleatorios

**Salida esperada:**
```
=== ANÁLISIS DE COMPRESIÓN ===
Ratio promedio: 0.XXX
...

=== PROBLEMAS DETECTADOS ===
  Overlap léxico muy alto (>0.8)
```

##  Comparación de Modelos

**Script:** `src/models/compare_models.py`

**Propósito:** Comparar modelo baseline vs modelo mejorado.

**Ejecución:**
```bash
python src/models/compare_models.py
```

**Qué compara:**
- Longitud de predicciones vs ground truth
- Conteo de palabras
- Diferencia promedio

**Requiere:**
- Modelo baseline en `models/t5_generator/model`
- Modelo mejorado en `models/t5_generator/final_model` (después de entrenar)

##  Workflow Recomendado

### Mientras esperas que termine el entrenamiento:

1. **Analizar datos actuales:**
   ```bash
   python src/analysis/analyze_synthetic_pairs.py
   ```

2. **Revisar problemas encontrados:**
   - Si ratio > 0.9: Pares muy similares
   - Si overlap > 0.8: Cambios insuficientes
   - Decidir si necesitas regenerar pares

3. **Preparar para descargar modelo:**
   - Verificar espacio en disco
   - Preparar ruta de destino local

### Después del entrenamiento:

1. **Descargar modelo desde Colab:**
   ```python
   # En Colab
   from google.colab import files
   files.download('/content/models/t5_generator/final_model')
   ```

2. **Copiar modelo a repositorio:**
   ```bash
   # En local
   cp final_model/* models/t5_generator/final_model/
   ```

3. **Comparar modelos:**
   ```bash
   python src/models/compare_models.py
   ```

4. **Ejecutar evaluación completa:**
   ```bash
   python src/models/evaluate_generator.py
   ```

##  Checklist Post-Entrenamiento

- [ ] Descargar modelo desde Colab
- [ ] Copiar a `models/t5_generator/final_model/`
- [ ] Ejecutar `compare_models.py`
- [ ] Ejecutar `evaluate_generator.py`
- [ ] Comparar métricas con baseline
- [ ] Verificar que ROUGE-L ≥ 0.42

##  Troubleshooting

### Error: Modelo no encontrado
```bash
# Verificar que el modelo esté en la ruta correcta
ls models/t5_generator/final_model/
```

### Error: Out of memory
```bash
# Reducir número de ejemplos en el script
test_examples = load_test_examples(n=10)  # Reducir de 20 a 10
```

##  Interpretación de Resultados

### Ratio de compresión
- **> 0.9**: Muy alto (textos casi idénticos)
- **0.3-0.8**: Bueno - **< 0.3**: Muy bajo (simplifica demasiado)

### Overlap léxico
- **> 0.8**: Muy alto (pocos cambios)
- **0.5-0.8**: Aceptable
- **< 0.5**: Bajo (muchos cambios, posible pérdida de información)

### Mejora en modelo
- **> 0**: Mejor que baseline - **< 0**: Peor que baseline 

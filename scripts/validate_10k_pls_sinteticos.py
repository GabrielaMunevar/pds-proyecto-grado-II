"""
VALIDACIÓN COMPLETA DE PLS SINTÉTICOS
=====================================

Este script valida los PLS sintéticos generados con GPT-4o-mini.
Soporta validación de 10K, 20K o cualquier cantidad de pares.

El script realiza:
1. Validación automática (FRE, FKG, longitud)
2. Validación con clasificador (probabilidad PLS)
3. Métricas de simplificación (ROUGE & SARI)
4. Comparación estadística con PLS reales Cochrane
5. Filtrado por calidad (cuádruple criterio)
6. Generación de reportes y visualizaciones

Métricas incluidas:
- FRE (Flesch Reading Ease): Legibilidad del texto
- FKG (Flesch-Kincaid Grade): Nivel de grado escolar
- ROUGE (1, 2, L): Overlap de n-gramas con texto original
- SARI: Métrica específica para evaluar simplificación
- Clasificador: Probabilidad de ser PLS real

Uso:
    python scripts/validate_10k_pls_sinteticos.py

Dependencias adicionales:
    pip install rouge-score easse

Archivo de entrada:
    - data/synthetic_pls/pls_produccion_10k.csv (ruta por defecto)
    - data/synthetic_pls/pls_produccion_20k_v7.csv (20K pares)
    - Cualquier archivo CSV con columnas 'texto_original' y 'pls_generado'

Salidas:
    - pls_validado.csv (todos con métricas)
    - pls_final_aprobados.csv (solo los que pasan filtros)
    - validacion_report.txt (reporte completo)
    - validacion_distribucion.png (gráficas con 9 paneles)
    - validacion_metrics.json (métricas en JSON)

Tiempo estimado: 15-20 minutos (10K), 30-40 minutos (20K)
"""

import pandas as pd
import numpy as np
import joblib
import textstat
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
from rouge_score import rouge_scorer
from easse.sari import corpus_sari
from scipy.sparse import hstack

# Intentar importar ReadabilityFeatureExtractor si está disponible
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.train_classifier import ReadabilityFeatureExtractor
    READABILITY_EXTRACTOR_AVAILABLE = True
except ImportError:
    READABILITY_EXTRACTOR_AVAILABLE = False
    ReadabilityFeatureExtractor = None

warnings.filterwarnings('ignore')

# Configuración de visualización
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)


def convertir_a_json_serializable(obj):
    """
    Convierte valores numpy/pandas a tipos nativos de Python para JSON.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convertir_a_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convertir_a_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

class ValidadorPLSSinteticos:
    """
    Validador completo de PLS sintéticos generados con LLM.
    
    Implementa validación multi-nivel:
    - Nivel 1: Métricas automáticas (FRE, FKG, longitud)
    - Nivel 2: Clasificador entrenado (probabilidad PLS)
    - Nivel 2.5: Métricas de simplificación (ROUGE, SARI)
    - Nivel 3: Comparación estadística con PLS reales
    """
    
    def __init__(self, csv_path=None):
        """
        Args:
            csv_path: Ruta al CSV con PLS sintéticos generados. Si es None, busca automáticamente.
        """
        if csv_path is None:
            # Buscar archivo automáticamente
            posibles_archivos = [
                'data/synthetic_pls/pls_produccion_20k_v7.csv',  # 20K pares
                'data/synthetic_pls/pls_produccion_10k.csv',  # 10K pares
                'pls_produccion_20k_v7.csv',
                'pls_produccion_10k.csv',
                'pls_20k.csv',
                'pls_10k.csv',
                'pls_sinteticos_20k.csv',
                'pls_sinteticos_10k.csv'
            ]
            
            csv_path = None
            for archivo in posibles_archivos:
                if Path(archivo).exists():
                    csv_path = archivo
                    break
            
            if csv_path is None:
                raise FileNotFoundError(
                    f"No se encontró archivo de PLS sintéticos.\n"
                    f"Buscados: {posibles_archivos}\n"
                    f"Asegúrate de que el archivo esté en el directorio correcto."
                )
        
        self.csv_path = csv_path
        self.df = None
        self.clf = None
        self.vectorizer = None
        self.readability_extractor = None
        self.pls_reales_probs = None
        self.resultados = {}
        
        print("="*80, flush=True)
        print("VALIDADOR DE PLS SINTETICOS", flush=True)
        print("="*80, flush=True)
        print(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"Archivo: {csv_path}", flush=True)
        
    def cargar_datos(self):
        """Carga los PLS sintéticos generados."""
        print("\n CARGANDO DATOS...")
        print("-"*80)
        
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(
                f" No se encontró {self.csv_path}\n"
                f"   Asegúrate de que la generación haya terminado."
            )
        
        self.df = pd.read_csv(self.csv_path)
        
        print(f" Cargados: {len(self.df):,} PLS sintéticos")
        
        # Verificar columnas requeridas
        required_cols = ['texto_original', 'pls_generado']
        missing = [col for col in required_cols if col not in self.df.columns]
        
        if missing:
            raise ValueError(f" Faltan columnas: {missing}")
        
        # Mapear columnas si tienen nombres diferentes
        if 'flesch_reading_ease' in self.df.columns and 'fre' not in self.df.columns:
            self.df['fre'] = self.df['flesch_reading_ease']
        if 'flesch_kincaid_grade' in self.df.columns and 'fkg' not in self.df.columns:
            self.df['fkg'] = self.df['flesch_kincaid_grade']
        
        # Estadísticas básicas
        if 'fre' in self.df.columns:
            print(f"   FRE medio: {self.df['fre'].mean():.1f}")
        if 'fkg' in self.df.columns:
            print(f"   FKG medio: {self.df['fkg'].mean():.1f}")
        if 'longitud_pls' in self.df.columns:
            print(f"   Longitud media: {self.df['longitud_pls'].mean():.0f} palabras")
        
        return self.df
    
    def cargar_clasificador(self):
        """Carga el clasificador entrenado."""
        print("\n CARGANDO CLASIFICADOR...")
        print("-"*80)
        
        model_dir = Path('models/baseline_classifier')
        
        if not model_dir.exists():
            print("  No se encontró clasificador entrenado")
            print("   Ejecuta primero: python src/models/train_classifier.py")
            return False
        
        # Importar ReadabilityFeatureExtractor ANTES de cargar el pickle
        # Esto es necesario para que joblib pueda deserializar el objeto
        readability_path = model_dir / 'readability_extractor.pkl'
        if readability_path.exists():
            try:
                # Importar la clase antes de cargar
                import sys
                script_path = Path(__file__).parent.parent
                sys.path.insert(0, str(script_path))
                from src.models.train_classifier import ReadabilityFeatureExtractor
                # Registrar la clase en el módulo actual para joblib
                import scripts.validate_10k_pls_sinteticos as current_module
                current_module.ReadabilityFeatureExtractor = ReadabilityFeatureExtractor
            except ImportError as e:
                print(f"  No se pudo importar ReadabilityFeatureExtractor: {e}")
                print("  El clasificador requiere features de legibilidad pero no se pueden cargar")
                return False
        
        try:
            self.vectorizer = joblib.load(model_dir / 'vectorizer.pkl')
            self.clf = joblib.load(model_dir / 'classifier.pkl')
            
            # Cargar extractor de features de legibilidad si existe
            if readability_path.exists():
                try:
                    self.readability_extractor = joblib.load(readability_path)
                    print(" Clasificador cargado correctamente (con features de legibilidad)")
                except Exception as e:
                    print(f"  Error cargando extractor de legibilidad: {e}")
                    print("  El clasificador requiere estas features, no se puede continuar")
                    return False
            else:
                self.readability_extractor = None
                print(" Clasificador cargado correctamente (solo TF-IDF)")
            
            # Cargar test set para comparación
            try:
                import pickle
                with open('models/baseline_classifier/test_set.pkl', 'rb') as f:
                    test_data = pickle.load(f)
                
                X_test = test_data['X_test']
                y_test = test_data['y_test']
                
                # Transformar test set con el mismo pipeline que se usó en entrenamiento
                X_test_tfidf = self.vectorizer.transform(X_test)
                if self.readability_extractor is not None:
                    # Necesitamos recrear el extractor para el test set
                    # Por ahora, saltamos la comparación si no tenemos el extractor
                    print("  Test set requiere extractor de legibilidad, skip comparación")
                    self.pls_reales_probs = None
                else:
                    # Calcular probabilidades de PLS reales
                    probs = self.clf.predict_proba(X_test_tfidf)[:, 1]
                    self.pls_reales_probs = probs[np.array(y_test) == 1]
                    print(f" Cargados {len(self.pls_reales_probs)} PLS reales para comparación")
                
            except Exception as e:
                print(f"  No se pudo cargar test set: {e}")
                self.pls_reales_probs = None
            
            return True
            
        except Exception as e:
            print(f" Error cargando clasificador: {e}")
            return False
    
    def validacion_automatica(self):
        """
        Nivel 1: Validación con métricas automáticas.
        """
        print("\n" + "="*80)
        print("NIVEL 1: VALIDACIÓN AUTOMÁTICA (Métricas de Legibilidad)")
        print("="*80)
        
        # Si ya tienen métricas, skip
        if 'fre' in self.df.columns and 'fkg' in self.df.columns:
            print(" Métricas ya calculadas en el CSV")
        else:
            print(" Calculando métricas de legibilidad...")
            
            fre_scores = []
            fkg_scores = []
            avg_sent_lengths = []
            
            for pls in self.df['pls_generado']:
                if pd.notna(pls):
                    fre_scores.append(textstat.flesch_reading_ease(str(pls)))
                    fkg_scores.append(textstat.flesch_kincaid_grade(str(pls)))
                    avg_sent_lengths.append(textstat.avg_sentence_length(str(pls)))
                else:
                    fre_scores.append(np.nan)
                    fkg_scores.append(np.nan)
                    avg_sent_lengths.append(np.nan)
            
            self.df['fre'] = fre_scores
            self.df['fkg'] = fkg_scores
            self.df['avg_sentence_length'] = avg_sent_lengths
            
            print(" Métricas calculadas")
        
        # Calcular longitud si no existe
        if 'longitud_pls' not in self.df.columns:
            self.df['longitud_pls'] = self.df['pls_generado'].str.split().str.len()
        
        # Estadísticas
        print("\n RESULTADOS MÉTRICAS AUTOMÁTICAS:")
        print("-"*80)
        
        metricas = {
            'fre': {
                'media': self.df['fre'].mean(),
                'mediana': self.df['fre'].median(),
                'std': self.df['fre'].std(),
                'min': self.df['fre'].min(),
                'max': self.df['fre'].max(),
                'target': '60-70',
                'cumple': (self.df['fre'] >= 55).sum()
            },
            'fkg': {
                'media': self.df['fkg'].mean(),
                'mediana': self.df['fkg'].median(),
                'std': self.df['fkg'].std(),
                'min': self.df['fkg'].min(),
                'max': self.df['fkg'].max(),
                'target': '7-9',
                'cumple': (self.df['fkg'] <= 10).sum()
            },
            'longitud': {
                'media': self.df['longitud_pls'].mean(),
                'mediana': self.df['longitud_pls'].median(),
                'std': self.df['longitud_pls'].std(),
                'min': self.df['longitud_pls'].min(),
                'max': self.df['longitud_pls'].max(),
                'target': '150-250',
                'cumple': ((self.df['longitud_pls'] >= 150) & 
                          (self.df['longitud_pls'] <= 250)).sum()
            }
        }
        
        # Mostrar resultados
        print("\n1. FLESCH READING EASE (target: 60-70, mínimo: 55)")
        print(f"   Media:   {metricas['fre']['media']:.1f}")
        print(f"   Mediana: {metricas['fre']['mediana']:.1f}")
        print(f"   Std:     {metricas['fre']['std']:.1f}")
        print(f"   Rango:   [{metricas['fre']['min']:.1f}, {metricas['fre']['max']:.1f}]")
        print(f"    Cumplen FRE≥55: {metricas['fre']['cumple']:,}/{len(self.df):,} "
              f"({metricas['fre']['cumple']/len(self.df)*100:.1f}%)")
        
        print("\n2. FLESCH-KINCAID GRADE (target: 7-9, máximo: 10)")
        print(f"   Media:   {metricas['fkg']['media']:.1f}")
        print(f"   Mediana: {metricas['fkg']['mediana']:.1f}")
        print(f"   Std:     {metricas['fkg']['std']:.1f}")
        print(f"   Rango:   [{metricas['fkg']['min']:.1f}, {metricas['fkg']['max']:.1f}]")
        print(f"    Cumplen FKG≤10: {metricas['fkg']['cumple']:,}/{len(self.df):,} "
              f"({metricas['fkg']['cumple']/len(self.df)*100:.1f}%)")
        
        print("\n3. LONGITUD (target: 150-250 palabras)")
        print(f"   Media:   {metricas['longitud']['media']:.0f} palabras")
        print(f"   Mediana: {metricas['longitud']['mediana']:.0f} palabras")
        print(f"   Std:     {metricas['longitud']['std']:.0f}")
        print(f"   Rango:   [{metricas['longitud']['min']:.0f}, {metricas['longitud']['max']:.0f}]")
        print(f"    Cumplen 150-250: {metricas['longitud']['cumple']:,}/{len(self.df):,} "
              f"({metricas['longitud']['cumple']/len(self.df)*100:.1f}%)")
        
        # Aprobados (triple criterio)
        aprobados_auto = self.df[
            (self.df['fre'] >= 55) &
            (self.df['fkg'] <= 10) &
            (self.df['longitud_pls'] >= 150) &
            (self.df['longitud_pls'] <= 250)
        ]
        
        print(f"\n APROBADOS (triple criterio): {len(aprobados_auto):,}/{len(self.df):,} "
              f"({len(aprobados_auto)/len(self.df)*100:.1f}%)")
        
        self.resultados['metricas_automaticas'] = metricas
        self.resultados['aprobados_automaticos'] = len(aprobados_auto)
        
        return metricas
    
    def validacion_clasificador(self):
        """
        Nivel 2: Validación con clasificador entrenado.
        """
        print("\n" + "="*80)
        print("NIVEL 2: VALIDACIÓN CON CLASIFICADOR")
        print("="*80)
        
        if self.clf is None or self.vectorizer is None:
            print("  Clasificador no disponible, skip validación")
            return None
        
        print(f"\n Clasificando {len(self.df):,} PLS sintéticos...")
        print("   (Procesando en batches de 1000)")
        
        probas_pls = []
        batch_size = 1000
        
        for i in range(0, len(self.df), batch_size):
            batch = self.df['pls_generado'].iloc[i:i+batch_size].tolist()
            batch_clean = [str(text) for text in batch if pd.notna(text)]
            
            # Transformar con TF-IDF
            X_batch_tfidf = self.vectorizer.transform(batch_clean)
            
            # Agregar features de legibilidad si están disponibles
            if self.readability_extractor is not None:
                X_batch_readability = self.readability_extractor.transform(batch_clean)
                X_batch = hstack([X_batch_tfidf, X_batch_readability])
            else:
                X_batch = X_batch_tfidf
            
            probas_batch = self.clf.predict_proba(X_batch)[:, 1]
            probas_pls.extend(probas_batch)
            
            print(f"   Procesados: {min(i+batch_size, len(self.df)):,}/{len(self.df):,}")
        
        self.df['clf_prob_pls'] = probas_pls
        
        # Estadísticas
        print("\n RESULTADOS CLASIFICADOR:")
        print("-"*80)
        
        prob_media = self.df['clf_prob_pls'].mean()
        prob_mediana = self.df['clf_prob_pls'].median()
        prob_std = self.df['clf_prob_pls'].std()
        
        print(f"\nProbabilidad PLS:")
        print(f"   Media:   {prob_media:.4f}")
        print(f"   Mediana: {prob_mediana:.4f}")
        print(f"   Std:     {prob_std:.4f}")
        print(f"   Min:     {self.df['clf_prob_pls'].min():.4f}")
        print(f"   Max:     {self.df['clf_prob_pls'].max():.4f}")
        
        # Clasificados por umbral
        for umbral in [0.5, 0.6, 0.7, 0.8, 0.9]:
            n_clasificados = (self.df['clf_prob_pls'] >= umbral).sum()
            print(f"\n   Prob ≥{umbral}: {n_clasificados:,}/{len(self.df):,} "
                  f"({n_clasificados/len(self.df)*100:.1f}%)")
        
        # Umbral recomendado: 0.75
        aprobados_clf = (self.df['clf_prob_pls'] >= 0.75).sum()
        print(f"\n APROBADOS (prob ≥0.75): {aprobados_clf:,}/{len(self.df):,} "
              f"({aprobados_clf/len(self.df)*100:.1f}%)")
        
        self.resultados['prob_pls_media'] = float(prob_media)
        self.resultados['prob_pls_std'] = float(prob_std)
        self.resultados['aprobados_clasificador_075'] = int(aprobados_clf)
        
        return {
            'media': prob_media,
            'mediana': prob_mediana,
            'std': prob_std,
            'aprobados_075': aprobados_clf
        }
    
    def validacion_rouge_sari(self):
        """
        Evaluación con métricas ROUGE para simplificación de texto.
        
        ROUGE: Mide el overlap de n-gramas entre original y simplificado.
        NOTA: SARI está desactivado porque requiere referencias humanas reales
              (sin referencias, siempre da 100, que no es informativo).
        """
        print("\n" + "="*80)
        print("NIVEL 2.5: MÉTRICAS DE SIMPLIFICACIÓN (ROUGE)")
        print("="*80)
        
        print("\n Calculando ROUGE scores...")
        print("   (Mide similitud entre texto original y PLS)")
        
        # Inicializar ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        # Calcular ROUGE para cada par
        for idx, row in self.df.iterrows():
            if pd.notna(row['texto_original']) and pd.notna(row['pls_generado']):
                original = str(row['texto_original'])
                pls = str(row['pls_generado'])
                
                scores = scorer.score(original, pls)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            else:
                rouge1_scores.append(np.nan)
                rouge2_scores.append(np.nan)
                rougeL_scores.append(np.nan)
            
            if (idx + 1) % 1000 == 0:
                print(f"   Procesados: {idx+1:,}/{len(self.df):,}")
        
        self.df['rouge1'] = rouge1_scores
        self.df['rouge2'] = rouge2_scores
        self.df['rougeL'] = rougeL_scores
        
        print("\n RESULTADOS ROUGE:")
        print("-"*80)
        print(f"\nROUGE-1 (unigram overlap):")
        print(f"   Media:   {self.df['rouge1'].mean():.4f}")
        print(f"   Mediana: {self.df['rouge1'].median():.4f}")
        print(f"   Std:     {self.df['rouge1'].std():.4f}")
        
        print(f"\nROUGE-2 (bigram overlap):")
        print(f"   Media:   {self.df['rouge2'].mean():.4f}")
        print(f"   Mediana: {self.df['rouge2'].median():.4f}")
        print(f"   Std:     {self.df['rouge2'].std():.4f}")
        
        print(f"\nROUGE-L (longest common subsequence):")
        print(f"   Media:   {self.df['rougeL'].mean():.4f}")
        print(f"   Mediana: {self.df['rougeL'].median():.4f}")
        print(f"   Std:     {self.df['rougeL'].std():.4f}")
        
        # Calcular SARI
        print("\nINFO: SARI Score:")
        print("   SARI requiere referencias humanas reales para ser significativo.")
        print("   Sin referencias apropiadas, el score no es informativo.")
        print("   → SARI desactivado (necesita múltiples referencias de PLS reales)")
        
        # NOTA: SARI se desactiva porque usar el PLS como su propia referencia
        # resulta en score = 100 (perfecto), lo cual no es útil.
        # Para calcular SARI significativo necesitaríamos:
        #   - Múltiples referencias humanas para cada texto original
        #   - O usar PLS reales de Cochrane como referencias (requiere matching)
        
        sari_score = None
        self.resultados['sari_score'] = None
        
        print("\n   Alternativa: ROUGE es suficiente para medir:")
        print("      - Conservación de contenido (ROUGE-1)")
        print("      - Conservación de frases (ROUGE-2)")
        print("      - Conservación de estructura (ROUGE-L)")
        
        # Guardar resultados
        self.resultados['rouge_scores'] = {
            'rouge1_mean': float(self.df['rouge1'].mean()),
            'rouge1_std': float(self.df['rouge1'].std()),
            'rouge2_mean': float(self.df['rouge2'].mean()),
            'rouge2_std': float(self.df['rouge2'].std()),
            'rougeL_mean': float(self.df['rougeL'].mean()),
            'rougeL_std': float(self.df['rougeL'].std())
        }
        
        # Interpretación
        print("\n INTERPRETACIÓN:")
        print("-"*80)
        rouge1_mean = self.df['rouge1'].mean()
        
        if rouge1_mean > 0.5:
            print("  ✓ ROUGE-1 > 0.5: Buena conservación del contenido original")
        elif rouge1_mean > 0.3:
            print("  [ADV] ROUGE-1 0.3-0.5: Conservación moderada del contenido")
        else:
            print("  ✗ ROUGE-1 < 0.3: Baja conservación (posible paráfrasis excesiva)")
        
        print("\n  Nota: Para simplificación, ROUGE bajo puede indicar:")
        print("    • Buena paráfrasis (positivo)")
        print("    • Pérdida de información importante (negativo)")
        print("    → Debe evaluarse junto con otras métricas")
        
        return {
            'rouge1': self.df['rouge1'].mean(),
            'rouge2': self.df['rouge2'].mean(),
            'rougeL': self.df['rougeL'].mean(),
            'sari': sari_score
        }
    
    def comparacion_con_reales(self):
        """
        Nivel 3: Comparación estadística con PLS reales Cochrane.
        """
        print("\n" + "="*80)
        print("NIVEL 3: COMPARACIÓN CON PLS REALES COCHRANE")
        print("="*80)
        
        if self.pls_reales_probs is None:
            print("  No hay PLS reales para comparar")
            return None
        
        if 'clf_prob_pls' not in self.df.columns:
            print("  No hay probabilidades de clasificador para comparar")
            return None
        
        # Comparar distribuciones
        prob_reales_media = self.pls_reales_probs.mean()
        prob_sinteticos_media = self.df['clf_prob_pls'].mean()
        
        print("\n COMPARACIÓN DE PROBABILIDADES:")
        print("-"*80)
        print(f"\nPLS Reales Cochrane:")
        print(f"   Media: {prob_reales_media:.4f}")
        print(f"   Std:   {self.pls_reales_probs.std():.4f}")
        print(f"   n:     {len(self.pls_reales_probs):,}")
        
        print(f"\nPLS Sintéticos GPT:")
        print(f"   Media: {prob_sinteticos_media:.4f}")
        print(f"   Std:   {self.df['clf_prob_pls'].std():.4f}")
        print(f"   n:     {len(self.df):,}")
        
        print(f"\nDiferencia absoluta: {abs(prob_reales_media - prob_sinteticos_media):.4f}")
        
        # Test estadístico: Mann-Whitney U
        print("\n MANN-WHITNEY U TEST:")
        print("-"*80)
        print("H0: Las distribuciones son iguales")
        print("H1: Las distribuciones son diferentes")
        
        statistic, p_value = stats.mannwhitneyu(
            self.pls_reales_probs,
            self.df['clf_prob_pls'].values,
            alternative='two-sided'
        )
        
        print(f"\nU-statistic: {statistic:.2f}")
        print(f"p-value:     {p_value:.4f}")
        print(f"Alpha:       0.05")
        
        if p_value > 0.05:
            print(f"\n CONCLUSIÓN: No hay diferencia significativa (p={p_value:.4f} > 0.05)")
            print("   Los PLS sintéticos son estadísticamente indistinguibles")
            print("   de los PLS reales Cochrane según el clasificador.")
            conclusion = "No diferencia significativa"
        else:
            print(f"\n  CONCLUSIÓN: Hay diferencia significativa (p={p_value:.4f} < 0.05)")
            print("   Los PLS sintéticos difieren de los PLS reales.")
            if prob_sinteticos_media < prob_reales_media:
                print("   → Sintéticos tienen probabilidades MÁS BAJAS (más técnicos)")
            else:
                print("   → Sintéticos tienen probabilidades MÁS ALTAS")
            conclusion = "Diferencia significativa"
        
        self.resultados['comparacion'] = {
            'prob_reales_media': float(prob_reales_media),
            'prob_sinteticos_media': float(prob_sinteticos_media),
            'diferencia': float(abs(prob_reales_media - prob_sinteticos_media)),
            'u_statistic': float(statistic),
            'p_value': float(p_value),
            'conclusion': conclusion
        }
        
        return self.resultados['comparacion']
    
    def filtrado_final(self, fre_min=55, fkg_max=10, long_min=150, long_max=250, prob_min=0.75):
        """
        Filtra PLS por cuádruple criterio de calidad.
        """
        print("\n" + "="*80)
        print("FILTRADO FINAL (Cuádruple Criterio)")
        print("="*80)
        
        print(f"\nCriterios de aprobación:")
        print(f"  1. FRE ≥ {fre_min}")
        print(f"  2. FKG ≤ {fkg_max}")
        print(f"  3. Longitud: {long_min}-{long_max} palabras")
        print(f"  4. Clasificador prob ≥ {prob_min}")
        
        # Aplicar filtros progresivamente
        total = len(self.df)
        
        df_f1 = self.df[self.df['fre'] >= fre_min].copy()
        print(f"\nDespués filtro FRE:           {len(df_f1):,}/{total:,} ({len(df_f1)/total*100:.1f}%)")
        
        df_f2 = df_f1[df_f1['fkg'] <= fkg_max].copy()
        print(f"Después filtro FKG:           {len(df_f2):,}/{total:,} ({len(df_f2)/total*100:.1f}%)")
        
        df_f3 = df_f2[
            (df_f2['longitud_pls'] >= long_min) &
            (df_f2['longitud_pls'] <= long_max)
        ].copy()
        print(f"Después filtro Longitud:      {len(df_f3):,}/{total:,} ({len(df_f3)/total*100:.1f}%)")
        
        if 'clf_prob_pls' in df_f3.columns:
            df_final = df_f3[df_f3['clf_prob_pls'] >= prob_min].copy()
            print(f"Después filtro Clasificador:  {len(df_final):,}/{total:,} ({len(df_final)/total*100:.1f}%)")
        else:
            df_final = df_f3
            print("  Sin filtro de clasificador (no disponible)")
        
        print(f"\n{'='*80}")
        print(f" DATASET FINAL APROBADO: {len(df_final):,}/{total:,} ({len(df_final)/total*100:.1f}%)")
        print(f"{'='*80}")
        
        # Estadísticas del dataset final
        if len(df_final) > 0:
            print(f"\nESTADÍSTICAS DATASET FINAL:")
            print(f"  FRE medio:  {df_final['fre'].mean():.1f}")
            print(f"  FKG medio:  {df_final['fkg'].mean():.1f}")
            print(f"  Long media: {df_final['longitud_pls'].mean():.0f} palabras")
            if 'clf_prob_pls' in df_final.columns:
                print(f"  Prob media: {df_final['clf_prob_pls'].mean():.3f}")
        
        self.resultados['dataset_final'] = {
            'total_original': int(total),
            'total_aprobado': int(len(df_final)),
            'tasa_aprobacion': float(len(df_final) / total),
            'criterios': {
                'fre_min': fre_min,
                'fkg_max': fkg_max,
                'longitud_min': long_min,
                'longitud_max': long_max,
                'prob_min': prob_min
            }
        }
        
        return df_final
    
    def generar_visualizaciones(self, df_final):
        """
        Genera visualizaciones de la validación.
        """
        print("\n GENERANDO VISUALIZACIONES...")
        print("-"*80)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        n_total = len(self.df)
        fig.suptitle(f'Validación de {n_total:,} PLS Sintéticos', fontsize=16, fontweight='bold')
        
        # 1. Distribución FRE
        ax1 = axes[0, 0]
        ax1.hist(self.df['fre'], bins=50, alpha=0.6, color='blue', label='Todos', edgecolor='black')
        if len(df_final) > 0:
            ax1.hist(df_final['fre'], bins=50, alpha=0.6, color='green', label='Aprobados', edgecolor='black')
        ax1.axvline(55, color='red', linestyle='--', linewidth=2, label='Umbral (55)')
        ax1.axvline(60, color='orange', linestyle='--', linewidth=2, label='Target (60)')
        ax1.set_xlabel('Flesch Reading Ease')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución FRE')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Distribución FKG
        ax2 = axes[0, 1]
        ax2.hist(self.df['fkg'], bins=50, alpha=0.6, color='blue', label='Todos', edgecolor='black')
        if len(df_final) > 0:
            ax2.hist(df_final['fkg'], bins=50, alpha=0.6, color='green', label='Aprobados', edgecolor='black')
        ax2.axvline(10, color='red', linestyle='--', linewidth=2, label='Umbral (10)')
        ax2.axvline(8, color='orange', linestyle='--', linewidth=2, label='Target (8)')
        ax2.set_xlabel('Flesch-Kincaid Grade')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución FKG')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Distribución Longitud
        ax3 = axes[0, 2]
        ax3.hist(self.df['longitud_pls'], bins=50, alpha=0.6, color='blue', label='Todos', edgecolor='black')
        if len(df_final) > 0:
            ax3.hist(df_final['longitud_pls'], bins=50, alpha=0.6, color='green', label='Aprobados', edgecolor='black')
        ax3.axvline(150, color='red', linestyle='--', linewidth=2, label='Min (150)')
        ax3.axvline(250, color='red', linestyle='--', linewidth=2, label='Max (250)')
        ax3.set_xlabel('Longitud (palabras)')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución Longitud')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Probabilidades Clasificador
        if 'clf_prob_pls' in self.df.columns:
            ax4 = axes[1, 0]
            ax4.hist(self.df['clf_prob_pls'], bins=50, alpha=0.6, color='blue', 
                    label='Sintéticos', edgecolor='black')
            if self.pls_reales_probs is not None:
                ax4.hist(self.pls_reales_probs, bins=50, alpha=0.6, color='orange',
                        label='Reales Cochrane', edgecolor='black')
            ax4.axvline(0.75, color='red', linestyle='--', linewidth=2, label='Umbral (0.75)')
            ax4.set_xlabel('Probabilidad PLS')
            ax4.set_ylabel('Frecuencia')
            ax4.set_title('Distribución Prob. Clasificador')
            ax4.legend()
            ax4.grid(alpha=0.3)
        
        # 5. Boxplot comparativo
        ax5 = axes[1, 1]
        data_box = []
        labels_box = []
        
        if self.pls_reales_probs is not None:
            data_box.append(self.pls_reales_probs)
            labels_box.append('PLS Reales\nCochrane')
        
        if 'clf_prob_pls' in self.df.columns:
            data_box.append(self.df['clf_prob_pls'].values)
            labels_box.append('PLS Sintéticos\nGPT-4o-mini')
            
            if len(df_final) > 0 and 'clf_prob_pls' in df_final.columns:
                data_box.append(df_final['clf_prob_pls'].values)
                labels_box.append('PLS Aprobados\n(filtrados)')
        
        if len(data_box) > 0:
            bp = ax5.boxplot(data_box, labels=labels_box, patch_artist=True)
            colors = ['orange', 'blue', 'green']
            for patch, color in zip(bp['boxes'], colors[:len(data_box)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax5.axhline(0.75, color='red', linestyle='--', linewidth=2, label='Umbral')
            ax5.set_ylabel('Probabilidad PLS')
            ax5.set_title('Comparación de Distribuciones')
            ax5.legend()
            ax5.grid(alpha=0.3, axis='y')
        else:
            ax5.text(0.5, 0.5, 'Clasificador no disponible', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Comparación de Distribuciones')
            ax5.axis('off')
        
        # 6. Resumen de aprobación
        ax6 = axes[1, 2]
        
        categorias = ['Total\nGenerados', 'FRE≥55', 'FKG≤10', 'Long\n150-250', 
                     'Prob≥0.75', 'Final\nAprobados']
        valores = [
            len(self.df),
            (self.df['fre'] >= 55).sum(),
            (self.df['fkg'] <= 10).sum(),
            ((self.df['longitud_pls'] >= 150) & (self.df['longitud_pls'] <= 250)).sum(),
            (self.df['clf_prob_pls'] >= 0.75).sum() if 'clf_prob_pls' in self.df.columns else 0,
            len(df_final)
        ]
        
        colors_bar = ['gray', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'green']
        bars = ax6.bar(categorias, valores, color=colors_bar, edgecolor='black', linewidth=1.5)
        
        # Agregar valores encima de barras
        for bar, valor in zip(bars, valores):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{valor:,}\n({valor/len(self.df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax6.set_ylabel('Cantidad de PLS')
        ax6.set_title('Pipeline de Filtrado')
        ax6.grid(alpha=0.3, axis='y')
        
        # 7. Distribución ROUGE-1
        if 'rouge1' in self.df.columns:
            ax7 = axes[2, 0]
            ax7.hist(self.df['rouge1'].dropna(), bins=50, alpha=0.6, color='purple', 
                    label='ROUGE-1', edgecolor='black')
            ax7.axvline(self.df['rouge1'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Media: {self.df["rouge1"].mean():.3f}')
            ax7.set_xlabel('ROUGE-1 Score')
            ax7.set_ylabel('Frecuencia')
            ax7.set_title('Distribución ROUGE-1 (unigram overlap)')
            ax7.legend()
            ax7.grid(alpha=0.3)
        else:
            axes[2, 0].axis('off')
        
        # 8. Distribución ROUGE-2
        if 'rouge2' in self.df.columns:
            ax8 = axes[2, 1]
            ax8.hist(self.df['rouge2'].dropna(), bins=50, alpha=0.6, color='teal', 
                    label='ROUGE-2', edgecolor='black')
            ax8.axvline(self.df['rouge2'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Media: {self.df["rouge2"].mean():.3f}')
            ax8.set_xlabel('ROUGE-2 Score')
            ax8.set_ylabel('Frecuencia')
            ax8.set_title('Distribución ROUGE-2 (bigram overlap)')
            ax8.legend()
            ax8.grid(alpha=0.3)
        else:
            axes[2, 1].axis('off')
        
        # 9. Distribución ROUGE-L
        if 'rougeL' in self.df.columns:
            ax9 = axes[2, 2]
            ax9.hist(self.df['rougeL'].dropna(), bins=50, alpha=0.6, color='magenta', 
                    label='ROUGE-L', edgecolor='black')
            ax9.axvline(self.df['rougeL'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Media: {self.df["rougeL"].mean():.3f}')
            ax9.set_xlabel('ROUGE-L Score')
            ax9.set_ylabel('Frecuencia')
            ax9.set_title('Distribución ROUGE-L (LCS)')
            ax9.legend()
            ax9.grid(alpha=0.3)
        else:
            axes[2, 2].axis('off')
        
        plt.tight_layout()
        
        # Guardar
        output_path = 'validacion_distribucion.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f" Visualizaciones guardadas: {output_path}")
        
        plt.close()
    
    def guardar_resultados(self, df_final):
        """
        Guarda todos los resultados de la validación.
        """
        print("\n GUARDANDO RESULTADOS...")
        print("-"*80)
        
        # 1. CSV con todos los PLS y métricas
        n_total = len(self.df)
        output_all = f'pls_{n_total//1000}k_validado.csv' if n_total >= 1000 else f'pls_{n_total}_validado.csv'
        self.df.to_csv(output_all, index=False)
        print(f" Guardado: {output_all}")
        print(f"   Contiene: {len(self.df):,} PLS con todas las métricas")
        
        # 2. CSV solo con aprobados
        output_final = f'pls_{n_total//1000}k_final_aprobados.csv' if n_total >= 1000 else f'pls_{n_total}_final_aprobados.csv'
        df_final.to_csv(output_final, index=False)
        print(f" Guardado: {output_final}")
        print(f"   Contiene: {len(df_final):,} PLS aprobados")
        
        # 3. Reporte en texto
        output_report = 'validacion_report.txt'
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"REPORTE DE VALIDACIÓN DE {len(self.df):,} PLS SINTÉTICOS\n")
            f.write("="*80 + "\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Archivo fuente: {self.csv_path}\n")
            f.write("\n")
            
            f.write("RESUMEN EJECUTIVO:\n")
            f.write("-"*80 + "\n")
            f.write(f"Total PLS generados:    {len(self.df):,}\n")
            f.write(f"PLS aprobados:          {len(df_final):,}\n")
            f.write(f"Tasa de aprobación:     {len(df_final)/len(self.df)*100:.1f}%\n")
            f.write("\n")
            
            f.write("MÉTRICAS AUTOMÁTICAS:\n")
            f.write("-"*80 + "\n")
            if 'metricas_automaticas' in self.resultados:
                for metrica, datos in self.resultados['metricas_automaticas'].items():
                    f.write(f"\n{metrica.upper()}:\n")
                    f.write(f"  Media: {datos['media']:.2f}\n")
                    f.write(f"  Target: {datos['target']}\n")
                    f.write(f"  Cumplen: {datos['cumple']:,}/{len(self.df):,}\n")
            f.write("\n")
            
            f.write("CLASIFICADOR:\n")
            f.write("-"*80 + "\n")
            if 'prob_pls_media' in self.resultados:
                f.write(f"Prob. PLS media: {self.resultados['prob_pls_media']:.4f}\n")
                f.write(f"Aprobados (≥0.75): {self.resultados['aprobados_clasificador_075']:,}\n")
            f.write("\n")
            
            f.write("MÉTRICAS DE SIMPLIFICACIÓN (ROUGE & SARI):\n")
            f.write("-"*80 + "\n")
            if 'rouge_scores' in self.resultados:
                rs = self.resultados['rouge_scores']
                f.write(f"ROUGE-1 (unigram overlap): {rs['rouge1_mean']:.4f} ± {rs['rouge1_std']:.4f}\n")
                f.write(f"ROUGE-2 (bigram overlap):  {rs['rouge2_mean']:.4f} ± {rs['rouge2_std']:.4f}\n")
                f.write(f"ROUGE-L (LCS):             {rs['rougeL_mean']:.4f} ± {rs['rougeL_std']:.4f}\n")
            if 'sari_score' in self.resultados and self.resultados['sari_score'] is not None:
                f.write(f"SARI Score (corpus-level): {self.resultados['sari_score']:.4f}/100\n")
            f.write("\n")
            
            if 'comparacion' in self.resultados:
                f.write("COMPARACIÓN CON PLS REALES:\n")
                f.write("-"*80 + "\n")
                comp = self.resultados['comparacion']
                f.write(f"Prob. PLS Reales:     {comp['prob_reales_media']:.4f}\n")
                f.write(f"Prob. PLS Sintéticos: {comp['prob_sinteticos_media']:.4f}\n")
                f.write(f"Mann-Whitney p-value: {comp['p_value']:.4f}\n")
                f.write(f"Conclusión: {comp['conclusion']}\n")
            
        print(f" Guardado: {output_report}")
        
        # 4. JSON con métricas
        output_json = 'validacion_metrics.json'
        # Convertir valores numpy/pandas a tipos nativos de Python
        resultados_serializables = convertir_a_json_serializable(self.resultados)
        with open(output_json, 'w') as f:
            json.dump(resultados_serializables, f, indent=2)
        print(f" Guardado: {output_json}")
        
        print("\n" + "="*80)
        print(" VALIDACIÓN COMPLETADA")
        print("="*80)
        print(f"\nArchivos generados:")
        print(f"  1. {output_all} - Todos los PLS con métricas")
        print(f"  2. {output_final} - Solo PLS aprobados ({len(df_final):,})")
        print(f"  3. {output_report} - Reporte en texto")
        print(f"  4. {output_json} - Métricas en JSON")
        print(f"  5. validacion_distribucion.png - Visualizaciones")
        
        print(f"\n PRÓXIMO PASO:")
        print(f"   Usar {output_final} para fine-tuning de T5")
        print(f"   Dataset validado: {len(df_final):,} pares de alta calidad")
    
    def ejecutar_validacion_completa(self):
        """
        Pipeline completo de validación.
        """
        # 1. Cargar datos
        self.cargar_datos()
        
        # 2. Cargar clasificador
        self.cargar_clasificador()
        
        # 3. Validación automática
        self.validacion_automatica()
        
        # 4. Validación con clasificador
        if self.clf is not None:
            self.validacion_clasificador()
        
        # 4.5. Validación ROUGE y SARI
        self.validacion_rouge_sari()
        
        # 5. Comparación con reales
        if self.pls_reales_probs is not None:
            self.comparacion_con_reales()
        
        # 6. Filtrado final
        df_final = self.filtrado_final()
        
        # 7. Visualizaciones
        self.generar_visualizaciones(df_final)
        
        # 8. Guardar resultados
        self.guardar_resultados(df_final)
        
        return df_final


def main():
    """
    Script principal de validación.
    """
    # Forzar flush inmediato de prints
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None
    sys.stderr.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stderr, 'reconfigure') else None
    
    # Print inicial para verificar que funciona
    print("\n", flush=True)
    print("="*80, flush=True)
    print(" " + " "*20 + "VALIDACION DE PLS SINTETICOS" + " "*30, flush=True)
    print("="*80, flush=True)
    print(flush=True)
    
    # El validador ahora busca automáticamente el archivo
    # Pero también acepta un argumento de línea de comandos
    csv_path = None
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not Path(csv_path).exists():
            print(f" ERROR: Archivo especificado no encontrado: {csv_path}", flush=True)
            return
    
    # Ejecutar validación (busca automáticamente si no se especificó)
    try:
        print("Iniciando validador...", flush=True)
        validador = ValidadorPLSSinteticos(csv_path)
    except FileNotFoundError as e:
        print(f" ERROR: {e}", flush=True)
        return
    except Exception as e:
        print(f" ERROR al inicializar validador: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    try:
        print("Ejecutando validacion completa...", flush=True)
        df_final = validador.ejecutar_validacion_completa()
        
        print("\n" + "="*80, flush=True)
        print(" VALIDACION EXITOSA", flush=True)
        print("="*80, flush=True)
        print(f"\n Dataset final: {len(df_final):,} PLS aprobados", flush=True)
        print(f" Listos para fine-tuning de T5", flush=True)
        
    except Exception as e:
        print(f"\n ERROR durante validacion: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
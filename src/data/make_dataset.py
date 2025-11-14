from __future__ import annotations
import csv
import json
import sys
import hashlib
import unicodedata
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Optional
from datetime import datetime

# Librerías opcionales (instalar si no están)
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42  # Reproducibilidad
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("[WARN] langdetect no disponible. Instalar con: pip install langdetect")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("[WARN] textstat no disponible. Instalar con: pip install textstat")

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("[WARN] PyPDF2 no disponible. Instalar con: pip install PyPDF2")

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# ==================== CONFIGURACIÓN ====================

# Formatos admitidos
ALLOWED_SUFFIXES = {".txt", ".csv", ".json", ".jsonl", ".pdf"}
SKIP_SUFFIXES = set()

# FILTROS DE CALIDAD MEJORADOS (para textos médicos)
MIN_LEN_TEXT = 100  # ~15-20 palabras mínimo
MAX_LEN_TEXT = 5000  # ~750 palabras máximo
MIN_WORDS = 15  # Mínimo de palabras
MAX_WORDS = 800  # Máximo de palabras

# Validación de idioma
MIN_ENGLISH_CONFIDENCE = 0.9  # 90% debe ser inglés

# Fragmentación de párrafos
MIN_PARAGRAPH_WORDS = 20  # Mínimo palabras por párrafo
PARAGRAPH_SEPARATOR = r'\n\s*\n'  # Dos saltos de línea

# Detección de problemas de OCR
OCR_ERROR_PATTERNS = [
    r'\b\w{1,2}\s+\w{1,2}\b',  # Palabras fragmentadas: "pa tient"
    r'[^\x00-\x7F]{5,}',  # Caracteres no-ASCII consecutivos
    r'\b\d{10,}\b',  # Números excesivamente largos
]

# Heurísticas de columnas
TEXT_CANDIDATES = {
    "texto", "text", "source", "article", "document", "original",
    "input", "content", "body", "source_text", "original_text",
    "full_text", "document_text",
}
PLS_CANDIDATES = {
    "resumen", "summary", "pls", "simple", "plain_language",
    "simplified", "plain_language_summary", "plainlanguage",
    "lay_summary", "plain_summary", "pls_text",
}


# ==================== UTILIDADES ====================

def norm(s: str) -> str:
    """Normalización SELECTIVA: preserva párrafos pero limpia espacios."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    
    # Normalización unicode
    s = unicodedata.normalize("NFKC", s)
    
    # Normalizar saltos de línea
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    
    # Preservar dobles saltos (separadores de párrafos)
    # pero eliminar más de 2 saltos consecutivos
    s = re.sub(r'\n{3,}', '\n\n', s)
    
    # Colapsar espacios múltiples en la misma línea
    lines = s.split('\n')
    lines = [' '.join(line.split()) for line in lines]
    s = '\n'.join(lines)
    
    return s.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """
    Divide texto en párrafos válidos.
    
    Reglas:
    - Separa por doble salto de línea
    - Filtra párrafos muy cortos (<MIN_PARAGRAPH_WORDS palabras)
    - Preserva contexto médico completo
    """
    # Dividir por doble salto de línea
    paragraphs = re.split(PARAGRAPH_SEPARATOR, text)
    
    valid_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        
        word_count = len(p.split())
        if word_count >= MIN_PARAGRAPH_WORDS:
            valid_paragraphs.append(p)
    
    return valid_paragraphs


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detecta idioma y confianza.
    
    Returns:
        (idioma, confianza) ej: ('en', 0.95)
    """
    if not LANGDETECT_AVAILABLE:
        return 'en', 1.0  # Asumir inglés si no hay librería
    
    try:
        # Detectar en los primeros 1000 chars para eficiencia
        sample = text[:1000]
        lang = detect(sample)
        # langdetect no da confianza directamente, usamos 0.95 si detecta
        confidence = 0.95 if lang == 'en' else 0.5
        return lang, confidence
    except:
        return 'unk', 0.0


def calculate_readability(text: str) -> Dict[str, float]:
    """
    Calcula métricas de legibilidad.
    
    Returns:
        {
            'flesch_reading_ease': 0-100 (más alto = más fácil),
            'flesch_kincaid_grade': nivel escolar,
            'avg_word_length': promedio longitud palabra,
            'avg_sentence_length': promedio palabras por oración,
            'complex_words_ratio': proporción de palabras complejas
        }
    """
    metrics = {
        'flesch_reading_ease': 0.0,
        'flesch_kincaid_grade': 0.0,
        'avg_word_length': 0.0,
        'avg_sentence_length': 0.0,
        'complex_words_ratio': 0.0,
    }
    
    if not TEXTSTAT_AVAILABLE:
        return metrics
    
    try:
        metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        
        # Métricas adicionales
        words = text.split()
        if words:
            metrics['avg_word_length'] = sum(len(w) for w in words) / len(words)
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences and words:
            metrics['avg_sentence_length'] = len(words) / len(sentences)
        
        # Palabras complejas (>3 sílabas) - aproximación simple
        complex_words = [w for w in words if len(w) > 10]
        if words:
            metrics['complex_words_ratio'] = len(complex_words) / len(words)
        
    except Exception as e:
        print(f"[WARN] Error calculando legibilidad: {e}", file=sys.stderr)
    
    return metrics


def detect_ocr_errors(text: str) -> List[str]:
    """
    Detecta posibles errores de OCR.
    
    Returns:
        Lista de problemas detectados
    """
    issues = []
    
    for pattern in OCR_ERROR_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            issues.append(f"OCR_pattern_{pattern[:20]}: {len(matches)} matches")
    
    # Detectar palabras fragmentadas comunes
    fragmented_words = re.findall(r'\b\w{1,2}\s+\w{1,2}\s+\w{1,2}\b', text)
    if fragmented_words:
        issues.append(f"Fragmented_words: {len(fragmented_words)}")
    
    # Detectar exceso de caracteres especiales
    special_chars = len(re.findall(r'[^\w\s.,;:!?()\-\'\"\/]', text))
    if special_chars > len(text) * 0.02:  # >2% de caracteres especiales
        issues.append(f"Excess_special_chars: {special_chars}")
    
    return issues


def calculate_medical_density(text: str) -> Dict[str, float]:
    """
    Calcula densidad de términos médicos.
    
    Returns:
        {
            'abbrev_count': número de abreviaturas médicas,
            'abbrev_density': proporción de abreviaturas,
            'number_density': proporción de números,
            'medical_terms_ratio': estimación de términos médicos
        }
    """
    words = text.split()
    if not words:
        return {'abbrev_count': 0, 'abbrev_density': 0, 'number_density': 0, 'medical_terms_ratio': 0}
    
    # Abreviaturas en mayúsculas (ej: FDA, WHO, HIV)
    abbrevs = re.findall(r'\b[A-Z]{2,}\b', text)
    
    # Números
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    
    # Términos médicos comunes (lista simplificada)
    medical_terms = [
        'patient', 'treatment', 'study', 'trial', 'efficacy', 'safety',
        'adverse', 'dose', 'therapy', 'clinical', 'diagnosis', 'symptom',
        'disease', 'condition', 'medication', 'drug', 'placebo', 'randomized'
    ]
    medical_count = sum(1 for w in words if w.lower() in medical_terms)
    
    return {
        'abbrev_count': len(abbrevs),
        'abbrev_density': len(abbrevs) / len(words),
        'number_density': len(numbers) / len(words),
        'medical_terms_ratio': medical_count / len(words),
    }


def quality_check(text: str, context: str = "texto_original") -> Tuple[bool, List[str]]:
    """
    Verifica calidad del texto con criterios estrictos.
    
    Returns:
        (es_válido, lista_de_problemas)
    """
    issues = []
    
    if not text or not text.strip():
        return False, ["Empty text"]
    
    text = text.strip()
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    
    # 1. Validar longitud
    if char_count < MIN_LEN_TEXT:
        issues.append(f"Too_short: {char_count} chars")
    if char_count > MAX_LEN_TEXT:
        issues.append(f"Too_long: {char_count} chars")
    if word_count < MIN_WORDS:
        issues.append(f"Too_few_words: {word_count}")
    if word_count > MAX_WORDS:
        issues.append(f"Too_many_words: {word_count}")
    
    # 2. Validar idioma
    lang, confidence = detect_language(text)
    if lang != 'en':
        issues.append(f"Wrong_language: {lang}")
    if confidence < MIN_ENGLISH_CONFIDENCE:
        issues.append(f"Low_confidence: {confidence:.2f}")
    
    # 3. Detectar errores de OCR
    ocr_errors = detect_ocr_errors(text)
    issues.extend(ocr_errors)
    
    # 4. Validar estructura mínima (al menos una oración completa)
    sentences = re.split(r'[.!?]+', text)
    valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    if not valid_sentences:
        issues.append("No_valid_sentences")
    
    # Decisión final
    critical_issues = [i for i in issues if any(x in i for x in ['Empty', 'Too_short', 'Too_few', 'Wrong_language'])]
    is_valid = len(critical_issues) == 0
    
    return is_valid, issues


def infer_meta_from_path(fp: Path):
    """Infiere metadatos desde la ruta del archivo."""
    parts = [p.lower() for p in fp.parts]
    
    source_dataset = ""
    try:
        raw_idx = parts.index("raw")
        if raw_idx + 1 < len(parts):
            source_dataset = parts[raw_idx + 1]
    except ValueError:
        pass
    
    source_bucket = ""
    if source_dataset:
        start = parts.index(source_dataset) + 1
        if start < len(parts) - 1:
            source_bucket = "/".join(parts[start:-1])
    
    split = "unsplit"
    if "train" in parts:
        split = "train"
    if "test" in parts:
        split = "test"
    
    label = ""
    if "pls" in parts:
        label = "pls"
    if ("non_pls" in parts) or ("non-pls" in parts):
        label = "non_pls"
    
    # Normalizar nombres
    mapping = {
        "cochrane": "cochrane",
        "pfizer": "pfizer",
        "trial summaries": "trialsummaries",
        "trialsummaries": "trialsummaries",
        "clinicaltrials.gov": "clinicaltrials",
        "clinicaltrials": "clinicaltrials",
    }
    source_dataset = mapping.get(source_dataset, source_dataset)
    
    return source_dataset, source_bucket, split, label


def pick_columns(header: List[str]) -> Tuple[str, str]:
    """Detecta columnas de texto/resumen en CSV/JSON."""
    lower = [h.lower().strip() for h in header]
    text_col = ""
    pls_col = ""
    for c in lower:
        if not text_col and c in TEXT_CANDIDATES:
            text_col = c
        if not pls_col and c in PLS_CANDIDATES:
            pls_col = c
    if not text_col and lower:
        text_col = lower[0]
    if not pls_col and len(lower) > 1:
        pls_col = lower[1]
    return text_col, pls_col


# ==================== PARSERS MEJORADOS ====================

def iter_txt(fp: Path) -> Iterator[Dict[str, any]]:
    """
    Lee .txt con fragmentación por PÁRRAFOS (no líneas).
    
    MEJORA PRINCIPAL: Agrupa líneas en párrafos coherentes.
    """
    source_dataset, source_bucket, split, label = infer_meta_from_path(fp)
    
    try:
        content = fp.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Error leyendo {fp}: {e}", file=sys.stderr)
        return
    
    # Normalizar primero
    content = norm(content)
    
    # Dividir en párrafos
    paragraphs = split_into_paragraphs(content)
    
    for i, paragraph in enumerate(paragraphs):
        # Detectar pares TEXTO ||| PLS
        if "|||" in paragraph:
            src, pls = paragraph.split("|||", 1)
            src = norm(src)
            pls = norm(pls)
            
            # Validar ambos lados
            src_valid, src_issues = quality_check(src, "texto_original")
            pls_valid, pls_issues = quality_check(pls, "resumen")
            
            if src_valid or pls_valid:  # Al menos uno válido
                # Calcular métricas
                readability = calculate_readability(src if src_valid else pls)
                medical = calculate_medical_density(src if src_valid else pls)
                
                yield {
                    "texto_original": src,
                    "resumen": pls,
                    "source": fp.parent.name,
                    "doc_id": f"{source_dataset}_{fp.stem}#p{i+1}",
                    "split": split,
                    "label": label,
                    "source_dataset": source_dataset,
                    "source_bucket": source_bucket,
                    "has_pair": True,
                    # Métricas nuevas
                    "word_count_src": len(src.split()),
                    "word_count_pls": len(pls.split()),
                    "flesch_score": readability['flesch_reading_ease'],
                    "avg_word_length": readability['avg_word_length'],
                    "medical_density": medical['medical_terms_ratio'],
                    "quality_issues": "|".join(src_issues + pls_issues) if (src_issues or pls_issues) else "",
                    "processing_date": datetime.now().isoformat(),
                }
        else:
            # Párrafo suelto
            text_valid, issues = quality_check(paragraph)
            
            if not text_valid:
                continue  # Saltar párrafos inválidos
            
            # Calcular métricas
            readability = calculate_readability(paragraph)
            medical = calculate_medical_density(paragraph)
            
            # Determinar si es PLS o técnico
            if label == "pls":
                yield {
                    "texto_original": "",
                    "resumen": paragraph,
                    "source": fp.parent.name,
                    "doc_id": f"{source_dataset}_{fp.stem}#p{i+1}",
                    "split": split,
                    "label": label,
                    "source_dataset": source_dataset,
                    "source_bucket": source_bucket,
                    "has_pair": False,
                    "word_count_src": 0,
                    "word_count_pls": len(paragraph.split()),
                    "flesch_score": readability['flesch_reading_ease'],
                    "avg_word_length": readability['avg_word_length'],
                    "medical_density": medical['medical_terms_ratio'],
                    "quality_issues": "|".join(issues) if issues else "",
                    "processing_date": datetime.now().isoformat(),
                }
            else:
                yield {
                    "texto_original": paragraph,
                    "resumen": "",
                    "source": fp.parent.name,
                    "doc_id": f"{source_dataset}_{fp.stem}#p{i+1}",
                    "split": split,
                    "label": label,
                    "source_dataset": source_dataset,
                    "source_bucket": source_bucket,
                    "has_pair": False,
                    "word_count_src": len(paragraph.split()),
                    "word_count_pls": 0,
                    "flesch_score": readability['flesch_reading_ease'],
                    "avg_word_length": readability['avg_word_length'],
                    "medical_density": medical['medical_terms_ratio'],
                    "quality_issues": "|".join(issues) if issues else "",
                    "processing_date": datetime.now().isoformat(),
                }


def iter_csv(fp: Path) -> Iterator[Dict[str, any]]:
    """Lee CSV con validación de calidad."""
    import pandas as pd
    
    source_dataset, source_bucket, split, label = infer_meta_from_path(fp)
    
    for enc in ("utf-8", "latin1"):
        try:
            df = pd.read_csv(fp, encoding=enc)
            break
        except Exception:
            if enc == "latin1":
                print(f"[WARN] Error leyendo {fp}", file=sys.stderr)
                return
    
    if df.empty:
        return
    
    tcol, pcol = pick_columns(df.columns.tolist())
    
    for i, row in df.iterrows():
        texto = norm(row.get(tcol, ""))
        pls = norm(row.get(pcol, ""))
        
        # Validar calidad
        texto_valid, texto_issues = quality_check(texto, "texto_original") if texto else (False, [])
        pls_valid, pls_issues = quality_check(pls, "resumen") if pls else (False, [])
        
        if not texto_valid and not pls_valid:
            continue  # Saltar si ninguno es válido
        
        # Métricas
        readability = calculate_readability(texto if texto_valid else pls)
        medical = calculate_medical_density(texto if texto_valid else pls)
        
        yield {
            "texto_original": texto,
            "resumen": pls,
            "source": fp.parent.name,
            "doc_id": f"{source_dataset}_{fp.name}#r{i}",
            "split": split,
            "label": label,
            "source_dataset": source_dataset,
            "source_bucket": source_bucket,
            "has_pair": bool(texto_valid and pls_valid),
            "word_count_src": len(texto.split()) if texto else 0,
            "word_count_pls": len(pls.split()) if pls else 0,
            "flesch_score": readability['flesch_reading_ease'],
            "avg_word_length": readability['avg_word_length'],
            "medical_density": medical['medical_terms_ratio'],
            "quality_issues": "|".join(texto_issues + pls_issues) if (texto_issues or pls_issues) else "",
            "processing_date": datetime.now().isoformat(),
        }


def iter_jsonl(fp: Path) -> Iterator[Dict[str, any]]:
    """Lee JSONL con validación."""
    source_dataset, source_bucket, split, label = infer_meta_from_path(fp)
    
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            
            keys = list(obj.keys())
            tcol, pcol = pick_columns(keys)
            texto = norm(obj.get(tcol, ""))
            pls = norm(obj.get(pcol, ""))
            
            texto_valid, texto_issues = quality_check(texto) if texto else (False, [])
            pls_valid, pls_issues = quality_check(pls) if pls else (False, [])
            
            if not texto_valid and not pls_valid:
                continue
            
            readability = calculate_readability(texto if texto_valid else pls)
            medical = calculate_medical_density(texto if texto_valid else pls)
            
            yield {
                "texto_original": texto,
                "resumen": pls,
                "source": fp.parent.name,
                "doc_id": f"{source_dataset}_{fp.name}#l{i}",
                "split": split,
                "label": label,
                "source_dataset": source_dataset,
                "source_bucket": source_bucket,
                "has_pair": bool(texto_valid and pls_valid),
                "word_count_src": len(texto.split()) if texto else 0,
                "word_count_pls": len(pls.split()) if pls else 0,
                "flesch_score": readability['flesch_reading_ease'],
                "avg_word_length": readability['avg_word_length'],
                "medical_density": medical['medical_terms_ratio'],
                "quality_issues": "|".join(texto_issues + pls_issues),
                "processing_date": datetime.now().isoformat(),
            }


def iter_json(fp: Path) -> Iterator[Dict[str, any]]:
    """Lee JSON con validación."""
    source_dataset, source_bucket, split, label = infer_meta_from_path(fp)
    
    try:
        data = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return
    
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return
    
    if data and isinstance(data[0], dict):
        keys = list(data[0].keys())
        tcol, pcol = pick_columns(keys)
        
        for i, obj in enumerate(data):
            texto = norm(obj.get(tcol, ""))
            pls = norm(obj.get(pcol, ""))
            
            texto_valid, texto_issues = quality_check(texto) if texto else (False, [])
            pls_valid, pls_issues = quality_check(pls) if pls else (False, [])
            
            if not texto_valid and not pls_valid:
                continue
            
            readability = calculate_readability(texto if texto_valid else pls)
            medical = calculate_medical_density(texto if texto_valid else pls)
            
            yield {
                "texto_original": texto,
                "resumen": pls,
                "source": fp.parent.name,
                "doc_id": f"{source_dataset}_{fp.name}#o{i}",
                "split": split,
                "label": label,
                "source_dataset": source_dataset,
                "source_bucket": source_bucket,
                "has_pair": bool(texto_valid and pls_valid),
                "word_count_src": len(texto.split()) if texto else 0,
                "word_count_pls": len(pls.split()) if pls else 0,
                "flesch_score": readability['flesch_reading_ease'],
                "avg_word_length": readability['avg_word_length'],
                "medical_density": medical['medical_terms_ratio'],
                "quality_issues": "|".join(texto_issues + pls_issues),
                "processing_date": datetime.now().isoformat(),
            }


def iter_pdf(fp: Path) -> Iterator[Dict[str, any]]:
    """Lee PDF con extracción de texto."""
    if not PDF_AVAILABLE:
        print(f"[WARN] PyPDF2 no disponible, saltando {fp}", file=sys.stderr)
        return
    
    source_dataset, source_bucket, split, label = infer_meta_from_path(fp)
    
    try:
        with fp.open("rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n"
            
            if not text_content.strip():
                return
            
            # Normalizar el texto extraído
            text_content = norm(text_content)
            
            # Dividir en párrafos
            paragraphs = split_into_paragraphs(text_content)
            
            for i, paragraph in enumerate(paragraphs):
                text_valid, issues = quality_check(paragraph)
                
                if not text_valid:
                    continue
                
                readability = calculate_readability(paragraph)
                medical = calculate_medical_density(paragraph)
                
                yield {
                    "texto_original": paragraph,
                    "resumen": "",
                    "source": fp.parent.name,
                    "doc_id": f"{source_dataset}_{fp.stem}#pdf_p{i+1}",
                    "split": split,
                    "label": label,
                    "source_dataset": source_dataset,
                    "source_bucket": source_bucket,
                    "has_pair": False,
                    "word_count_src": len(paragraph.split()),
                    "word_count_pls": 0,
                    "flesch_score": readability['flesch_reading_ease'],
                    "avg_word_length": readability['avg_word_length'],
                    "medical_density": medical['medical_terms_ratio'],
                    "quality_issues": "|".join(issues) if issues else "",
                    "processing_date": datetime.now().isoformat(),
                }
                
    except Exception as e:
        print(f"[WARN] Error procesando PDF {fp}: {e}", file=sys.stderr)


# ==================== ITERADOR MAESTRO ====================

def parse_files() -> Iterator[Dict[str, any]]:
    """Itera sobre todos los archivos con parsers mejorados."""
    for fp in RAW.rglob("*"):
        if not fp.is_file():
            continue
        suf = fp.suffix.lower()
        if suf in SKIP_SUFFIXES or suf not in ALLOWED_SUFFIXES:
            continue
        try:
            if suf == ".txt":
                yield from iter_txt(fp)
            elif suf == ".csv":
                yield from iter_csv(fp)
            elif suf == ".jsonl":
                yield from iter_jsonl(fp)
            elif suf == ".json":
                yield from iter_json(fp)
            elif suf == ".pdf":
                yield from iter_pdf(fp)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}", file=sys.stderr)


# ==================== GUARDADO ====================

def save_streaming_jsonl(stem: str = "dataset_clean") -> Path:
    """Guarda dataset con deduplicación y estadísticas mejoradas."""
    out_jsonl = PROC / f"{stem}.jsonl"
    seen = set()
    
    stats = {
        'total_processed': 0,
        'kept': 0,
        'duplicates': 0,
        'invalid_quality': 0,
        'by_source': {},
        'by_label': {},
        'with_pairs': 0,
        'avg_flesch': 0.0,
    }
    
    flesch_scores = []
    
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in parse_files():
            stats['total_processed'] += 1
            
            # Deduplicación
            key = (r.get("texto_original", "") + "||" + r.get("resumen", "")).encode("utf-8")
            h = hashlib.sha256(key).hexdigest()
            
            if h in seen:
                stats['duplicates'] += 1
                continue
            
            seen.add(h)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats['kept'] += 1
            
            # Estadísticas
            source = r.get('source_dataset', 'unknown')
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            
            label = r.get('label', 'unlabeled')
            stats['by_label'][label] = stats['by_label'].get(label, 0) + 1
            
            if r.get('has_pair'):
                stats['with_pairs'] += 1
            
            if r.get('flesch_score'):
                flesch_scores.append(r['flesch_score'])
            
            # Log progreso
            if stats['total_processed'] % 10000 == 0:
                print(f"[PROGRESS] Procesados: {stats['total_processed']}, "
                      f"Mantenidos: {stats['kept']}, "
                      f"Duplicados: {stats['duplicates']}")
    
    if flesch_scores:
        stats['avg_flesch'] = sum(flesch_scores) / len(flesch_scores)
    
    # Reporte final
    print("\n" + "="*80)
    print("RESULTADO DEL PROCESAMIENTO")
    print("="*80)
    print(f"Archivo generado: {out_jsonl}")
    print(f"\nEstadísticas:")
    print(f"  Total procesados: {stats['total_processed']:,}")
    print(f"  Mantenidos: {stats['kept']:,}")
    print(f"  Duplicados eliminados: {stats['duplicates']:,}")
    print(f"  Con pares (has_pair=True): {stats['with_pairs']:,} ({stats['with_pairs']/stats['kept']*100:.1f}%)")
    print(f"  Flesch Reading Ease promedio: {stats['avg_flesch']:.1f}")
    print(f"\nPor fuente:")
    for source, count in sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {source}: {count:,}")
    print(f"\nPor etiqueta:")
    for label, count in sorted(stats['by_label'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {label}: {count:,}")
    print("="*80)
    
    # Guardar estadísticas
    stats_file = PROC / f"{stem}_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nEstadísticas guardadas en: {stats_file}")
    
    return out_jsonl


def jsonl_to_csv(jsonl_path: Path, stem: str = "dataset_clean") -> Path:
    """Convierte JSONL a CSV con todas las columnas."""
    import pandas as pd
    
    df = pd.read_json(jsonl_path, lines=True)
    
    # Orden de columnas
    cols = [
        c for c in [
            "texto_original", "resumen", "source", "doc_id", "split", "label",
            "source_dataset", "source_bucket", "has_pair",
            "word_count_src", "word_count_pls",
            "flesch_score", "avg_word_length", "medical_density",
            "quality_issues", "processing_date"
        ] if c in df.columns
    ]
    
    out_csv = PROC / f"{stem}.csv"
    df.to_csv(out_csv, index=False, columns=cols)
    print(f"\nCSV generado: {out_csv} ({len(df):,} filas)")
    
    return out_csv


# ==================== MAIN ====================

def main():
    """Ejecuta el pipeline completo."""
    print("\n" + "="*80)
    print("GENERACIÓN DE DATASET v2 - CALIDAD MEJORADA")
    print("="*80)
    print("\nMEJORAS IMPLEMENTADAS:")
    print("  Procesamiento de PDFs y archivos de texto")
    print("  Fragmentación por párrafos (no líneas)")
    print("  Filtros de calidad estrictos (100-5000 chars)")
    print("  Validación de idioma inglés")
    print("  Métricas de legibilidad (Flesch)")
    print("  Detección de errores de OCR")
    print("  Metadata enriquecida")
    print("  Quality flags por registro")
    print("="*80 + "\n")
    
    # Verificar librerías opcionales
    if not LANGDETECT_AVAILABLE:
        print("WARNING: langdetect no disponible - no se validará idioma")
    if not TEXTSTAT_AVAILABLE:
        print("WARNING: textstat no disponible - no se calcularán métricas de legibilidad")
    if not PDF_AVAILABLE:
        print("WARNING: PyPDF2 no disponible - no se procesarán archivos PDF")
    
    print("\nIniciando procesamiento...\n")
    
    # Procesar
    jl = save_streaming_jsonl("dataset_clean")
    csv_path = jsonl_to_csv(jl, "dataset_clean")
    
    print("\n" + "="*80)
    print("PROCESAMIENTO COMPLETADO")
    print("="*80)
    print(f"\nArchivos generados:")
    print(f"  - {jl}")
    print(f"  - {csv_path}")
    print(f"  - {PROC}/dataset_clean_stats.json")
    print("\nSiguiente paso: python src/data/split_dataset.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


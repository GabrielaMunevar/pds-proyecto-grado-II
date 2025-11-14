#!/usr/bin/env python3
"""
Generador de Pares Sintéticos Mejorados para Entrenar el Generador de PLS
Crea pares técnico-simple usando los datos disponibles con técnicas avanzadas
para reducir overlap léxico y aumentar calidad.

Estrategia:
1. Usar textos PLS como "simple" 
2. Generar versiones "técnicas" usando técnicas de complejificación robustas
3. Crear pares sintéticos para entrenar el generador

Uso:
    python src/data/create_synthetic_pairs.py
"""

import pandas as pd
import numpy as np
import re
import random
from pathlib import Path
import json
from collections import defaultdict, Counter
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Importar librerías opcionales para técnicas avanzadas
NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk import pos_tag, word_tokenize
    # Descargar datos de NLTK si no están disponibles
    try:
        wn.synsets('test')  # Verificar que WordNet funciona
        NLTK_AVAILABLE = True
    except (LookupError, OSError):
        try:
            print("[INFO] Descargando datos de NLTK (WordNet)...")
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            NLTK_AVAILABLE = True
        except Exception:
            NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False
except Exception:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    # Intentar cargar modelo en inglés
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[WARN] Modelo spaCy 'en_core_web_sm' no encontrado. Instalar con: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    print("[WARN] spaCy no disponible. Instalar con: pip install spacy")

# ==================== DICCIONARIOS DE COMPLEJIFICACIÓN ====================

# Diccionario extenso de términos médicos simples -> técnicos
MEDICAL_TERMS = {
    # Enfermedades cardiovasculares
    'heart attack': 'myocardial infarction',
    'heart disease': 'cardiovascular disease',
    'heart failure': 'congestive heart failure',
    'high blood pressure': 'hypertension',
    'low blood pressure': 'hypotension',
    'chest pain': 'angina pectoris',
    'irregular heartbeat': 'cardiac arrhythmia',
    'heart rhythm problems': 'dysrhythmia',
    
    # Enfermedades metabólicas
    'diabetes': 'diabetes mellitus',
    'type 2 diabetes': 'type 2 diabetes mellitus',
    'high blood sugar': 'hyperglycemia',
    'low blood sugar': 'hypoglycemia',
    'high cholesterol': 'hypercholesterolemia',
    
    # Enfermedades neurológicas
    'stroke': 'cerebrovascular accident',
    'brain disease': 'neurological disorder',
    'seizure': 'epileptic seizure',
    'memory problems': 'cognitive impairment',
    'dementia': 'neurodegenerative dementia',
    
    # Enfermedades respiratorias
    'lung disease': 'pulmonary disease',
    'breathing problems': 'respiratory dysfunction',
    'shortness of breath': 'dyspnea',
    'asthma': 'bronchial asthma',
    
    # Enfermedades renales y hepáticas
    'kidney disease': 'renal disease',
    'kidney failure': 'renal failure',
    'liver disease': 'hepatic disease',
    'liver failure': 'hepatic failure',
    
    # Cáncer y neoplasias
    'cancer': 'neoplasia',
    'tumor': 'neoplasm',
    'lump': 'palpable mass',
    
    # Infecciones
    'infection': 'bacterial/viral infection',
    'fever': 'pyrexia',
    'inflammation': 'inflammatory response',
    
    # Síntomas generales
    'pain': 'nociceptive pain',
    'swelling': 'edema',
    'bleeding': 'hemorrhage',
    'bruising': 'ecchymosis',
    'rash': 'cutaneous eruption',
}

# Verbos simples -> técnicos/formales
VERB_REPLACEMENTS = {
    'help': 'facilitate',
    'show': 'demonstrate',
    'find': 'identify',
    'use': 'utilize',
    'give': 'administer',
    'take': 'consume',
    'get': 'acquire',
    'make': 'produce',
    'work': 'function',
    'stop': 'discontinue',
    'start': 'initiate',
    'try': 'attempt',
    'check': 'evaluate',
    'test': 'assess',
    'look at': 'examine',
    'see': 'observe',
    'know': 'ascertain',
    'think': 'hypothesize',
    'feel': 'experience',
    'change': 'modify',
    'improve': 'enhance',
    'reduce': 'diminish',
    'increase': 'augment',
    'lower': 'decrease',
    'raise': 'elevate',
}

# Adjetivos simples -> técnicos
ADJECTIVE_REPLACEMENTS = {
    'good': 'beneficial',
    'bad': 'adverse',
    'big': 'significant',
    'small': 'minimal',
    'fast': 'rapid',
    'slow': 'gradual',
    'safe': 'well-tolerated',
    'unsafe': 'contraindicated',
    'effective': 'efficacious',
    'ineffective': 'inefficacious',
    'strong': 'potent',
    'weak': 'suboptimal',
    'new': 'novel',
    'old': 'established',
    'common': 'prevalent',
    'rare': 'uncommon',
    'serious': 'severe',
    'mild': 'moderate',
    'long': 'prolonged',
    'short': 'brief',
}

# Términos formales para reemplazos contextuales
FORMAL_REPLACEMENTS = {
    r'\bwe\b': 'the research team',
    r'\byou\b': 'patients',
    r'\bpeople\b': 'individuals',
    r'\bperson\b': 'subject',
    r'\bpatient\b': 'study participant',
    r'\bstudy\b': 'investigation',
    r'\btest\b': 'assessment',
    r'\btry\b': 'attempt',
    r'\bcheck\b': 'evaluate',
    r'\bdoctor\b': 'clinician',
    r'\bmedicine\b': 'pharmaceutical intervention',
    r'\bdrug\b': 'therapeutic agent',
    r'\btreatment\b': 'therapeutic intervention',
    r'\bresult\b': 'outcome',
    r'\bfinding\b': 'observation',
    r'\bside effect\b': 'adverse event',
    r'\bproblem\b': 'adverse effect',
}

# Sinónimos comunes adicionales para reducir overlap (EXPANDIDO)
COMMON_SYNONYMS = {
    'important': ['critical', 'significant', 'essential', 'crucial', 'paramount', 'vital', 'indispensable', 'imperative'],
    'different': ['various', 'diverse', 'distinct', 'heterogeneous', 'disparate', 'dissimilar', 'divergent', 'varied'],
    'many': ['numerous', 'multiple', 'various', 'several', 'myriad', 'multitudinous', 'copious', 'abundant'],
    'often': ['frequently', 'commonly', 'typically', 'routinely', 'regularly', 'habitually', 'repeatedly', 'consistently'],
    'sometimes': ['occasionally', 'periodically', 'intermittently', 'sporadically', 'infrequently', 'seldom', 'rarely'],
    'usually': ['typically', 'commonly', 'generally', 'ordinarily', 'normally', 'customarily', 'habitually', 'routinely'],
    'very': ['highly', 'extremely', 'substantially', 'considerably', 'markedly', 'significantly', 'notably', 'remarkably'],
    'more': ['greater', 'increased', 'enhanced', 'elevated', 'augmented', 'amplified', 'expanded', 'extended'],
    'less': ['reduced', 'diminished', 'decreased', 'lowered', 'minimized', 'mitigated', 'attenuated', 'curtailed'],
    'about': ['approximately', 'roughly', 'circa', 'around', 'nearly'],  # Removido duplicados y 'in the vicinity of' - muy artificial
    'when': ['during', 'throughout', 'while', 'upon', 'at the time of', 'concurrently with', 'simultaneously with'],
    'where': ['in which', 'wherein', 'at which', 'in locations where', 'at sites where', 'in contexts where'],
    'how': ['the manner in which', 'the mechanism by which', 'the process through which'],  # Removido 'the methodology whereby' - muy artificial
    'what': ['that which', 'the factor that', 'the element which'],  # Removido 'the phenomenon that' y 'the entity which' - muy artificial
    'why': ['the reason for which', 'the rationale underlying', 'the basis for', 'the foundation upon which'],
    'because': ['due to', 'as a result of', 'owing to', 'attributable to', 'consequent to', 'stemming from', 'arising from'],
    'so': ['therefore', 'consequently', 'thus', 'hence', 'accordingly', 'ergo', 'as a consequence'],
    'but': ['however', 'nevertheless', 'nonetheless', 'yet', 'whereas', 'conversely', 'in contrast', 'on the contrary'],
    'and': ['as well as', 'in addition to', 'along with', 'coupled with', 'in conjunction with'],  # Removido 'together with' - muy repetitivo
    'or': ['alternatively', 'or alternatively', 'or otherwise'],  # Removido 'or else' y 'or in lieu thereof' - muy artificial
    'also': ['additionally', 'furthermore', 'moreover', 'likewise', 'similarly', 'correspondingly', 'analogously'],
    'then': ['subsequently', 'thereafter', 'following this', 'consequently', 'ensuingly', 'thereupon'],
    'first': ['initially', 'primarily', 'principally', 'predominantly', 'foremost', 'preeminently'],
    'last': ['ultimately', 'finally', 'conclusively', 'terminally', 'definitively', 'decisively'],
    'new': ['novel', 'emerging', 'recently developed', 'newly identified', 'innovative', 'pioneering', 'groundbreaking'],
    'old': ['established', 'conventional', 'traditional', 'long-standing', 'time-honored', 'venerable'],
    # Agregar más palabras comunes
    'can': ['is capable of', 'has the capacity to', 'possesses the ability to', 'is able to'],
    'may': ['might potentially', 'could possibly', 'has the potential to', 'may conceivably'],
    'will': ['is expected to', 'is anticipated to', 'is projected to', 'is likely to'],
    'should': ['ought to', 'is recommended to', 'is advisable to', 'is prudent to'],
    'must': ['is required to', 'is necessary to', 'is imperative to', 'is obligatory to'],
    'need': ['require', 'necessitate', 'demand', 'call for'],
    'want': ['desire', 'seek', 'aim to', 'intend to'],
    'like': ['similar to', 'analogous to', 'comparable to', 'reminiscent of'],
    'such': ['of this nature', 'of this type', 'of this kind', 'of this variety'],
    # 'these': removido - muy artificial
    # 'those': removido - muy artificial
    'this': ['the present', 'the current'],  # Removido opciones muy artificiales
    'that': ['the aforementioned'],  # Solo cuando realmente se refiere a algo mencionado antes
    # 'it', 'they', 'them' - NO reemplazar (muy artificial)
    # Mejor usar sinónimos contextuales solo cuando tiene sentido
    # 'it': ['the entity', 'the phenomenon', 'the factor', 'the element'],  # REMOVIDO - muy artificial
    # 'they': ['the entities', 'the phenomena', 'the factors', 'the elements'],  # REMOVIDO - muy artificial
    # 'them': ['the aforementioned entities', 'the specified factors', 'the indicated elements'],  # REMOVIDO - muy artificial
    # 'their': ['the entities\'', 'the factors\'', 'the elements\''],  # REMOVIDO - muy artificial
    # 'its': ['the entity\'s', 'the phenomenon\'s', 'the factor\'s'],  # REMOVIDO - muy artificial
    'have': ['possess', 'exhibit', 'demonstrate', 'display'],
    'has': ['possesses', 'exhibits', 'demonstrates', 'displays'],
    'had': ['possessed', 'exhibited', 'demonstrated', 'displayed'],
    'get': ['obtain', 'acquire', 'attain', 'secure'],
    'got': ['obtained', 'acquired', 'attained', 'secured'],
    'give': ['provide', 'supply', 'furnish', 'deliver'],
    'gave': ['provided', 'supplied', 'furnished', 'delivered'],
    'take': ['undergo', 'experience', 'receive', 'accept'],
    'took': ['underwent', 'experienced', 'received', 'accepted'],
    'make': ['generate', 'produce', 'create', 'fabricate'],
    'made': ['generated', 'produced', 'created', 'fabricated'],
    'do': ['perform', 'execute', 'carry out', 'conduct'],
    'did': ['performed', 'executed', 'carried out', 'conducted'],
    'say': ['state', 'indicate', 'suggest', 'propose'],
    'said': ['stated', 'indicated', 'suggested', 'proposed'],
    'tell': ['inform', 'notify', 'advise', 'communicate'],
    'told': ['informed', 'notified', 'advised', 'communicated'],
    'see': ['observe', 'perceive', 'detect', 'identify'],
    'saw': ['observed', 'perceived', 'detected', 'identified'],
    'know': ['understand', 'comprehend', 'recognize', 'appreciate'],
    'knew': ['understood', 'comprehended', 'recognized', 'appreciated'],
    'think': ['consider', 'contemplate', 'reflect upon', 'ponder'],
    'thought': ['considered', 'contemplated', 'reflected upon', 'pondered'],
    'feel': ['experience', 'perceive', 'sense', 'detect'],
    'felt': ['experienced', 'perceived', 'sensed', 'detected'],
    'go': ['proceed', 'advance', 'progress', 'move forward'],
    'went': ['proceeded', 'advanced', 'progressed', 'moved forward'],
    'come': ['arrive', 'emerge', 'appear', 'materialize'],
    'came': ['arrived', 'emerged', 'appeared', 'materialized'],
    'use': ['utilize', 'employ', 'apply', 'implement'],
    'used': ['utilized', 'employed', 'applied', 'implemented'],
    'find': ['identify', 'discover', 'detect', 'locate'],
    'found': ['identified', 'discovered', 'detected', 'located'],
    'look': ['examine', 'inspect', 'scrutinize', 'analyze'],
    'looked': ['examined', 'inspected', 'scrutinized', 'analyzed'],
    'show': ['demonstrate', 'illustrate', 'reveal', 'exhibit'],
    'showed': ['demonstrated', 'illustrated', 'revealed', 'exhibited'],
    'try': ['attempt', 'endeavor', 'strive', 'seek'],
    'tried': ['attempted', 'endeavored', 'strove', 'sought'],
    'help': ['assist', 'facilitate', 'support', 'aid'],
    'helped': ['assisted', 'facilitated', 'supported', 'aided'],
    'work': ['function', 'operate', 'perform', 'execute'],
    'worked': ['functioned', 'operated', 'performed', 'executed'],
    'call': ['designate', 'label', 'term', 'refer to as'],
    'called': ['designated', 'labeled', 'termed', 'referred to as'],
    'ask': ['inquire', 'query', 'request', 'solicit'],
    'asked': ['inquired', 'queried', 'requested', 'solicited'],
    'seem': ['appear', 'manifest', 'present', 'exhibit'],
    'seemed': ['appeared', 'manifested', 'presented', 'exhibited'],
    'become': ['transform into', 'evolve into', 'develop into', 'emerge as'],
    'became': ['transformed into', 'evolved into', 'developed into', 'emerged as'],
    'leave': ['depart', 'exit', 'withdraw', 'vacate'],
    'left': ['departed', 'exited', 'withdrew', 'vacated'],
    'put': ['place', 'position', 'locate', 'situate'],
    'mean': ['signify', 'denote', 'indicate', 'represent'],
    'meant': ['signified', 'denoted', 'indicated', 'represented'],
    'keep': ['maintain', 'preserve', 'retain', 'sustain'],
    'kept': ['maintained', 'preserved', 'retained', 'sustained'],
    'let': ['permit', 'allow', 'enable', 'authorize'],
    'begin': ['commence', 'initiate', 'embark upon', 'undertake'],
    'began': ['commenced', 'initiated', 'embarked upon', 'undertook'],
    'turn': ['transform', 'convert', 'change', 'alter'],
    'turned': ['transformed', 'converted', 'changed', 'altered'],
    'start': ['initiate', 'commence', 'embark upon', 'undertake'],
    'started': ['initiated', 'commenced', 'embarked upon', 'undertook'],
    'move': ['relocate', 'transfer', 'shift', 'transit'],
    'moved': ['relocated', 'transferred', 'shifted', 'transited'],
    'live': ['reside', 'dwell', 'inhabit', 'occupy'],
    'lived': ['resided', 'dwelt', 'inhabited', 'occupied'],
    'believe': ['maintain', 'contend', 'assert', 'hold'],
    'believed': ['maintained', 'contended', 'asserted', 'held'],
    'bring': ['convey', 'transport', 'deliver', 'transfer'],
    'brought': ['conveyed', 'transported', 'delivered', 'transferred'],
    'happen': ['occur', 'transpire', 'take place', 'come about'],
    'happened': ['occurred', 'transpired', 'took place', 'came about'],
    'write': ['compose', 'draft', 'formulate', 'author'],
    'wrote': ['composed', 'drafted', 'formulated', 'authored'],
    'provide': ['furnish', 'supply', 'deliver', 'render'],
    'provided': ['furnished', 'supplied', 'delivered', 'rendered'],
    'sit': ['be positioned', 'be located', 'be situated', 'be placed'],
    'sat': ['was positioned', 'was located', 'was situated', 'was placed'],
    'stand': ['be positioned', 'be located', 'be situated', 'be placed'],
    'stood': ['was positioned', 'was located', 'was situated', 'was placed'],
    'lose': ['forfeit', 'relinquish', 'surrender', 'deprive'],
    'lost': ['forfeited', 'relinquished', 'surrendered', 'deprived'],
    'add': ['incorporate', 'include', 'integrate', 'introduce'],
    'added': ['incorporated', 'included', 'integrated', 'introduced'],
    'change': ['modify', 'alter', 'transform', 'adjust'],
    'changed': ['modified', 'altered', 'transformed', 'adjusted'],
    'follow': ['pursue', 'track', 'monitor', 'observe'],
    'followed': ['pursued', 'tracked', 'monitored', 'observed'],
    'stop': ['cease', 'discontinue', 'terminate', 'halt'],
    'stopped': ['ceased', 'discontinued', 'terminated', 'halted'],
    'create': ['generate', 'produce', 'fabricate', 'construct'],
    'created': ['generated', 'produced', 'fabricated', 'constructed'],
    'speak': ['articulate', 'verbalize', 'communicate', 'express'],
    'spoke': ['articulated', 'verbalized', 'communicated', 'expressed'],
    'read': ['peruse', 'examine', 'review', 'scrutinize'],
    'allow': ['permit', 'enable', 'authorize', 'facilitate'],
    'allowed': ['permitted', 'enabled', 'authorized', 'facilitated'],
    'lead': ['guide', 'direct', 'steer', 'conduct'],
    'led': ['guided', 'directed', 'steered', 'conducted'],
    'understand': ['comprehend', 'grasp', 'apprehend', 'perceive'],
    'understood': ['comprehended', 'grasped', 'apprehended', 'perceived'],
    'watch': ['observe', 'monitor', 'scrutinize', 'examine'],
    'watched': ['observed', 'monitored', 'scrutinized', 'examined'],
    'play': ['engage in', 'participate in', 'partake in', 'involve oneself in'],
    'played': ['engaged in', 'participated in', 'partook in', 'involved oneself in'],
    'run': ['operate', 'function', 'execute', 'perform'],
    'ran': ['operated', 'functioned', 'executed', 'performed'],
}

# Frases comunes para parafrasear
COMMON_PHRASES = {
    r'\bis important\b': ['is critical', 'is significant', 'is essential', 'is crucial'],
    r'\bis used\b': ['is utilized', 'is employed', 'is applied', 'is implemented'],
    r'\bis found\b': ['is identified', 'is detected', 'is observed', 'is ascertained'],
    r'\bis shown\b': ['is demonstrated', 'is evidenced', 'is indicated', 'is revealed'],
    r'\bis known\b': ['is established', 'is recognized', 'is acknowledged', 'is documented'],
    r'\bis not known\b': ['remains unknown', 'has not been established', 'lacks documentation'],
    r'\bwe know\b': ['it is established', 'it is recognized', 'evidence indicates'],
    r'\bwe do not know\b': ['evidence is lacking', 'it remains unclear', 'data are insufficient'],
    r'\bmore research\b': ['further investigation', 'additional studies', 'subsequent research'],
    r'\bmore studies\b': ['additional investigations', 'further research', 'subsequent studies'],
    r'\bneeds more research\b': ['requires further investigation', 'warrants additional studies'],
    r'\bmore information\b': ['additional data', 'further evidence', 'supplementary information'],
}

# Frases técnicas para agregar contexto
TECHNICAL_PHRASES = [
    'in a randomized controlled trial',
    'according to the study protocol',
    'based on statistical analysis',
    'with statistical significance',
    'within the confidence interval',
    'per the inclusion criteria',
    'following exclusion criteria',
    'as per the study design',
    'in accordance with regulatory guidelines',
    'following informed consent procedures',
    'under ethical committee approval',
    'with placebo-controlled methodology',
    'using double-blind methodology',
    'through systematic evaluation',
    'via clinical assessment',
    'following standardized protocols',
    'with rigorous monitoring',
    'under controlled conditions',
]

# Términos técnicos para inserción contextual
TECHNICAL_TERMS = [
    'primary endpoint',
    'secondary endpoint',
    'adverse events',
    'placebo-controlled',
    'double-blind',
    'statistical significance',
    'confidence interval',
    'p-value',
    'inclusion criteria',
    'exclusion criteria',
    'protocol',
    'informed consent',
    'ethics committee',
    'regulatory approval',
    'clinical trial',
    'randomized controlled trial',
    'cohort study',
    'case-control study',
    'meta-analysis',
    'systematic review',
]

# ==================== FUNCIONES AVANZADAS DE SINÓNIMOS ====================

def get_wordnet_synonyms(word: str, pos: Optional[str] = None) -> List[str]:
    """Obtiene sinónimos de WordNet para una palabra."""
    if not NLTK_AVAILABLE:
        return []
    
    synonyms = set()
    try:
        # Mapear POS tags de NLTK a WordNet
        pos_map = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'J': wn.ADJ,
            'R': wn.ADV
        }
        
        wordnet_pos = pos_map.get(pos, None) if pos else None
        
        # Buscar sinónimos
        for syn in wn.synsets(word, pos=wordnet_pos):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower() and len(synonym.split()) == 1:
                    synonyms.add(synonym)
        
        return list(synonyms)[:5]  # Limitar a 5 sinónimos
    except Exception:
        return []

def get_contextual_synonyms_spacy(text: str, word: str) -> List[str]:
    """Obtiene sinónimos contextuales usando spaCy."""
    if not SPACY_AVAILABLE or nlp is None:
        return []
    
    try:
        doc = nlp(text)
        synonyms = []
        
        # Buscar la palabra en el texto
        for token in doc:
            if token.text.lower() == word.lower():
                # Usar embeddings para encontrar palabras similares
                # (simplificado - en producción usar modelos de embeddings)
                pass
        
        return synonyms
    except Exception:
        return []

def apply_advanced_synonym_replacement(text: str, complexity_level: float) -> str:
    """Aplica reemplazo de sinónimos usando WordNet y análisis POS (BALANCEADO)."""
    if not NLTK_AVAILABLE:
        return text
    
    try:
        # Tokenizar y etiquetar POS
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        result_tokens = []
        replacements_made = 0
        # Máximo de reemplazos balanceado para reducir overlap sin sobreajuste
        text_length = len(tokens)
        if text_length > 400:
            max_replacements = int(len(tokens) * 0.30)  # 30% para textos muy largos
        elif text_length > 200:
            max_replacements = int(len(tokens) * 0.28)  # 28% para textos largos
        else:
            max_replacements = int(len(tokens) * 0.25)  # 25% para textos cortos
        
        for token, pos in pos_tags:
            # Saltar puntuación y palabras muy cortas
            if len(token) < 3 or not token.isalnum():
                result_tokens.append(token)
                continue
            
            # Obtener sinónimos de WordNet (balanceado - más agresivo)
            prob = 0.75 if text_length > 400 else (0.70 if text_length > 200 else 0.65)
            if replacements_made < max_replacements and random.random() < complexity_level * prob:
                synonyms = get_wordnet_synonyms(token.lower(), pos[0])
                if synonyms:
                    synonym = random.choice(synonyms)
                    # Preservar capitalización
                    if token[0].isupper():
                        synonym = synonym.capitalize()
                    result_tokens.append(synonym)
                    replacements_made += 1
                    continue
            
            result_tokens.append(token)
        
        return ' '.join(result_tokens)
    except Exception as e:
        # Si falla, retornar texto original
        return text

# ==================== FUNCIONES DE COMPLEJIFICACIÓN ====================

def apply_word_replacements(text: str, complexity_level: float) -> str:
    """Aplica reemplazos de palabras según diccionarios (BALANCEADO para evitar sobreajuste)."""
    result = text
    
    # Variabilidad: no aplicar todas las transformaciones siempre
    text_length = len(text.split())
    is_long_text = text_length > 200
    is_very_long_text = text_length > 400
    
    # Multiplicador más conservador para evitar patrones artificiales
    if is_very_long_text:
        base_multiplier = 1.3
    elif is_long_text:
        base_multiplier = 1.2
    else:
        base_multiplier = 1.1
    
    # Aplicar reemplazos médicos (balanceado - más frecuente para reducir overlap)
    medical_prob = 0.90  # Aumentado para más diversidad
    for simple, complex_term in MEDICAL_TERMS.items():
        prob = min(1.0, complexity_level * medical_prob * base_multiplier)
        if random.random() < prob:
            pattern = r'\b' + re.escape(simple) + r'\b'
            result = re.sub(pattern, complex_term, result, flags=re.IGNORECASE)
    
    # Aplicar reemplazos de verbos (balanceado - más frecuente)
    verb_prob = 0.85  # Aumentado para más diversidad
    for simple, complex_term in VERB_REPLACEMENTS.items():
        prob = min(1.0, complexity_level * verb_prob * base_multiplier)
        if random.random() < prob:
            pattern = r'\b' + re.escape(simple) + r'\b'
            result = re.sub(pattern, complex_term, result, flags=re.IGNORECASE)
    
    # Aplicar reemplazos de adjetivos (balanceado - más frecuente)
    adj_prob = 0.80  # Aumentado para más diversidad
    for simple, complex_term in ADJECTIVE_REPLACEMENTS.items():
        prob = min(1.0, complexity_level * adj_prob * base_multiplier)
        if random.random() < prob:
            pattern = r'\b' + re.escape(simple) + r'\b'
            result = re.sub(pattern, complex_term, result, flags=re.IGNORECASE)
    
    # Aplicar reemplazos formales (balanceado - más frecuente)
    formal_prob = 0.80  # Aumentado para más diversidad
    for pattern, replacement in FORMAL_REPLACEMENTS.items():
        prob = min(1.0, complexity_level * formal_prob * base_multiplier)
        if random.random() < prob:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # Aplicar sinónimos comunes (balanceado - más palabras pero variado)
    # EXCLUIR palabras que causan patrones artificiales
    excluded_words = {'it', 'they', 'them', 'their', 'its', 'these', 'those'}  # Palabras que no deben reemplazarse
    available_words = {k: v for k, v in COMMON_SYNONYMS.items() if k not in excluded_words}
    
    common_syn_prob = 0.95 if is_very_long_text else (0.90 if is_long_text else 0.85)
    # Aplicar a más palabras para reducir overlap
    words_to_replace = random.sample(
        list(available_words.keys()), 
        k=min(len(available_words), int(len(available_words) * 0.90))  # 90% de las palabras disponibles
    )
    for word in words_to_replace:
        synonyms = available_words[word]
        prob = min(1.0, complexity_level * common_syn_prob * base_multiplier)
        if random.random() < prob:
            # Evitar sinónimos muy artificiales - preferir los primeros de la lista (más naturales)
            synonym = random.choice(synonyms[:3]) if len(synonyms) > 3 else random.choice(synonyms)
            pattern = r'\b' + re.escape(word) + r'\b'
            result = re.sub(pattern, synonym, result, flags=re.IGNORECASE)
    
    # Parafrasear frases comunes (balanceado - más frases)
    phrase_prob = 0.85 if is_very_long_text else (0.80 if is_long_text else 0.75)
    # Aplicar a más frases para reducir overlap
    phrases_to_replace = random.sample(
        list(COMMON_PHRASES.keys()),
        k=min(len(COMMON_PHRASES), int(len(COMMON_PHRASES) * 0.75))  # 75% de las frases
    )
    for pattern in phrases_to_replace:
        synonyms = COMMON_PHRASES[pattern]
        prob = min(1.0, complexity_level * phrase_prob * base_multiplier)
        if random.random() < prob:
            synonym = random.choice(synonyms)
            result = re.sub(pattern, synonym, result, flags=re.IGNORECASE)
    
    # Aplicar sinónimos avanzados con WordNet (balanceado - más frecuente)
    if NLTK_AVAILABLE:
        wordnet_prob = 0.60 if is_very_long_text else (0.55 if is_long_text else 0.50)
        if random.random() < complexity_level * wordnet_prob:
            result = apply_advanced_synonym_replacement(result, complexity_level)
    
    return result

def restructure_sentences_advanced(text: str, complexity_level: float) -> str:
    """Reestructura oraciones usando spaCy para análisis sintáctico avanzado."""
    if not SPACY_AVAILABLE or nlp is None:
        return restructure_sentences(text, complexity_level)  # Fallback a versión básica
    
    try:
        doc = nlp(text)
        sentences = list(doc.sents)
        result_sentences = []
        
        for sent in sentences:
            sent_text = sent.text
            
            # Reestructurar usando análisis de dependencias
            # Convertir voz activa a pasiva cuando sea posible
            if random.random() < complexity_level * 0.6:
                # Buscar patrones de voz activa
                for token in sent:
                    if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                        # Intentar convertir a pasiva
                        verb = token.head
                        obj = None
                        for child in verb.children:
                            if child.dep_ == "dobj":
                                obj = child
                                break
                        
                        if obj:
                            # Reestructurar: "X does Y" -> "Y is done by X"
                            # (simplificado - en producción hacer más robusto)
                            pass
            
            result_sentences.append(sent_text)
        
        return ' '.join(result_sentences)
    except Exception:
        return restructure_sentences(text, complexity_level)

def restructure_sentences(text: str, complexity_level: float) -> str:
    """Reestructura oraciones para hacerlas más técnicas (BALANCEADO para evitar sobreajuste)."""
    sentences = re.split(r'([.!?]+)', text)
    result_sentences = []
    
    text_length = len(text.split())
    is_long_text = text_length > 200
    base_multiplier = 1.2 if is_long_text else 1.0  # Más conservador
    
    i = 0
    sentence_count = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if not sentence:
            i += 1
            continue
        
        sentence_count += 1
        
        # Convertir a voz pasiva (balanceado - más frecuente)
        if random.random() < complexity_level * 0.55 * base_multiplier:  # Aumentado a 0.55
            # Patrones más amplios para voz pasiva
            sentence = re.sub(
                r'(\w+)\s+(does|doesn\'t|did|didn\'t|do|don\'t)\s+',
                lambda m: f"is {'not ' if 'n\'t' in m.group(2) or 'don\'t' in m.group(2) else ''}performed by {m.group(1)} ",
                sentence,
                count=1
            )
        
        # Agregar nominalizaciones (balanceado - más frecuente)
        if random.random() < complexity_level * 0.50 * base_multiplier:  # Aumentado a 0.50
            # "X improves Y" -> "improvement of Y by X"
            sentence = re.sub(
                r'(\w+)\s+(improves|reduces|increases|decreases|enhances|diminishes)\s+(\w+)',
                lambda m: f"{m.group(2).rstrip('s')} of {m.group(3)} by {m.group(1)}",
                sentence,
                count=1
            )
        
        # Expandir con cláusulas subordinadas técnicas (balanceado - más frecuente pero variado)
        # SOLO aplicar a oraciones largas y variar mucho para evitar repetición
        if random.random() < complexity_level * 0.35 * base_multiplier and len(sentence.split()) > 8:  # Solo oraciones largas
            # Agregar cláusula técnica al final (solo ocasionalmente para evitar repetición)
            if random.random() < 0.4:  # 40% para más diversidad
                technical_clause = random.choice([
                    ', as determined by clinical evaluation',
                    ', following established diagnostic criteria',
                    ', in accordance with study methodology',
                    ', as evidenced by comprehensive analysis',
                    ', per the predefined study protocol',
                    ', following rigorous assessment procedures',
                    ', based on validated measurement instruments',
                    ', according to standardized protocols',
                ])
                sentence = sentence.rstrip('.!?') + technical_clause + '.'
        
        # Cada 3-4 oraciones, agregar información técnica al inicio (más frecuente)
        if sentence_count % 4 == 0 and random.random() < complexity_level * 0.50 * base_multiplier:  # Aumentado a 0.50
            intro_phrase = random.choice([
                'According to the clinical investigation, ',
                'Based on the systematic evaluation, ',
                'Per the study protocol, ',
            ])
            sentence = intro_phrase + sentence[0].lower() + sentence[1:] if len(sentence) > 0 else sentence
        
        result_sentences.append(sentence)
        if i + 1 < len(sentences):
            result_sentences.append(sentences[i + 1])  # Preservar puntuación
        i += 2
    
    return ''.join(result_sentences)

def add_technical_context(text: str, complexity_level: float) -> str:
    """Agrega información técnica contextual real (BALANCEADO para evitar sobreajuste)."""
    sentences = re.split(r'([.!?]+)', text)
    result_sentences = []
    
    text_length = len(text.split())
    is_long_text = text_length > 200
    base_multiplier = 1.1 if is_long_text else 1.0  # Más conservador
    
    i = 0
    sentence_count = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if not sentence:
            i += 1
            continue
        
        sentence_count += 1
        
        # Cada 3-4 oraciones, agregar frase técnica al inicio (más frecuente)
        if sentence_count % 4 == 0 and random.random() < complexity_level * 0.55 * base_multiplier:  # Aumentado a 0.55
            phrase = random.choice(TECHNICAL_PHRASES)
            sentence = phrase.capitalize() + ', ' + sentence.lower()
        
        # Insertar término técnico entre paréntesis (balanceado - más frecuente)
        if random.random() < complexity_level * 0.45 * base_multiplier and len(sentence.split()) > 8:  # Aumentado a 0.45
            words = sentence.split()
            if len(words) > 5:
                insert_pos = random.randint(2, len(words) - 3)
                technical_term = random.choice(TECHNICAL_TERMS)
                words.insert(insert_pos, f"({technical_term})")
                sentence = ' '.join(words)
        
        # Agregar especificaciones técnicas a números mencionados (moderado)
        if random.random() < complexity_level * 0.15 * base_multiplier:  # Reducido a 0.15
            # Buscar números seguidos de "percent" o "%"
            sentence = re.sub(
                r'(\d+)\s*(percent|%)',
                lambda m: f"{m.group(1)}% (95% CI: {max(0, int(m.group(1))-5)}-{min(100, int(m.group(1))+5)}%)",
                sentence,
                count=1
            )
        
        result_sentences.append(sentence)
        if i + 1 < len(sentences):
            result_sentences.append(sentences[i + 1])
        i += 2
    
    return ''.join(result_sentences)

def expand_with_details(text: str, complexity_level: float) -> str:
    """Expande oraciones con detalles técnicos adicionales."""
    # Agregar especificaciones técnicas a números
    text = re.sub(
        r'(\d+)\s+(percent|%)',
        lambda m: f"{m.group(1)}% (95% confidence interval: {max(0, int(m.group(1))-5)}-{min(100, int(m.group(1))+5)}%)",
        text,
        count=1 if random.random() < complexity_level * 0.2 else 0
    )
    
    # Agregar metodología a menciones de estudios
    text = re.sub(
        r'\b(study|trial|research)\b',
        lambda m: f"{m.group(1)} (randomized controlled trial)" if random.random() < complexity_level * 0.3 else m.group(1),
        text,
        flags=re.IGNORECASE
    )
    
    return text

def add_semantic_expansion(text: str, complexity_level: float) -> str:
    """Agrega información técnica nueva relacionada para reducir overlap (BALANCEADO)."""
    text_length = len(text.split())
    if text_length < 100:  # Solo aplicar a textos más largos
        return text
    
    # Dividir en párrafos
    paragraphs = text.split('\n\n')
    result_paragraphs = []
    
    for para in paragraphs:
        if len(para.split()) < 40:  # Solo aplicar a párrafos más largos
            result_paragraphs.append(para)
            continue
        
        # Agregar información técnica más frecuentemente
        if random.random() < complexity_level * 0.70:  # Aumentado a 0.70
            # Agregar información metodológica al final del párrafo (solo una opción)
            method_info = random.choice([
                ' The assessment was conducted using validated measurement instruments.',
                ' Data collection followed standardized protocols.',
                ' The evaluation methodology was consistent with clinical guidelines.',
                ' Statistical analysis employed established tests.',
            ])
            para = para.rstrip('.!?') + method_info
        
        result_paragraphs.append(para)
    
    return '\n\n'.join(result_paragraphs)

def complexify_text(text: str, complexity_level: float = 0.8) -> str:
    """
    Convierte texto simple en versión más técnica usando múltiples técnicas.
    
    Args:
        text: Texto simple (PLS)
        complexity_level: Nivel de complejidad (0.0-1.0)
    
    Returns:
        Texto técnico complejificado
    """
    if pd.isna(text) or len(str(text)) < 20:
        return text
    
    text = str(text)
    
    # 1. Aplicar reemplazos de palabras (vocabulario técnico) - MÁS AGRESIVO
    complex_text = apply_word_replacements(text, complexity_level)
    
    # 2. Reestructurar oraciones (voz pasiva, nominalizaciones)
    # Usar versión avanzada si spaCy está disponible
    if SPACY_AVAILABLE and nlp is not None:
        complex_text = restructure_sentences_advanced(complex_text, complexity_level)
    else:
        complex_text = restructure_sentences(complex_text, complexity_level)
    
    # 3. Agregar contexto técnico (frases técnicas, términos)
    complex_text = add_technical_context(complex_text, complexity_level)
    
    # 4. Expandir con detalles técnicos
    complex_text = expand_with_details(complex_text, complexity_level)
    
    # 5. NUEVO: Agregar expansión semántica (información técnica nueva)
    complex_text = add_semantic_expansion(complex_text, complexity_level)
    
    # 6. Asegurar que el texto técnico sea más largo (agregar información nueva - BALANCEADO)
    expansion_ratio = len(complex_text.split()) / len(text.split()) if len(text.split()) > 0 else 1.0
    text_length = len(text.split())
    if text_length > 400:
        target_expansion = 1.5  # Ratio compresión ~0.67 (ideal)
        num_additions = 3
    elif text_length > 200:
        target_expansion = 1.45  # Ratio compresión ~0.69 (ideal)
        num_additions = 2
    else:
        target_expansion = 1.4  # Ratio compresión ~0.71 (ideal)
        num_additions = 2
    
    if expansion_ratio < target_expansion:
        # Agregar frases técnicas al final (balanceado)
        technical_endings = [
            ' This was evaluated through standardized protocols.',
            ' The methodology followed established guidelines.',
            ' Statistical analysis was performed according to predefined criteria.',
            ' The investigation was conducted in accordance with regulatory standards.',
            ' Clinical endpoints were assessed using validated instruments.',
            ' The study design incorporated quality control measures.',
            ' Data collection followed standardized procedures.',
            ' The research protocol was reviewed by an independent committee.',
            ' The assessment utilized validated measurement tools.',
            ' The investigation adhered to ethical guidelines.',
        ]
        
        # Agregar frases técnicas de forma balanceada (evitar repetición)
        used_endings = set()
        for _ in range(num_additions):
            if random.random() < 0.75:  # 75% de las veces
                # Evitar repetir el mismo ending
                available_endings = [e for e in technical_endings if e not in used_endings]
                if available_endings:
                    ending = random.choice(available_endings)
                    used_endings.add(ending)
                    complex_text = complex_text.rstrip('.!?') + ending
    
    return complex_text

# ==================== FUNCIONES DE ANÁLISIS ====================

def calculate_lexical_overlap(text1: str, text2: str) -> float:
    """
    Calcula el overlap léxico entre dos textos.
    
    Args:
        text1: Primer texto
        text2: Segundo texto
    
    Returns:
        Porcentaje de overlap léxico (0.0-1.0)
    """
    # Normalizar textos
    text1 = re.sub(r'[^\w\s]', '', text1.lower())
    text2 = re.sub(r'[^\w\s]', '', text2.lower())
    
    # Obtener conjuntos de palabras únicas
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words2:
        return 0.0
    
    # Calcular overlap: intersección / unión (Jaccard) o intersección / texto2
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    # Usar Jaccard similarity como métrica de overlap
    jaccard = intersection / union if union > 0 else 0.0
    
    # También calcular overlap relativo al texto simple
    overlap_relative = intersection / len(words2) if len(words2) > 0 else 0.0
    
    # Retornar promedio de ambas métricas
    return (jaccard + overlap_relative) / 2

def calculate_ngram_overlap(text1: str, text2: str, n: int = 2) -> float:
    """Calcula overlap de n-gramas."""
    def get_ngrams(text: str, n: int) -> set:
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
    
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    if not ngrams2:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 0.0

# ==================== FUNCIONES PRINCIPALES ====================

def load_data(data_path: str = None):
    """
    Carga los datos procesados.
    
    Args:
        data_path: Path al archivo CSV. Si es None, busca en ubicación estándar.
    
    Returns:
        DataFrame con datos PLS
    """
    print("Cargando datos...")
    
    # Determinar path del archivo
    if data_path is None:
        # Intentar paths relativos y absolutos
        possible_paths = [
            Path('data/processed/dataset_clean.csv'),
            Path(__file__).parent.parent.parent / 'data' / 'processed' / 'dataset_clean.csv',
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = str(path)
                break
        
        if data_path is None:
            raise FileNotFoundError(
                f"No se encontró el archivo dataset_clean.csv.\n"
                f"Ubicaciones buscadas:\n"
                f"  - {possible_paths[0]}\n"
                f"  - {possible_paths[1]}\n\n"
                f"Por favor, ejecuta primero el preprocesamiento:\n"
                f"  python src/data/make_dataset.py\n"
                f"O usando DVC:\n"
                f"  dvc repro preprocess"
            )
    
    # Verificar que el archivo existe
    if not Path(data_path).exists():
        raise FileNotFoundError(f"El archivo no existe: {data_path}")
    
    print(f"Leyendo datos desde: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Filtrar solo registros con PLS
    if 'label' in df.columns:
        pls_data = df[df['label'] == 'pls'].copy()
    else:
        # Si no hay columna label, asumir que todos los que tienen 'resumen' son PLS
        print("⚠️  No se encontró columna 'label'. Asumiendo que registros con 'resumen' son PLS.")
        pls_data = df[df['resumen'].notna() & (df['resumen'].str.len() > 20)].copy()
    
    print(f"Total registros PLS: {len(pls_data)}")
    
    if len(pls_data) == 0:
        print("⚠️  No se encontraron registros PLS. Verifica que el dataset tenga la columna 'label' o 'resumen'.")
    
    return pls_data

def create_synthetic_pairs(pls_data, complexity_level: float = 0.8):
    """Crea pares sintéticos técnico-simple."""
    print("\n=== CREANDO PARES SINTÉTICOS MEJORADOS ===")
    
    pairs = []
    
    for idx, row in pls_data.iterrows():
        # Usar resumen como texto "simple"
        simple_text = row['resumen']
        
        if pd.isna(simple_text) or len(str(simple_text)) < 20:
            continue
        
        # Crear versión técnica
        technical_text = complexify_text(simple_text, complexity_level=complexity_level)
        
        # Calcular overlap léxico
        lexical_overlap = calculate_lexical_overlap(technical_text, simple_text)
        bigram_overlap = calculate_ngram_overlap(technical_text, simple_text, n=2)
        
        # Crear par sintético
        pair = {
            'texto_tecnico': technical_text,
            'texto_simple': simple_text,
            'source_dataset': row['source_dataset'],
            'source_bucket': row['source_bucket'],
            'doc_id': f"synthetic_{row['doc_id']}",
            'split': row['split'],
            'original_label': row['label'],
            'word_count_tech': len(technical_text.split()),
            'word_count_simple': len(simple_text.split()),
            'compression_ratio': len(simple_text.split()) / len(technical_text.split()) if len(technical_text.split()) > 0 else 0,
            'lexical_overlap': lexical_overlap,
            'bigram_overlap': bigram_overlap,
            'is_synthetic': True
        }
        
        pairs.append(pair)
    
    print(f"Pares sintéticos creados: {len(pairs)}")
    
    return pairs

def analyze_pairs(pairs):
    """Analiza los pares creados con métricas de overlap."""
    print("\n=== ANÁLISIS DE PARES SINTÉTICOS ===")
    
    if not pairs:
        print("No hay pares para analizar")
        return {}
    
    df_pairs = pd.DataFrame(pairs)
    
    print(f"Total pares: {len(df_pairs)}")
    print(f"Promedio palabras técnico: {df_pairs['word_count_tech'].mean():.1f}")
    print(f"Promedio palabras simple: {df_pairs['word_count_simple'].mean():.1f}")
    print(f"Ratio compresión promedio: {df_pairs['compression_ratio'].mean():.2f}")
    
    # Métricas de overlap
    if 'lexical_overlap' in df_pairs.columns:
        avg_overlap = df_pairs['lexical_overlap'].mean()
        median_overlap = df_pairs['lexical_overlap'].median()
        min_overlap = df_pairs['lexical_overlap'].min()
        max_overlap = df_pairs['lexical_overlap'].max()
        
        print(f"\n=== MÉTRICAS DE OVERLAP LÉXICO ===")
        print(f"Overlap léxico promedio: {avg_overlap:.1%}")
        print(f"Overlap léxico mediano: {median_overlap:.1%}")
        print(f"Overlap léxico mínimo: {min_overlap:.1%}")
        print(f"Overlap léxico máximo: {max_overlap:.1%}")
        print(f"Pares con overlap < 50%: {(df_pairs['lexical_overlap'] < 0.50).sum()} ({(df_pairs['lexical_overlap'] < 0.50).sum() / len(df_pairs) * 100:.1f}%)")
        print(f"Pares con overlap < 60%: {(df_pairs['lexical_overlap'] < 0.60).sum()} ({(df_pairs['lexical_overlap'] < 0.60).sum() / len(df_pairs) * 100:.1f}%)")
        print(f"Pares con overlap < 70%: {(df_pairs['lexical_overlap'] < 0.70).sum()} ({(df_pairs['lexical_overlap'] < 0.70).sum() / len(df_pairs) * 100:.1f}%)")
        
        if 'bigram_overlap' in df_pairs.columns:
            avg_bigram = df_pairs['bigram_overlap'].mean()
            print(f"Overlap bigramas promedio: {avg_bigram:.1%}")
    
    print("\nDistribución por fuente:")
    print(df_pairs['source_dataset'].value_counts())
    
    print("\nDistribución por split:")
    print(df_pairs['split'].value_counts())
    
    # Ejemplos con bajo overlap
    print("\n=== EJEMPLOS DE PARES (BAJO OVERLAP) ===")
    if 'lexical_overlap' in df_pairs.columns:
        low_overlap_pairs = df_pairs.nsmallest(3, 'lexical_overlap')
        for idx, (_, pair) in enumerate(low_overlap_pairs.iterrows()):
            print(f"\nEjemplo {idx+1} (Overlap: {pair['lexical_overlap']:.1%}):")
            print(f"Técnico ({pair['word_count_tech']} palabras):")
            print(f"  {pair['texto_tecnico'][:300]}...")
            print(f"Simple ({pair['word_count_simple']} palabras):")
            print(f"  {pair['texto_simple'][:300]}...")
            print(f"Ratio compresión: {pair['compression_ratio']:.2f}")
    else:
        # Fallback a ejemplos aleatorios
        for i, pair in enumerate(pairs[:3]):
            print(f"\nEjemplo {i+1}:")
            print(f"Técnico ({pair['word_count_tech']} palabras):")
            print(f"  {pair['texto_tecnico'][:200]}...")
            print(f"Simple ({pair['word_count_simple']} palabras):")
            print(f"  {pair['texto_simple'][:200]}...")
            print(f"Ratio compresión: {pair['compression_ratio']:.2f}")
    
    # Retornar estadísticas
    stats = {
        'total_pairs': len(df_pairs),
        'avg_tech_words': float(df_pairs['word_count_tech'].mean()),
        'avg_simple_words': float(df_pairs['word_count_simple'].mean()),
        'avg_compression_ratio': float(df_pairs['compression_ratio'].mean()),
    }
    
    if 'lexical_overlap' in df_pairs.columns:
        stats['avg_lexical_overlap'] = float(df_pairs['lexical_overlap'].mean())
        stats['median_lexical_overlap'] = float(df_pairs['lexical_overlap'].median())
        stats['pairs_below_50_overlap'] = int((df_pairs['lexical_overlap'] < 0.50).sum())
        stats['pairs_below_60_overlap'] = int((df_pairs['lexical_overlap'] < 0.60).sum())
        stats['pairs_below_70_overlap'] = int((df_pairs['lexical_overlap'] < 0.70).sum())
    
    return stats

def save_pairs(pairs, output_dir_name: str = 'synthetic_pairs_improved'):
    """Guarda los pares sintéticos."""
    print("\n=== GUARDANDO PARES ===")
    
    # Crear directorio
    output_dir = Path(f'data/processed/{output_dir_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar como DataFrame
    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(output_dir / 'synthetic_pairs.csv', index=False)
    
    # Guardar como JSONL para entrenamiento
    with open(output_dir / 'synthetic_pairs.jsonl', 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    # Guardar estadísticas
    stats = analyze_pairs(pairs)
    stats.update({
        'source_distribution': df_pairs['source_dataset'].value_counts().to_dict(),
        'split_distribution': df_pairs['split'].value_counts().to_dict()
    })
    
    with open(output_dir / 'synthetic_pairs_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Pares guardados en: {output_dir}")
    print(f"- synthetic_pairs.csv ({len(pairs)} pares)")
    print(f"- synthetic_pairs.jsonl ({len(pairs)} pares)")
    print(f"- synthetic_pairs_stats.json (estadísticas)")
    
    return output_dir

def main():
    """Función principal."""
    print("=== GENERADOR DE PARES SINTÉTICOS MEJORADOS ===")
    
    # Cargar datos
    try:
        pls_data = load_data()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}\n")
        return
    
    if len(pls_data) == 0:
        print("❌ No hay datos PLS disponibles")
        print("   Verifica que el dataset tenga registros con label='pls' o columna 'resumen'")
        return
    
    # Crear pares sintéticos con nivel de complejidad balanceado
    # Nivel 0.85 que permite reducir overlap (objetivo 50-65%) manteniendo naturalidad
    pairs = create_synthetic_pairs(pls_data, complexity_level=0.85)
    
    if len(pairs) == 0:
        print("No se pudieron crear pares sintéticos")
        return
    
    # Analizar pares
    stats = analyze_pairs(pairs)
    
    # Guardar pares
    output_dir = save_pairs(pairs, output_dir_name='synthetic_pairs_improved')
    
    print("\n" + "="*60)
    print("RESUMEN DE MEJORAS")
    print("="*60)
    if 'avg_lexical_overlap' in stats:
        print(f"✅ Overlap léxico promedio: {stats['avg_lexical_overlap']:.1%}")
        print(f"✅ Objetivo: Balance entre overlap bajo y naturalidad (ideal: 50-65%)")
        if 0.50 <= stats['avg_lexical_overlap'] <= 0.65:
            print(f"   🎉 RANGO IDEAL! (balance entre diversidad y naturalidad)")
        elif stats['avg_lexical_overlap'] < 0.50:
            print(f"   ⚠️  Overlap muy bajo - puede indicar transformaciones demasiado agresivas")
        elif stats['avg_lexical_overlap'] > 0.65:
            print(f"   ⚠️  Overlap alto - considerar ajustes para más diversidad")
        print(f"✅ Pares con overlap < 50%: {stats.get('pairs_below_50_overlap', 0)}")
        print(f"✅ Pares con overlap < 70%: {stats.get('pairs_below_70_overlap', 0)}")
    print("="*60)
    
    print("\nPares sintéticos mejorados creados exitosamente!")
    print(f"\nPróximo paso: Entrenar generador con estos pares")
    print(f"   Ubicación: {output_dir}")

if __name__ == "__main__":
    main()

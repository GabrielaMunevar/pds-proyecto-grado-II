"""
Dashboard PLS Generator - Versión AWS Ready
============================================

Dashboard moderno para generación de Plain Language Summaries
Desplegado en AWS EC2 con Streamlit Cloud o EC2.
"""

import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
import time
from datetime import datetime
import sys

# Importar utilidades y configuración
sys.path.append(str(Path(__file__).parent.parent))
from utils.text_chunking import split_into_chunks
from config import apply_prompt, get_prompt

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="PLS Generator | Biomedical Text Simplification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS PERSONALIZADO
# ============================================================================

st.markdown("""
<style>
    /* Colores principales */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea statues
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton107>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .result-box {
        background: #ffffff;
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    /* Tabs personalizados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHE DEL MODELO
# ============================================================================

@st.cache_resource(show_spinner="Cargando modelo T5...")
def load_model():
    """Cargar modelo T5 con cache"""
    model_path = Path('models/t5_generator/model')
    tokenizer_path = Path('models/t5_generator/tokenizer')
    
    if not model_path.exists():
        st.error("Modelo no encontrado")
        return None, None
    
    try:
        with st.spinner("Cargando modelo T5-BASE (esto toma ~10 segundos)"):
            model = T5ForConditionalGeneration.from_pretrained(str(model_path))
            tokenizer = T5Tokenizer.from_pretrained(str(tokenizer_path))
        return model, tokenizer
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

# ============================================================================
# HEADER
# ============================================================================

col_header1, col_header2 = st.columns([3, 1])

with col_header1:
    st.markdown("""
    <div class="main-header">
        <h1>Plain Language Summary Generator</h1>
        <p style="margin:0; opacity:0.9;">Biomedical Text Simplification using T5 Transformer</p>
    </div>
    """, unsafe_allow_html=True)

with col_header2:
    st.metric("ROUGE-L Score", "0.361", "Baseline")
    st.caption("Model Evaluation")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://via.placeholder.com/150x150/667eea/ffffff?text=T5", use_container_width=True)
    
    st.markdown("### About")
    st.info("""
    **Model**: T5-BASE (220M parameters)
    
    **Dataset**: 21,527 synthetic pairs
    
    **Training**: 5 epochs, A100 GPU
    
    **Status**: Production Ready
    """)
    
    st.markdown("---")
    
    st.markdown("### Settings")
    
    max_length = st.slider(
        "Maximum Length",
        min_value=50,
        max_value=300,
        value=256,
        step=10
    )
    
    num_beams = st.select_slider(
        "Beam Search Width",
        options=[1, 2, 4, 6],
        value=4
    )
    
    show_examples = st.checkbox("Show Example Texts", value=True)
    
    st.markdown("---")
    
    st.markdown("### Model Info")
    st.success("Model loaded from cache")
    st.caption(f"Loaded at: {datetime.now().strftime('%H:%M:%S')}")
    
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Cargar modelo
model, tokenizer = load_model()

if model is None:
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["Generate", "Examples", "About"])

# ============================================================================
# TAB 1: GENERATE
# ============================================================================

with tab1:
    col_input, col_output = st.columns(2)
    
    with col_input:
        st.markdown("#### Input Text (Biomedical)")
        
        input_text = st.text_area(
            "Enter technical text here...",
            height=350,
            placeholder="Example: This randomized controlled trial evaluated the efficacy of metformin..."
        )
        
        # Ejemplo pre-cargado
        if show_examples:
            with st.expander("Load Example"):
                examples = {
                    "Diabetes Study": """
                    This randomized controlled trial evaluated the efficacy of metformin 
                    in patients with type 2 diabetes mellitus. We conducted a double-blind, 
                    placebo-controlled study with 2,000 participants over 24 weeks. 
                    Primary endpoint was HbA1c reduction. Secondary endpoints included 
                    fasting glucose levels and adverse events.
                    """,
                    "Cancer Treatment": """
                    We assessed the efficacy of pembrolizumab in treating advanced 
                    non-small cell lung cancer. The study enrolled 450 patients 
                    randomized to receive pembrolizumab or chemotherapy. Overall survival 
                    served as the primary endpoint. Progression-free survival and 
                    objective response rate were secondary endpoints.
                    """,
                    "Cardiology": """
                    This multi-center clinical trial investigated the impact of 
                    beta-blockers on cardiovascular outcomes in post-myocardial 
                    infarction patients. Over 3,000 participants were followed for 
                    36 months. Primary composite endpoint included cardiovascular death, 
                    myocardial infarction, or stroke.
                    """
                }
                
                selected_example = st.selectbox("Choose:", list(examples.keys()))
                if st.button("Load Selected"):
                    st.session_state.input_text = examples[selected_example]
                    st.rerun()
    
    with col_output:
        st.markdown("#### Plain Language Summary")
        
        if st.button("Generate PLS", type="primary", use_container_width=True):
            if not input_text or len(input_text.strip()) < 10:
                st.warning("Please enter text to simplify")
            else:
                with st.spinner("Generating with T5..."):
                    start_time = time.time()
                    
                    # Verificar si el texto es largo y necesita chunking
                    tech_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
                    use_chunking = tech_tokens > 400
                    
                    if use_chunking:
                        st.info(f"📄 Long text detected ({tech_tokens} tokens). Using chunking strategy...")
                        
                        # Dividir en chunks
                        chunks = split_into_chunks(
                            input_text,
                            tokenizer=tokenizer.encode,
                            max_tokens=400,
                            overlap=50
                        )
                        
                        st.info(f"Divided into {len(chunks)} chunks")
                        
                        # Generar PLS para cada chunk
                        generated_chunks = []
                        progress_bar = st.progress(0)
                        
                        for i, chunk in enumerate(chunks):
                            # Usar prompt estándar centralizado
                            input_with_prefix = apply_prompt(chunk)
                            
                            inputs = tokenizer(
                                input_with_prefix,
                                return_tensors='pt',
                                max_length=512,
                                truncation=True,
                                padding=True
                            )
                            
                            with torch.no_grad():
                                outputs = model.generate(
                                    inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    max_length=max_length,
                                    num_beams=num_beams,
                                    early_stopping=True,
                                    length_penalty=1.1,
                                    do_sample=False
                                )
                            
                            chunk_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            generated_chunks.append(chunk_result)
                            progress_bar.progress((i + 1) / len(chunks))
                        
                        # Combinar chunks generados
                        combined = ' '.join(generated_chunks)
                        
                        # Limpiar repeticiones
                        words = combined.split()
                        cleaned_words = []
                        prev_word = None
                        for word in words:
                            if word != prev_word:
                                cleaned_words.append(word)
                            prev_word = word
                        
                        result = ' '.join(cleaned_words)
                    else:
                        # Texto corto: procesar directamente
                        # Usar prompt estándar centralizado
                        input_with_prefix = apply_prompt(input_text)
                        
                        inputs = tokenizer(
                            input_with_prefix,
                            return_tensors='pt',
                            max_length=512,
                            truncation=True,
                            padding=True
                        )
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                max_length=max_length,
                                num_beams=num_beams,
                                early_stopping=True,
                                length_penalty=1.1,
                                do_sample=False
                            )
                        
                        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    elapsed = time.time() - start_time
                    
                    # Mostrar resultado
                    st.success(f"Generated in {elapsed:.2f}s")
                    
                    st.text_area(
                        "Result",
                        result,
                        height=350,
                        key="result_output"
                    )
                    
                    # Métricas
                    st.markdown("### Metrics")
                    
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric(
                            "Input Words",
                            len(input_text.split()),
                            delta=None
                        )
                    
                    with metric_col2:
                        output_words = len(result.split()) if result else 0
                        st.metric(
                            "Output Words",
                            output_words
                        )
                    
                    with metric_col3:
                        compression = output_words / len(input_text.split()) if len(input_text.split()) > 0 else 0
                        st.metric(
                            "Compression",
                            f"{compression:.2f}"
                        )
                    
                    with metric_col4:
                        st.metric(
                            "Time",
                            f"{elapsed:.2f}s"
                        )

# ============================================================================
# TAB 2: EXAMPLES
# ============================================================================

with tab2:
    st.markdown("#### Example Medical Texts")
    
    examples_data = [
        {
            "title": "Diabetes Study",
            "technical": "This randomized controlled trial evaluated the efficacy of metformin in patients with type 2 diabetes mellitus. Primary endpoint was HbA1c reduction.",
            "simple": "We tested if metformin helps people with diabetes. We measured blood sugar levels."
        },
        {
            "title": "Cancer Treatment",
            "technical": "Assessed pembrolizumab efficacy in advanced non-small cell lung cancer. Primary endpoint was overall survival.",
            "simple": "We checked if a cancer drug helps people with lung cancer. We looked at how long people lived."
        }
    ]
    
    for i, ex in enumerate(examples_data):
        with st.expander(f"{i+1}. {ex['title']}"):
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                st.markdown("**Technical Text**")
                st.info(ex['technical'])
            with col_ex2:
                st.markdown("**Simple Summary**")
                st.success(ex['simple'])

# ============================================================================
# TAB 3: ABOUT
# ============================================================================

with tab3:
    st.markdown("#### Project Information")
    
    st.markdown("""
    ### Objective
    Generate Plain Language Summaries (PLS) from technical biomedical texts to improve 
    patient comprehension.
    
    ### Model
    - **Architecture**: T5-BASE (Encoder-Decoder Transformer)
    - **Parameters**: 220M
    - **Training**: 5 epochs on A100 GPU
    - **Evaluation**: ROUGE-L 0.361
    
    ### Dataset
    - **Source**: Cochrane, Pfizer, ClinicalTrials.gov
    - **Size**: 21,527 synthetic pairs
    - **Coverage**: Various medical domains
    
    ### Limitations
    Current model performance is limited by dataset quality:
    - Synthetic pairs have 94.5% lexical overlap
    - Insufficient variety in simplifications
    - Training yields marginal improvement over baseline
    
    ### Deployment
    - **Platform**: AWS EC2
    - **Framework**: Streamlit
    - **Status**: Production Ready
    
    ### Authors
    Academic research project for biomedical text simplification.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #667eea; padding: 1rem;">
    <p><strong>PLS Generator</strong> | T5-BASE Transformer | AWS Infrastructure</p>
    <p style="font-size: 0.9rem;">ROUGE-L: 0.361 | Evaluation Complete</p>
</div>
""", unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title  = 'CropLogic AI',
    page_icon   = '🌾',
    layout      = 'wide',
    initial_sidebar_state = 'expanded'
)

# ── Dark theme CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1923; color: #c8d8e8; }
    .stSidebar { background-color: #1a2535; }
    .metric-card {
        background-color: #1a2535;
        border: 1px solid #2a3f52;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #00d4aa; }
    .metric-label { font-size: 0.85rem; color: #8aa0b0; margin-top: 5px; }
    .result-box {
        background-color: #1a2535;
        border-left: 4px solid #00d4aa;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #2a1a1a;
        border-left: 4px solid #f87171;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    h1, h2, h3 { color: #00d4aa; }
    .stButton > button {
        background-color: #00d4aa;
        color: #0f1923;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    .stButton > button:hover { background-color: #00b894; }
    label { color: #c8d8e8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load models and data ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('agriAI_models.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_soil():
    return pd.read_csv('soil_assessment.csv')

@st.cache_resource
def load_rag():
    with open('rag_knowledge_base.json') as f:
        kb = json.load(f)['documents']
    with open('rag_vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts  = [d['text'] for d in kb]
    matrix = vec.transform(texts)
    return kb, vec, matrix

@st.cache_resource
def load_pest_model():
    try:
        from tensorflow import keras
        return keras.models.load_model('pest_model.keras')
    except Exception:
        return None

# ── Sidebar navigation ─────────────────────────────────────────────────
st.sidebar.markdown('# 🌾 CropLogic AI')
st.sidebar.markdown('*Agricultural Decision Support*')
st.sidebar.markdown('---')

page = st.sidebar.radio(
    'Navigate',
    ['🏠 Home', '📈 Yield Prediction', '🌱 Soil Assessment', '🔬 Pest Detection', '💬 RAG Chatbot']
)

st.sidebar.markdown('---')
st.sidebar.markdown('**CropLogic AI**')
st.sidebar.markdown('Youssef Mohamed · YH202000009')
st.sidebar.markdown('The Knowledge Hub Universities')

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════
if page == '🏠 Home':
    st.title('🌾 CropLogic AI')
    st.markdown('### Integrated Agricultural Decision Support System')
    st.markdown('---')

    st.markdown("""
    CropLogic AI is an end-to-end machine learning platform for US maize production,
    combining yield prediction, soil health assessment, pest detection, and an
    AI-powered advisory chatbot into one unified system.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">975</div>
            <div class="metric-label">US Counties</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">64</div>
            <div class="metric-label">Years (1961–2024)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">94.5%</div>
            <div class="metric-label">Pest Detection Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">5</div>
            <div class="metric-label">ML Models</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('### System Modules')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="result-box">
            <h4>📈 Yield Prediction</h4>
            <p>Enter climate and soil values for any county to get a predicted maize yield.
            Uses a weighted ensemble of Ridge, Random Forest, Extra Trees, and Gradient Boosting models
            trained on 62,400 county-year records.</p>
        </div>
        <div class="result-box">
            <h4>🌱 Soil Assessment</h4>
            <p>Explore soil health scores, constraint flags, and tier rankings across
            975 US counties and 15 states. Identify which soil properties limit yield most.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="result-box">
            <h4>🔬 Pest Detection</h4>
            <p>Upload a maize leaf image to classify it as Blight, Common Rust,
            Gray Leaf Spot, or Healthy using EfficientNetB0 transfer learning
            with 94.50% validation accuracy.</p>
        </div>
        <div class="result-box">
            <h4>💬 RAG Chatbot</h4>
            <p>Ask any question about maize yield, soil health, or pest identification.
            The chatbot retrieves answers from CropLogic AI's own knowledge base —
            grounded in your project's real data and findings.</p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — YIELD PREDICTION
# ══════════════════════════════════════════════════════════════════════
elif page == '📈 Yield Prediction':
    st.title('📈 Yield Prediction')
    st.markdown('Enter county climate and soil values to predict maize yield.')
    st.markdown('---')

    try:
        bundle = load_models()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('#### Climate Inputs')
            tas_C   = st.slider('Temperature (°C)',        0.0,  25.0, 10.0, 0.1)
            pr_mm   = st.slider('Precipitation (mm)',    200.0,1400.0,800.0,10.0)
            year    = st.slider('Year',                  1961,  2024, 2020,    1)

            st.markdown('#### Soil Inputs')
            ph      = st.slider('Soil pH',                 4.0,   9.0,  6.5, 0.1)
            soc     = st.slider('Organic Carbon (%)',       0.1,   6.0,  2.5, 0.1)
            clay    = st.slider('Clay (%)',                 5.0,  45.0, 23.0, 0.5)
            sand    = st.slider('Sand (%)',                15.0,  65.0, 35.0, 0.5)

        with col2:
            st.markdown('#### Location Inputs')
            lat     = st.slider('Latitude',               36.0,  49.0, 42.0, 0.1)
            lon     = st.slider('Longitude',             -104.0,-80.0,-93.0, 0.1)

            st.markdown('#### About the Model')
            st.markdown("""
            <div class="result-box">
            The ensemble combines 4 models weighted by R² score:
            Ridge Regression, Random Forest, Extra Trees, and Gradient Boosting.
            Trained on data from 1961–2009, evaluated on 2010–2024.
            </div>
            """, unsafe_allow_html=True)

        if st.button('Predict Yield'):
            # Build feature vector matching ALL_FEATURES from CropAI-01
            silt              = max(0, 100 - clay - sand)
            years_since_1961  = year - 1961
            t                 = years_since_1961

            # County averages (using input values as proxies)
            tas_sq            = tas_C ** 2
            pr_log            = np.log1p(pr_mm)
            aridity_index     = tas_C / (pr_mm + 1)
            tas_anomaly_county= 0.0
            pr_anomaly_county = 0.0
            whc_proxy         = clay * 0.4 + silt * 0.2
            ph_dev_opt        = abs(ph - 6.5)
            lat_norm          = (lat - 36.0) / (49.0 - 36.0)
            dist_corn_belt    = np.sqrt((lat - 42)**2 + (lon - (-93))**2)
            trend_linear      = t
            trend_sq          = t ** 2
            trend_sqrt        = np.sqrt(t)
            tas_x_pr          = tas_C * pr_mm
            heat_drought      = 0.0
            yield_lag1        = 7000.0
            state_code        = 4
            decade            = min(6, (years_since_1961 // 10) + 1)

            features = np.array([[
                tas_C, pr_mm,
                tas_sq, pr_log, aridity_index,
                tas_anomaly_county, pr_anomaly_county,
                ph, soc, clay, sand, silt,
                whc_proxy, ph_dev_opt,
                lat_norm, dist_corn_belt,
                years_since_1961, trend_linear, trend_sq, trend_sqrt,
                tas_x_pr, heat_drought,
                yield_lag1,
                state_code, decade
            ]])

            scaler  = bundle.get('scaler_temporal') or bundle.get('scaler_random')
            feat_sc = scaler.transform(features)

            ridge = bundle['ridge']
            rf    = bundle['rf']
            et    = bundle['et']
            gb    = bundle['gb']
            w     = bundle.get('ensemble_weights', np.array([0.1, 0.3, 0.3, 0.3]))
            w     = np.clip(w, 0, None)
            w     = w / w.sum()

            p_ridge = ridge.predict(feat_sc)[0]
            p_rf    = rf.predict(features)[0]
            p_et    = et.predict(features)[0]
            p_gb    = gb.predict(features)[0]
            p_ens   = w[0]*p_ridge + w[1]*p_rf + w[2]*p_et + w[3]*p_gb

            st.markdown('---')
            st.markdown('### Prediction Results')

            c1, c2, c3, c4, c5 = st.columns(5)
            for col, name, val in zip(
                [c1, c2, c3, c4, c5],
                ['Ridge', 'Random Forest', 'Extra Trees', 'Grad. Boost', 'Ensemble'],
                [p_ridge, p_rf, p_et, p_gb, p_ens]
            ):
                color = '#00d4aa' if name == 'Ensemble' else '#4fc3f7'
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{val:.0f}</div>
                    <div class="metric-label">{name}<br>kg/ha</div>
                </div>""", unsafe_allow_html=True)

            # Interpretation
            if p_ens > 10000:
                msg = '🟢 Excellent yield conditions'
            elif p_ens > 7000:
                msg = '🟡 Good yield conditions'
            elif p_ens > 4000:
                msg = '🟠 Moderate yield conditions'
            else:
                msg = '🔴 Poor yield conditions — check soil and climate inputs'

            st.markdown(f"""
            <div class="result-box">
                <strong>Ensemble Prediction: {p_ens:.0f} kg/ha</strong><br>
                {msg}<br>
                <small>National average: ~6,835 kg/ha (1961–2024)</small>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f'Could not load models: {e}')
        st.info('Make sure agriAI_models.pkl is in the same folder as this app.')

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — SOIL ASSESSMENT
# ══════════════════════════════════════════════════════════════════════
elif page == '🌱 Soil Assessment':
    st.title('🌱 Soil Assessment')
    st.markdown('County-level soil health analysis across 975 US maize counties.')
    st.markdown('---')

    try:
        soil_df = load_soil()

        # Summary metrics
        tier_counts = soil_df['soil_tier'].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#00d4aa">{tier_counts.get('Good',0)}</div>
                <div class="metric-label">Good Tier Counties</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#ffd166">{tier_counts.get('Moderate',0)}</div>
                <div class="metric-label">Moderate Tier Counties</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#f87171">{tier_counts.get('Poor',0)}</div>
                <div class="metric-label">Poor Tier Counties</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{soil_df['soil_health_score'].mean():.1f}</div>
                <div class="metric-label">Avg Health Score /100</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('---')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('#### State Rankings by Soil Health Score')
            state_rank = (
                soil_df.groupby('state')['soil_health_score']
                .mean()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={'soil_health_score': 'Avg Score'})
            )
            state_rank['Avg Score'] = state_rank['Avg Score'].round(1)
            state_rank['Tier'] = state_rank['Avg Score'].apply(
                lambda x: '🟢 Good' if x >= 70 else ('🟡 Moderate' if x >= 50 else '🔴 Poor')
            )
            st.dataframe(state_rank, use_container_width=True, height=420)

        with col2:
            st.markdown('#### Soil Constraint Prevalence')
            flag_cols   = ['flag_low_ph', 'flag_high_ph', 'flag_low_soc', 'flag_high_sand', 'flag_high_clay']
            flag_labels = ['Low pH (<5.8)', 'High pH (>7.2)', 'Low SOC (<1.5%)', 'High Sand (>55%)', 'High Clay (>40%)']
            constraint_data = []
            for col, label in zip(flag_cols, flag_labels):
                if col in soil_df.columns:
                    n   = int(soil_df[col].sum())
                    pct = n / len(soil_df) * 100
                    constraint_data.append({'Constraint': label, 'Counties': n, 'Percentage': f'{pct:.1f}%'})
            if constraint_data:
                st.dataframe(pd.DataFrame(constraint_data), use_container_width=True)

            st.markdown('#### Yield by Soil Health Tier')
            if 'mean_yield' in soil_df.columns:
                tier_yield = (
                    soil_df.groupby('soil_tier')['mean_yield']
                    .mean()
                    .reindex(['Good', 'Moderate', 'Poor'])
                    .reset_index()
                    .rename(columns={'mean_yield': 'Avg Yield (kg/ha)'})
                )
                tier_yield['Avg Yield (kg/ha)'] = tier_yield['Avg Yield (kg/ha)'].round(0)
                st.dataframe(tier_yield, use_container_width=True)

        st.markdown('---')
        st.markdown('#### County Explorer')
        search = st.text_input('Search for a county or state', placeholder='e.g. Iowa or IOW_001')
        if search:
            mask    = (
                soil_df['county_id'].str.contains(search, case=False, na=False) |
                soil_df['state'].str.contains(search, case=False, na=False)
            )
            results = soil_df[mask][['county_id', 'state', 'soil_health_score', 'soil_tier',
                                      'ph', 'soc', 'sand', 'clay']].round(2)
            if len(results) > 0:
                st.dataframe(results, use_container_width=True)
            else:
                st.info('No counties found. Try a state name like "Iowa" or "Illinois".')

    except Exception as e:
        st.error(f'Could not load soil data: {e}')
        st.info('Make sure soil_assessment.csv is in the same folder as this app.')

# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — PEST DETECTION
# ══════════════════════════════════════════════════════════════════════
elif page == '🔬 Pest Detection':
    st.title('🔬 Pest Detection')
    st.markdown('Upload a maize leaf image to classify the disease condition.')
    st.markdown('---')

    CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    CLASS_INFO  = {
        'Blight':        ('🔴', 'Northern Corn Leaf Blight — large cigar-shaped grey-green lesions caused by Exserohilum turcicum.'),
        'Common_Rust':   ('🟠', 'Common Rust — small oval brick-red pustules on both leaf surfaces caused by Puccinia sorghi.'),
        'Gray_Leaf_Spot':('🟡', 'Gray Leaf Spot — rectangular grey-brown lesions between leaf veins caused by Cercospora zeae-maydis.'),
        'Healthy':       ('🟢', 'Healthy — no visible disease symptoms. Leaf is deep green and uniform.'),
    }

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('#### Upload Leaf Image')
        uploaded = st.file_uploader(
            'Choose a maize leaf image',
            type=['jpg', 'jpeg', 'png'],
            help='Upload a clear photo of a single maize leaf'
        )

        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, caption='Uploaded Image', use_container_width=True)

    with col2:
        st.markdown('#### Disease Reference')
        for cls, (icon, desc) in CLASS_INFO.items():
            st.markdown(f"""
            <div class="result-box">
                <strong>{icon} {cls.replace('_', ' ')}</strong><br>
                <small>{desc}</small>
            </div>""", unsafe_allow_html=True)

    if uploaded:
        model = load_pest_model()
        if model is None:
            st.error('Pest detection model not found. Make sure pest_model.keras is in the same folder.')
        else:
            if st.button('Classify Disease'):
                with st.spinner('Analysing leaf...'):
                    import tensorflow as tf
                    img_resized = img.resize((224, 224))
                    img_array   = np.array(img_resized) / 255.0
                    img_array   = np.expand_dims(img_array, 0)
                    preds       = model.predict(img_array, verbose=0)[0]
                    pred_idx    = np.argmax(preds)
                    pred_class  = CLASS_NAMES[pred_idx]
                    confidence  = preds[pred_idx] * 100

                st.markdown('---')
                st.markdown('### Classification Result')

                icon, desc = CLASS_INFO[pred_class]
                box_color  = '#1a3a1a' if pred_class == 'Healthy' else '#3a1a1a'
                border_col = '#00d4aa' if pred_class == 'Healthy' else '#f87171'

                st.markdown(f"""
                <div style="background:{box_color};border-left:4px solid {border_col};
                            border-radius:5px;padding:20px;margin:10px 0;">
                    <h3>{icon} {pred_class.replace('_', ' ')}</h3>
                    <h4>Confidence: {confidence:.1f}%</h4>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('#### All Class Probabilities')
                prob_df = pd.DataFrame({
                    'Disease':     [c.replace('_', ' ') for c in CLASS_NAMES],
                    'Probability': [f'{p*100:.1f}%' for p in preds]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

    st.markdown('---')
    st.markdown("""
    <div class="result-box">
        <strong>Model Details</strong><br>
        Architecture: EfficientNetB0 (transfer learning from ImageNet)<br>
        Training: Two-phase — head only then full fine-tune<br>
        Classes: Blight · Common Rust · Gray Leaf Spot · Healthy<br>
        Validation accuracy: 94.50%
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — RAG CHATBOT
# ══════════════════════════════════════════════════════════════════════
elif page == '💬 RAG Chatbot':
    st.title('💬 RAG Advisory Chatbot')
    st.markdown('Ask questions about maize yield, soil health, or pest identification.')
    st.markdown('---')

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    from collections import defaultdict

    try:
        kb, vectorizer, tfidf_matrix = load_rag()

        def retrieve(query, top_k=5):
            qv   = vectorizer.transform([query])
            sims = cos_sim(qv, tfidf_matrix)[0]
            idx  = np.argsort(sims)[::-1][:top_k]
            return [{'text': kb[i]['text'], 'source': kb[i]['source'],
                     'similarity': float(sims[i])} for i in idx]

        def format_answer(question, docs):
            relevant = [d for d in docs if d['similarity'] >= 0.05]
            if not relevant:
                return None, []
            grouped = defaultdict(list)
            for d in relevant:
                grouped[d['source']].append(d)
            return grouped, relevant

        # Example questions
        st.markdown('#### Try these questions:')
        example_cols = st.columns(3)
        examples = [
            'What does Common Rust look like?',
            'Which state has the best soil health?',
            'How accurate is the yield model?',
            'What temperature is best for maize?',
            'How does drought affect yield?',
            'What is Gray Leaf Spot?',
        ]
        for i, ex in enumerate(examples):
            with example_cols[i % 3]:
                if st.button(ex, key=f'ex_{i}'):
                    st.session_state['chat_input'] = ex

        st.markdown('---')

        # Chat input
        question = st.text_input(
            'Your question:',
            value=st.session_state.get('chat_input', ''),
            placeholder='e.g. What is the ideal soil pH for maize?',
            key='main_input'
        )

        if st.button('Ask') or question:
            if question.strip():
                docs     = retrieve(question)
                grouped, relevant = format_answer(question, docs)

                if grouped is None:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>No relevant answer found.</strong><br>
                        Try using keywords like: soil, pH, yield, drought, rust, blight, temperature, Corn Belt
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f'### Answer for: *"{question}"*')
                    for source, source_docs in grouped.items():
                        source_colors = {
                            'Yield Model':         '#4fc3f7',
                            'Soil Assessment':     '#00d4aa',
                            'Pest Detection':      '#ff6b35',
                            'Agronomic Knowledge': '#a78bfa',
                        }
                        color = source_colors.get(source, '#ffd166')
                        st.markdown(f'<p style="color:{color};font-weight:bold;">[ {source} ]</p>',
                                    unsafe_allow_html=True)
                        for doc in source_docs:
                            st.markdown(f"""
                            <div class="result-box">
                                {doc['text']}
                                <br><small style="color:#8aa0b0;">Relevance: {doc['similarity']:.3f}</small>
                            </div>
                            """, unsafe_allow_html=True)

                    sources_used = list(grouped.keys())
                    best_score   = relevant[0]['similarity']
                    st.markdown(f"""
                    <p style="color:#8aa0b0;font-size:0.85rem;">
                    Sources: {' · '.join(sources_used)} &nbsp;|&nbsp;
                    Best match: {best_score:.3f} &nbsp;|&nbsp;
                    Documents shown: {len(relevant)}
                    </p>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f'Could not load RAG system: {e}')
        st.info('Make sure rag_knowledge_base.json and rag_vectorizer.pkl are in the same folder.')

import json
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

import streamlit as st
import streamlit.components.v1 as components

from model_utils import predict
from gradcam import make_gradcam_heatmap

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTION FOR IMAGE TO BASE64
# ═══════════════════════════════════════════════════════════════

def image_to_base64(image):
    """Convert PIL Image to base64 string for HTML embedding"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# ═══════════════════════════════════════════════════════════════
# SPECIES DATA (ALL 12 SPECIES)
# ═══════════════════════════════════════════════════════════════

SPECIES = {
    "common_leopard": {"name": "Common Leopard", "icon": "🐆", "color": "#f4a942", "class": "Mammal"},
    "grey_langur": {"name": "Grey Langur", "icon": "🐒", "color": "#a0bfa8", "class": "Mammal"},
    "jackal": {"name": "Jackal", "icon": "🐺", "color": "#c8a97e", "class": "Mammal"},
    "jungle_cat": {"name": "Jungle Cat", "icon": "🐱", "color": "#d4b483", "class": "Mammal"},
    "kalij_pheasant": {"name": "Kalij Pheasant", "icon": "🦚", "color": "#4fc3f7", "class": "Bird"},
    "leopard_cat": {"name": "Leopard Cat", "icon": "🐱", "color": "#e8b86d", "class": "Mammal"},
    "mongoose": {"name": "Mongoose", "icon": "🦡", "color": "#b5956a", "class": "Mammal"},
    "porcupine": {"name": "Porcupine", "icon": "🦔", "color": "#9e8b75", "class": "Mammal"},
    "red_fox": {"name": "Red Fox", "icon": "🦊", "color": "#e05c2a", "class": "Mammal"},
    "rhesus_macaque": {"name": "Rhesus Macaque", "icon": "🐒", "color": "#c4956a", "class": "Mammal"},
    "snow_leopard": {"name": "Snow Leopard", "icon": "🐆", "color": "#90caf9", "class": "Mammal"},
    "wild_boar": {"name": "Wild Boar", "icon": "🐗", "color": "#8d7b6a", "class": "Mammal"},
}

def get_species(key):
    key = key.lower().replace(" ", "_")
    return SPECIES.get(key, {"name": key.replace("_", " ").title(), "icon": "🐾", "color": "#5a7a65", "class": "Unknown"})

# ═══════════════════════════════════════════════════════════════
# PAGE SETUP
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="WildLens AI", page_icon="🦁", layout="wide")

# Load history from URL/localStorage
LS_KEY = "wildlens_history"

def migrate_history(history):
    migrated = []
    for entry in history:
        if 'class' not in entry:
            species_info = get_species(entry.get('species', entry.get('name', '')))
            entry['class'] = species_info['class']
            if 'name' not in entry:
                entry['name'] = species_info['name']
            if 'icon' not in entry:
                entry['icon'] = species_info['icon']
            if 'color' not in entry:
                entry['color'] = species_info['color']
        migrated.append(entry)
    return migrated

if "history" not in st.session_state:
    raw = st.query_params.get("data", None)
    if raw:
        try:
            loaded_history = json.loads(raw)
            st.session_state.history = migrate_history(loaded_history)
        except:
            st.session_state.history = []
    else:
        st.session_state.history = []

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

def save_history():
    serializable_history = []
    for entry in st.session_state.history:
        serializable_entry = {
            'species': entry.get('species', ''),
            'name': entry.get('name', ''),
            'icon': entry.get('icon', '🐾'),
            'color': entry.get('color', '#5a7a65'),
            'class': entry.get('class', 'Unknown'),
            'confidence': entry.get('confidence', 0.0),
            'time': entry.get('time', datetime.now().strftime("%H:%M:%S")),
            'filename': entry.get('filename', 'unknown')
        }
        serializable_history.append(serializable_entry)
    st.query_params["data"] = json.dumps(serializable_history)

# ═══════════════════════════════════════════════════════════════
# CSS STYLING
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .stApp { background: #0a0c0a; }
    [data-testid="stSidebar"] { background: #0e1110; border-right: 1px solid #1a2a1f; }
    
    .stButton > button { 
        background: transparent; 
        border: 1px solid #2a3a2a; 
        color: #8aa88a; 
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton > button:hover { 
        border-color: #4caf50; 
        color: #4caf50;
        background: rgba(76, 175, 80, 0.05);
    }
    
    [data-testid="stFileUploader"] {
        border: 2px dashed #2a3a2a !important;
        border-radius: 12px !important;
        background: #0f1210 !important;
    }
    
    .card { 
        background: #0f1210; 
        border-radius: 12px; 
        border: 1px solid #1e2a1e; 
        padding: 1rem; 
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        border-color: #4caf50;
    }
    
    .hero-card {
        background: linear-gradient(135deg, #0f1210 0%, #121a12 100%);
        border-radius: 16px;
        border: 1px solid #2a3a2a;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: #0f1210;
        border-radius: 12px;
        border: 1px solid #1e2a1e;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #4caf50;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.1);
    }
    
    .pred-card { 
        background: linear-gradient(135deg, #0f1210 0%, #121a12 100%);
        border-radius: 12px; 
        border: 1px solid #2a3a2a; 
        padding: 1.2rem; 
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .pred-card:hover {
        transform: translateY(-2px);
        border-color: #4caf50;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.1);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #0f1210 0%, #0a0f0a 100%);
        border: 1px solid #2a3a2a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        border-color: #4caf50;
        transform: translateY(-2px);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4caf50, #8bc34a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #8aa88a;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 6px;
    }
    
    .history-item { 
        display: flex; 
        align-items: center; 
        gap: 12px; 
        padding: 12px; 
        border-bottom: 1px solid #1e2a1e;
        transition: background 0.2s;
        border-radius: 8px;
    }
    .history-item:hover { 
        background: #121a12; 
    }
    
    hr {
        border-color: #1e2a1e;
        margin: 1.5rem 0;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 2px;
    }
    
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e0eae0;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4caf50;
        display: inline-block;
    }
    
    .home-greeting {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4caf50, #8bc34a, #4fc3f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .export-btn {
        margin-bottom: 1rem;
        text-align: right;
    }
    
    /* Image container for consistent sizing */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2a3a2a;
        background: #0f1210;
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        border-color: #4caf50;
        transform: translateY(-2px);
    }
    
    .image-container img {
        width: 100%;
        height: auto;
        display: block;
    }
    
    .image-caption {
        text-align: center;
        padding: 0.5rem;
        font-size: 0.75rem;
        color: #8aa88a;
        background: #0a0f0a;
        border-top: 1px solid #2a3a2a;
    }
    
    /* Chart container for consistent sizing */
    .chart-container {
        background: #0f1210;
        border-radius: 12px;
        border: 1px solid #2a3a2a;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        border-color: #4caf50;
        transform: translateY(-2px);
    }
    
    .chart-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #e0eae0;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2a3a2a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Special Download Button Styling */
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        animation: gradientShift 3s ease infinite !important;
        background-size: 200% 200% !important;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    div[data-testid="stDownloadButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    div[data-testid="stDownloadButton"] button:active {
        transform: translateY(0);
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model
    return load_model("model.h5")

@st.cache_data
def load_class_names():
    with open("class_names.txt") as f:
        return [l.strip() for l in f.readlines()]

class_names = load_class_names()

# Set seaborn style for better looking plots
sns.set_style("darkgrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# ═══════════════════════════════════════════════════════════════
# CSV EXPORT FUNCTION
# ═══════════════════════════════════════════════════════════════

def export_to_csv():
    if not st.session_state.history:
        return None
    
    df = pd.DataFrame(st.session_state.history)
    required_cols = ['filename', 'name', 'class', 'confidence', 'time']
    for col in required_cols:
        if col not in df.columns:
            df[col] = '—' if col != 'confidence' else 0.0
    
    df = df[required_cols]
    df.columns = ['Filename', 'Species', 'Class', 'Confidence (%)', 'Timestamp']
    df['Confidence (%)'] = (df['Confidence (%)'] * 100).round(2)
    
    return df.to_csv(index=False).encode('utf-8')

# ═══════════════════════════════════════════════════════════════
# ANALYTICS FUNCTIONS (CONSISTENT SIZING)
# ═══════════════════════════════════════════════════════════════

def show_analytics():
    hist = st.session_state.history
    if not hist:
        st.info("✨ No data available. Make some predictions first!")
        return
    
    df = pd.DataFrame(hist)
    
    # Export button row
    st.markdown('<div class="export-btn">', unsafe_allow_html=True)
    csv_data = export_to_csv()
    if csv_data:
        st.download_button(
            label="📥 Export All Data as CSV",
            data=csv_data,
            file_name=f"wildlens_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=False,
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Stats Cards Row - 4 equal cards
    total = len(df)
    unique_species = df['name'].nunique()
    avg_conf = df['confidence'].mean() * 100
    mammal_count = len(df[df['class'] == 'Mammal']) if 'class' in df.columns else 0
    bird_count = len(df[df['class'] == 'Bird']) if 'class' in df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{total}</div>
            <div class="stat-label">Total Predictions</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{unique_species}</div>
            <div class="stat-label">Unique Species</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{avg_conf:.1f}%</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{mammal_count} / {bird_count}</div>
            <div class="stat-label">Mammals / Birds</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 1: Species Distribution (full width, consistent size)
    st.markdown(f"""
    <div class="chart-container">
        <div class="chart-title">
            <span>📊</span> Species Distribution
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    species_counts = df['name'].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(species_counts)))
    bars = ax.barh(range(len(species_counts)), species_counts.values, color=colors_bar, edgecolor='#2a3a2a', linewidth=1)
    ax.set_yticks(range(len(species_counts)))
    ax.set_yticklabels(species_counts.index, fontsize=10)
    ax.set_xlabel('Count', color='#8aa88a', fontsize=10)
    ax.set_facecolor('#0f1210')
    fig.patch.set_facecolor('#0f1210')
    ax.tick_params(colors='#8aa88a')
    ax.spines['bottom'].set_color('#2a3a2a')
    ax.spines['top'].set_color('#2a3a2a')
    ax.spines['left'].set_color('#2a3a2a')
    ax.spines['right'].set_color('#2a3a2a')
    
    for i, (bar, val) in enumerate(zip(bars, species_counts.values)):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, str(val), 
               va='center', fontsize=9, color='#8aa88a')
    
    st.pyplot(fig, use_container_width=True)
    
    # Download button below chart
    species_df = pd.DataFrame({'Species': species_counts.index, 'Count': species_counts.values})
    species_csv = species_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Species Data",
        data=species_csv,
        file_name="species_distribution.csv",
        mime="text/csv",
        use_container_width=False
    )
    
    st.markdown("---")
    
    # Row 2: Two equal columns for Class Distribution and Confidence Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="chart-container">
            <div class="chart-title">
                <span>🦁</span> Class Distribution
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if 'class' in df.columns and df['class'].nunique() > 0:
            class_counts = df['class'].value_counts()
            
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            colors_donut = ['#4caf50', '#4fc3f7']
            wedges, texts, autotexts = ax2.pie(class_counts.values, labels=class_counts.index, 
                                                autopct='%1.1f%%', colors=colors_donut,
                                                textprops={'color': '#e0eae0', 'fontsize': 11, 'fontweight': 'bold'},
                                                wedgeprops={'edgecolor': '#0f1210', 'linewidth': 2})
            
            centre_circle = plt.Circle((0, 0), 0.70, fc='#0f1210', linewidth=2, edgecolor='#2a3a2a')
            fig2.gca().add_artist(centre_circle)
            
            ax2.set_title('Mammals vs Birds', color='#e0eae0', fontsize=12, pad=15)
            fig2.patch.set_facecolor('#0f1210')
            st.pyplot(fig2, use_container_width=True)
            
            class_df = pd.DataFrame({'Class': class_counts.index, 'Count': class_counts.values})
            class_csv = class_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Class Data",
                data=class_csv,
                file_name="class_distribution.csv",
                mime="text/csv",
                use_container_width=False
            )
    
    with col2:
        st.markdown(f"""
        <div class="chart-container">
            <div class="chart-title">
                <span>📈</span> Confidence Score Distribution
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        
        counts, bins, patches = ax3.hist(df['confidence'], bins=20, edgecolor='#1e2a1e', linewidth=1, alpha=0.7)
        
        for i, (count, bin_edge) in enumerate(zip(counts, bins[:-1])):
            if bin_edge >= 0.8:
                patches[i].set_facecolor('#10b981')
            elif bin_edge >= 0.6:
                patches[i].set_facecolor('#f59e0b')
            else:
                patches[i].set_facecolor('#8b5cf6')
        
        ax3.set_xlabel('Confidence Score', color='#8aa88a', fontsize=10)
        ax3.set_ylabel('Frequency', color='#8aa88a', fontsize=10)
        ax3.set_facecolor('#0f1210')
        fig3.patch.set_facecolor('#0f1210')
        ax3.tick_params(colors='#8aa88a')
        ax3.spines['bottom'].set_color('#2a3a2a')
        ax3.spines['top'].set_color('#2a3a2a')
        ax3.spines['left'].set_color('#2a3a2a')
        ax3.spines['right'].set_color('#2a3a2a')
        
        mean_conf = df['confidence'].mean()
        ax3.axvline(mean_conf, color='#ff9800', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.2f}')
        ax3.legend(facecolor='#0f1210', edgecolor='#2a3a2a', labelcolor='#e0eae0', loc='upper right')
        ax3.grid(True, alpha=0.2, color='#8aa88a')
        
        st.pyplot(fig3, use_container_width=True)
        
        hist_data = pd.DataFrame({'Bin_Start': bins[:-1], 'Bin_End': bins[1:], 'Count': counts})
        hist_csv = hist_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Confidence Data",
            data=hist_csv,
            file_name="confidence_distribution.csv",
            mime="text/csv",
            use_container_width=False
        )
    
    st.markdown("---")
    
    # Row 3: Two equal columns for Confidence Levels and Top Performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="chart-container">
            <div class="chart-title">
                <span>🎯</span> Confidence Level Breakdown
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        high_conf = len(df[df['confidence'] >= 0.8])
        med_conf = len(df[(df['confidence'] >= 0.6) & (df['confidence'] < 0.8)])
        low_conf = len(df[df['confidence'] < 0.6])
        
        conf_categories = ['High (≥80%)', 'Medium (60-80%)', 'Low (<60%)']
        conf_counts = [high_conf, med_conf, low_conf]
        conf_colors = ['#10b981', '#f59e0b', '#8b5cf6']
        
        if sum(conf_counts) > 0:
            fig4, ax4 = plt.subplots(figsize=(6, 6))
            wedges, texts, autotexts = ax4.pie(conf_counts, labels=conf_categories, autopct='%1.1f%%',
                                                colors=conf_colors, textprops={'color': '#e0eae0', 'fontsize': 10, 'fontweight': 'bold'},
                                                wedgeprops={'edgecolor': '#0f1210', 'linewidth': 2},
                                                startangle=90)
            
            centre_circle = plt.Circle((0, 0), 0.60, fc='#0f1210', linewidth=2, edgecolor='#2a3a2a')
            fig4.gca().add_artist(centre_circle)
            
            ax4.set_title('Confidence Level Distribution', color='#e0eae0', fontsize=12, pad=20)
            fig4.patch.set_facecolor('#0f1210')
            st.pyplot(fig4, use_container_width=True)
            
            conf_levels_df = pd.DataFrame({'Level': conf_categories, 'Count': conf_counts, 
                                          'Percentage': [f"{(c/sum(conf_counts)*100):.1f}%" for c in conf_counts]})
            conf_levels_csv = conf_levels_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Confidence Levels",
                data=conf_levels_csv,
                file_name="confidence_levels.csv",
                mime="text/csv",
                use_container_width=False
            )
        else:
            st.info("No predictions available")
    
    with col2:
        st.markdown(f"""
        <div class="chart-container">
            <div class="chart-title">
                <span>🏆</span> Top Performers (High Confidence)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        high_conf_predictions = df[df['confidence'] > 0.85].sort_values('confidence', ascending=False).head(5)
        
        if len(high_conf_predictions) > 0:
            top_performers_df = high_conf_predictions[['name', 'confidence', 'filename', 'time']].copy()
            top_performers_df.columns = ['Species', 'Confidence', 'Filename', 'Timestamp']
            top_performers_df['Confidence'] = (top_performers_df['Confidence'] * 100).round(2)
            top_csv = top_performers_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Top Performers",
                data=top_csv,
                file_name="top_performers.csv",
                mime="text/csv",
                use_container_width=False
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            for idx, row in high_conf_predictions.iterrows():
                icon = row.get('icon', '🐾')
                name = row.get('name', 'Unknown')
                filename = row.get('filename', 'unknown')[:35]
                confidence = row['confidence']
                conf_percent = confidence * 100
                
                if conf_percent >= 90:
                    bar_color = "#10b981"
                    badge_text = "Excellent"
                elif conf_percent >= 85:
                    bar_color = "#34d399"
                    badge_text = "Very High"
                else:
                    bar_color = "#f59e0b"
                    badge_text = "High"
                
                st.markdown(f"""
                <div class="card" style="margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 12px; flex: 1;">
                            <div style="font-size: 1.8rem;">{icon}</div>
                            <div style="flex: 1;">
                                <div style="font-weight: 600; color: #e0eae0; font-size: 0.85rem;">{name}</div>
                                <div style="font-size: 0.65rem; color: #6a8a6a;">{filename}</div>
                                <div style="margin-top: 6px; height: 4px; background: #1e2a1e; border-radius: 2px; width: 100%;">
                                    <div style="width: {conf_percent}%; height: 100%; background: {bar_color}; border-radius: 2px;"></div>
                                </div>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.2rem; font-weight: 700; color: {bar_color};">{conf_percent:.1f}%</div>
                            <div class="badge" style="background: {bar_color}20; color: {bar_color}; font-size: 0.6rem;">{badge_text}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No high confidence predictions (>85%) yet")

# ═══════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════

def show_home():
    st.markdown("""
    <div class="hero-card">
        <div class="home-greeting">Welcome to WildLens AI</div>
        <div style="font-size: 1.1rem; color: #8aa88a; margin-bottom: 1rem;">
            Advanced Wildlife Recognition System Powered by Deep Learning
        </div>
        <div style="font-size: 0.9rem; color: #6a8a6a;">
            Upload wildlife photos and let AI identify the species with high accuracy
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-title">🚀 Key Features</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem;">📸</div>
            <div style="font-weight: 600; margin: 0.5rem 0;">Multi-Image Upload</div>
            <div style="font-size: 0.75rem; color: #6a8a6a;">Upload single or multiple images for batch processing</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem;">🤖</div>
            <div style="font-weight: 600; margin: 0.5rem 0;">AI Classification</div>
            <div style="font-size: 0.75rem; color: #6a8a6a;">12 wildlife species with 95%+ accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem;">🔥</div>
            <div style="font-weight: 600; margin: 0.5rem 0;">Grad-CAM Heatmaps</div>
            <div style="font-size: 0.75rem; color: #6a8a6a;">Visualize what AI focuses on for predictions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem;">📊</div>
            <div style="font-weight: 600; margin: 0.5rem 0;">Analytics Dashboard</div>
            <div style="font-size: 0.75rem; color: #6a8a6a;">Track predictions, confidence scores and trends</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<p class="section-title">🦁 All 12 Supported Species</p>', unsafe_allow_html=True)
    
    species_list = list(SPECIES.items())
    
    for i in range(0, len(species_list), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(species_list):
                key, sp = species_list[i + j]
                with cols[j]:
                    st.markdown(f"""
                    <div class="card" style="text-align: center; padding: 0.8rem;">
                        <div style="font-size: 2rem;">{sp['icon']}</div>
                        <div style="font-weight: 600; font-size: 0.85rem; margin-top: 6px; color: {sp['color']};">{sp['name']}</div>
                        <div style="margin-top: 4px;">
                            <span class="badge" style="background: #4caf5020; color: #4caf50; font-size: 0.65rem;">{sp['class']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.history:
        st.markdown('<p class="section-title">📈 Session Statistics</p>', unsafe_allow_html=True)
        
        hist = st.session_state.history
        total = len(hist)
        unique = len(set(h.get('name', '') for h in hist))
        avg_conf = sum(h.get('confidence', 0) for h in hist) / total
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{total}</div>
                <div class="stat-label">Total Predictions</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{unique}</div>
                <div class="stat-label">Species Detected</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{avg_conf*100:.1f}%</div>
                <div class="stat-label">Average Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<p class="section-title">🕒 Recent Predictions</p>', unsafe_allow_html=True)
        
        for pred in reversed(hist[-5:]):
            conf = pred.get('confidence', 0)
            conf_color = "#4caf50" if conf >= 0.7 else "#ff9800" if conf >= 0.5 else "#f44336"
            st.markdown(f"""
            <div class="history-item">
                <div style="font-size: 1.5rem;">{pred.get('icon', '🐾')}</div>
                <div style="flex: 1;">
                    <div style="font-weight: 600;">{pred.get('name', 'Unknown')}</div>
                    <div style="font-size: 0.7rem; color: #6a8a6a;">{pred.get('filename', 'unknown')}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1rem; font-weight: 700; color: {conf_color};">{conf*100:.1f}%</div>
                    <div style="font-size: 0.65rem; color: #6a8a6a;">confidence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hero-card" style="margin-top: 1rem;">
            <div style="font-size: 2rem;">🎯 Ready to Start?</div>
            <div style="margin: 1rem 0;">
                Go to the <strong style="color: #4caf50;">Predict</strong> page and upload your first wildlife photo
            </div>
            <div style="font-size: 0.8rem; color: #6a8a6a;">
                Supported formats: JPG, JPEG, PNG, WEBP
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🦁 WildLens")
    st.caption("AI-Powered Wildlife Recognition")
    st.markdown("---")
    
    page = st.radio("", ["🏠 Home", "📸 Predict", "📊 History", "📈 Analytics", "ℹ️ Species"], label_visibility="collapsed")
    
    st.markdown("---")
    hist = st.session_state.history
    if hist:
        total_conf = sum(h.get('confidence', 0) for h in hist)
        avg_conf = total_conf / len(hist) if hist else 0
        st.metric("Total Predictions", len(hist))
        st.metric("Unique Species", len(set(h.get('name', '') for h in hist)))
        st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
        
        csv_data = export_to_csv()
        if csv_data:
            st.download_button(
                label="📥 Export as CSV",
                data=csv_data,
                file_name=f"wildlens_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.batch_results = []
            save_history()
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE ROUTING
# ═══════════════════════════════════════════════════════════════

if page == "🏠 Home":
    show_home()

elif page == "📸 Predict":
    st.markdown("<h1>📸 Identify Wildlife</h1>", unsafe_allow_html=True)
    st.caption("Upload one or multiple photos - AI will identify each species")
    
    uploaded_files = st.file_uploader(
        "Drop images here or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.markdown(f"### 📷 {len(uploaded_files)} image(s) selected")
        
        if st.button("🔍 Analyze All", type="primary", use_container_width=True):
            st.session_state.batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                image = Image.open(file).convert("RGB")
                img_model = image.resize((224, 224))
                img_array = np.expand_dims(np.array(img_model), axis=0)
                
                preds, model = predict(img_array)
                preds_flat = preds[0]
                top_idx = np.argmax(preds_flat)
                confidence = float(preds_flat[top_idx])
                species_key = class_names[top_idx]
                species = get_species(species_key)
                
                heatmap = make_gradcam_heatmap(img_array, model)
                
                st.session_state.batch_results.append({
                    "filename": file.name,
                    "image": image,
                    "species_key": species_key,
                    "species": species,
                    "confidence": confidence,
                    "heatmap": heatmap
                })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Analysis complete!")
            
            for result in st.session_state.batch_results:
                st.session_state.history.append({
                    "species": result['species_key'],
                    "name": result['species']['name'],
                    "icon": result['species']['icon'],
                    "color": result['species']['color'],
                    "class": result['species']['class'],
                    "confidence": result['confidence'],
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "filename": result['filename']
                })
            save_history()
        
        if st.session_state.batch_results:
            st.markdown("---")
            st.markdown("## 📋 Results")
            
            for idx, result in enumerate(st.session_state.batch_results):
                species = result['species']
                conf = result['confidence']
                
                st.markdown(f"### #{idx + 1} - {result['filename']}")
                
                col1, col2 = st.columns(2, gap="medium")
                
                with col1:
                    img_display = result['image'].resize((450, 450))
                    img_base64 = image_to_base64(img_display)
                    st.markdown(f"""
                    <div class="image-container">
                        <img src="data:image/jpeg;base64,{img_base64}" style="width: 100%; height: auto;">
                        <div class="image-caption">📷 Original Image</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    hr = cv2.resize(result['heatmap'], (450, 450))
                    hc = cv2.applyColorMap(np.uint8(255 * hr), cv2.COLORMAP_INFERNO)
                    img_display_np = np.array(result['image'].resize((450, 450)))
                    overlay = cv2.addWeighted(img_display_np, 0.55, hc, 0.45, 0)
                    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    overlay_pil = Image.fromarray(overlay_rgb)
                    overlay_base64 = image_to_base64(overlay_pil)
                    
                    st.markdown(f"""
                    <div class="image-container">
                        <img src="data:image/jpeg;base64,{overlay_base64}" style="width: 100%; height: auto;">
                        <div class="image-caption">🔥 Grad-CAM Heatmap</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                bar_color = "#4caf50" if conf >= 0.7 else "#ff9800" if conf >= 0.5 else "#f44336"
                st.markdown(f"""
                <div class="pred-card" style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div style="font-size: 3rem;">{species['icon']}</div>
                            <div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: {species['color']};">{species['name']}</div>
                                <div style="margin-top: 4px;">
                                    <span class="badge" style="background: #4caf5020; border: 1px solid #4caf50; color: #4caf50;">{species['class']}</span>
                                </div>
                            </div>
                        </div>
                        <div style="flex: 1; max-width: 300px;">
                            <div style="height: 8px; background: #1e2a1e; border-radius: 4px; overflow: hidden;">
                                <div style="width: {conf*100}%; height: 100%; background: {bar_color}; border-radius: 4px;"></div>
                            </div>
                            <div style="margin-top: 8px; text-align: center;">
                                <span style="font-size: 1.3rem; font-weight: 700; color: {bar_color};">{conf*100:.1f}%</span>
                                <span style="color: #8aa88a;"> Confidence</span>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if idx < len(st.session_state.batch_results) - 1:
                    st.markdown("<hr>", unsafe_allow_html=True)

elif page == "📊 History":
    st.markdown("<h1>📊 Prediction History</h1>", unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("✨ No predictions yet. Upload an image to get started!")
    else:
        hist = st.session_state.history
        total = len(hist)
        unique = len(set(h.get('name', 'Unknown') for h in hist))
        avg_conf = sum(h.get('confidence', 0) for h in hist) / total
        high_conf = len([h for h in hist if h.get('confidence', 0) >= 0.8])
        
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{total}</div>
                <div class="stat-label">Total Predictions</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{unique}</div>
                <div class="stat-label">Species Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{avg_conf*100:.0f}%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{high_conf}</div>
                <div class="stat-label">High Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📜 All Predictions")
        
        search = st.text_input("🔍 Search by species or filename", placeholder="Type to filter...")
        
        filtered = hist
        if search:
            filtered = [h for h in hist if search.lower() in h.get('name', '').lower() or search.lower() in h.get('filename', '').lower()]
        
        for pred in reversed(filtered[-50:]):
            conf = pred.get('confidence', 0)
            conf_color = "#4caf50" if conf >= 0.7 else "#ff9800" if conf >= 0.5 else "#f44336"
            st.markdown(f"""
            <div class="history-item">
                <div style="font-size: 1.8rem;">{pred.get('icon', '🐾')}</div>
                <div style="flex: 1;">
                    <div style="font-weight: 600; color: #e0eae0;">{pred.get('name', 'Unknown')}</div>
                    <div style="font-size: 0.7rem; color: #6a8a6a;">{pred.get('filename', 'unknown')} • {pred.get('time', '—')}</div>
                    <div style="font-size: 0.65rem; margin-top: 4px;">
                        <span class="badge" style="background: #4caf5020; color: #4caf50;">{pred.get('class', 'Unknown')}</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: {conf_color};">{conf*100:.1f}%</div>
                    <div style="font-size: 0.65rem; color: #6a8a6a;">confidence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

elif page == "📈 Analytics":
    st.markdown("<h1>📈 Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.caption("📊 Comprehensive insights from your wildlife observations")
    st.markdown("---")
    show_analytics()

elif page == "ℹ️ Species":
    st.markdown("<h1>ℹ️ Recognized Species</h1>", unsafe_allow_html=True)
    st.caption("Our AI can identify these 12 wildlife species")
    
    cols = st.columns(3)
    for idx, (key, sp) in enumerate(SPECIES.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <div style="font-size: 2.5rem;">{sp['icon']}</div>
                <div style="font-weight: 700; margin-top: 8px; color: {sp['color']};">{sp['name']}</div>
                <div style="margin-top: 8px;">
                    <div class="badge" style="background: #4caf5020; color: #4caf50;">{sp['class']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SYNC WITH LOCALSTORAGE
# ═══════════════════════════════════════════════════════════════

components.html(f"""
<script>
(function(){{
    const KEY = "{LS_KEY}";
    const stored = localStorage.getItem(KEY);
    const url = new URL(window.parent.location.href);
    if (stored && !url.searchParams.get("data")) {{
        url.searchParams.set("data", stored);
        window.parent.history.replaceState(null,"",url.toString());
        window.parent.location.reload();
    }}
    const param = url.searchParams.get("data");
    if (param) localStorage.setItem(KEY, param);
}})();
</script>
""", height=0)
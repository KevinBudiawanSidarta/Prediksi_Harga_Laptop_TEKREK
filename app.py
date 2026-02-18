import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from PIL import Image

# ===============================
# CONFIG & CUSTOM CSS
# ===============================
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih menarik
st.markdown("""
<style>
/* ===============================
   GLOBAL
=============================== */
.stApp {
    background-color: #ffffff;
}

body {
    background-color: #e1def5;
    color: #e1def5;
    font-family: 'Inter', sans-serif;
}

/* ===============================
   HEADERS
=============================== */
.main-header {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #2563EB 0%, #0D9488 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}

.sub-header {
    font-size: 1.8rem;
    font-weight: 600;
    background: linear-gradient(90deg, #2563EB 0%, #0D9488 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ===============================
   METRIC CARD
=============================== */
.metric-card {
    background: linear-gradient(135deg, #2563EB 0%, #0D9488 100%);
    padding: 1.5rem;
    border-radius: 16px;
    color: white;
    box-shadow: 0 12px 25px rgba(37, 99, 235, 0.25);
}

/* ===============================
   FEATURE CARD
=============================== */
.feature-card {
    background: linear-gradient(180deg, #1E1B4B 0%, #020617 100%);
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    border-left: 6px solid #2563EB;
    margin-bottom: 1.25rem;
}

/* ===============================
   BUTTON
=============================== */
.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #0D9488 100%);
    color: white;
    border: none;
    padding: 0.85rem 2rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 30px rgba(37, 99, 235, 0.35);
}

/* ===============================
   SIDEBAR
=============================== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E1B4B 0%, #020617 100%);
}

/* ===============================
   SLIDER
=============================== */
/* ===============================
   SLIDER (FIXED)
=============================== */

/* Slider container */
div[data-baseweb="slider"] {
    background: #000000;
    padding: 0.9rem;
    border-radius: 12px;
    border: 2px solid #E2E8F0;
}

/* All slider text */
div[data-baseweb="slider"] * {
    color: #FF3399 !important;
}

/* Slider track */
div[data-baseweb="slider"] > div > div {
    background-color: #FF3399;
}

/* Slider label */
label[data-testid="stWidgetLabel"] {
    color: #FF3399 !important;
}



/* ===============================
   SELECTBOX & RADIO
=============================== */
.stSelectbox, .stRadio {
    background: linear-gradient(180deg, #1E1B4B 0%, #020617 100%);
    padding: 0.5rem;
    border-radius: 12px;
    border: 2px solid #E2E8F0;
    margin-top: 0.8rem;
}

/* ===============================
   SUCCESS BOX
=============================== */
.success-box {
    background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
    padding: 2rem;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 14px 30px rgba(34, 197, 94, 0.35);
}

/* ===============================
   TABS
=============================== */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.5rem;
    background-color: #F1F5F9;
    padding: 0.6rem;
    border-radius: 14px;
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    background: linear-gradient(180deg, #1E1B4B 0%, #020617 100%);
    border-radius: 12px 12px 0 0;
    padding: 10px 22px;
    font-weight: 600;
    color: #ffffff;
}

</style>
""", unsafe_allow_html=True)



# Color palette
COLORS = {
    "primary": "#ffffff",
    "secondary": "#764ba2",
    "accent": "#ed64a6",
    "success": "#48bb78",
    "warning": "#ed8936",
    "info": "#4299e1",
    "light": "#FF1493",
    "dark": "#483D8B"
}

# ===============================
# LOAD DATA & MODEL
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("data_final1_TEKREK.csv")
    return df.drop(columns=["Unnamed: 0"])

@st.cache_resource
def load_model():
    model = joblib.load("prediksi_model_linear_TEKREK.pkl")
    scaler = joblib.load("scaler_linear_TEKREK.pkl")
    return model, scaler

df = load_data()
model, scaler = load_model()

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='color: {COLORS["primary"]}; font-size: 1.5rem;'>💻 Prediksi Harga Laptop</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation dengan styling yang lebih baik
    st.markdown("### 📍 Navigasi")
    page = st.radio(
        "",
        ["🏠 Dashboard", "📊 Analisis", "🔮 Prediksi"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Informasi dataset
    st.markdown("### 📦 Dataset Info")
    st.info(f"""
    **Total Samples:** {len(df):,}
    
    **Features:** {len(df.columns)}
    
    **Price Range:** €{df['Price (Euro)'].min():,.0f} - €{df['Price (Euro)'].max():,.0f}
    """)
    
    st.markdown("---")
    

# ===============================
# DASHBOARD PAGE
# ===============================
if page == "🏠 Dashboard":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style="text-align:center;">
            <span style="font-size:3rem;">💻</span>
            <h1 class="main-header" style="text-align:center;">
                Prediksi Harga Laptop
            </h1>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; opacity: 0.9;">📊 Total Data</div>
            <div style="font-size: 2rem; font-weight: 700;">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ed64a6 0%, #ed64a6 100%);">
            <div style="font-size: 0.9rem; opacity: 0.9;">💶 Avg Price</div>
            <div style="font-size: 2rem; font-weight: 700;">€{df['Price (Euro)'].mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4299e1 0%, #4299e1 100%);">
            <div style="font-size: 0.9rem; opacity: 0.9;">💾 Max RAM</div>
            <div style="font-size: 2rem; font-weight: 700;">{df['RAM (GB)'].max()} GB</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">
            <div style="font-size: 0.9rem; opacity: 0.9;">⚙️ Max CPU</div>
            <div style="font-size: 2rem; font-weight: 700;">{df['CPU_Frequency (GHz)'].max()} GHz</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features Section
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:1.8rem;">🎯</span>
        <h2 class="sub-header">Fitur-Fitur Utama untuk Prediksi</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        ("💾 Kapasitas RAM", "Menentukan kemampuan multitasking", "RAM (GB)"),
        ("⚡ Kecepatan CPU", "Berpengaruh terhadap performa pemrosesan", "CPU_Frequency (GHz)"),
        ("💾 Penyimpanan", "SSD vs HDD", "SSD/HDD"),
        ("🖥️ Kualitas Layar", "Resolusi & IPS Panel", "Resolution"),
        ("⚖️ Berat", "Berat dalam KG", "Weight (kg)"),
        ("👆 Touchscreen", "Kemampuan interaktif laptop", "Touchscreen")
    ]
    
    for i, (title, desc, col_name) in enumerate(features):
        with [col1, col2, col3][i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <h4 style="color: {COLORS['primary']}; margin-top: 0;">{title}</h4>
                <p style="color: #718096; font-size: 0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Preview
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:1.8rem;">📄</span>
        <h2 class="sub-header">Dataset Preview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 10 baris pertama", "📈 Statistik data", "🎯 Distribusi target (Price)"])
    
    with tab1:
        st.dataframe(df.head(10).style.background_gradient(subset=['Price (Euro)'], cmap='Purples'), 
                    use_container_width=True, height=350)
    
    with tab2:
        st.dataframe(df.describe().style.background_gradient(cmap='Blues'), 
                    use_container_width=True, height=350)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            price_chart = alt.Chart(df).mark_bar(color=COLORS['primary']).encode(
                alt.X('Price (Euro):Q', bin=alt.Bin(maxbins=30), title='Price (€)'),
                alt.Y('count()', title='Frequency'),
                tooltip=['count()']
            ).properties(height=300)
            st.altair_chart(price_chart, use_container_width=True)

# ===============================
# ANALYTICS PAGE
# ===============================
elif page == "📊 Analisis":
    st.markdown("""
            <div style="text-align:left;">
            <span style="font-size:3rem;">📊</span>
            <h1 class="main-header" style="display:inline;">
                Analisis Data
            </h1>
            """, unsafe_allow_html=True)
    st.markdown('<p style="color: #718096; font-size: 1.1rem;">Visualisasi interaktif untuk menganalisis hubungan antara spesifikasi laptop dan harga.</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive filters
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
        "<h4 style='color:#000080; margin-bottom:0.1rem;'>💰 Rentang Harga   (€)</h4>",
        unsafe_allow_html=True
        )
        min_price, max_price = st.slider(
            "",
            float(df['Price (Euro)'].min()),
            float(df['Price (Euro)'].max()),
            (float(df['Price (Euro)'].min()), float(df['Price (Euro)'].max())),
            step=100.0
        )
    
    with col2:
        st.markdown(
        "<h4 style='color:#000080; margin-bottom:0.1rem;'>💾 RAM Filter</h4>",
        unsafe_allow_html=True
        )
        ram_filter = st.multiselect(
            "",
            options=sorted(df['RAM (GB)'].unique()),
            default=sorted(df['RAM (GB)'].unique())
        )
    
    with col3:
        st.markdown(
        "<h4 style='color:#000080; margin-bottom:0.1rem;'>🖥️ IPS Panel</h4>",
        unsafe_allow_html=True
        )
        ips_filter = st.multiselect(
            "",
            options=[0, 1],
            default=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
    
    # Filter data
    filtered_df = df[
        (df['Price (Euro)'] >= min_price) & 
        (df['Price (Euro)'] <= max_price) &
        (df['RAM (GB)'].isin(ram_filter)) &
        (df['IPS_Panel'].isin(ips_filter))
    ]
    
    st.markdown(f"*Showing {len(filtered_df)} of {len(df)} records*")
    
    # Visualizations grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<h3 style="color: {COLORS["warning"]};">💾 Price vs RAM</h3>', unsafe_allow_html=True)
        chart_ram = alt.Chart(filtered_df).mark_circle(
            size=70, opacity=0.7
        ).encode(
            x=alt.X('RAM (GB):Q', title='RAM (GB)', scale=alt.Scale(zero=False)),
            y=alt.Y('Price (Euro):Q', title='Price (€)', scale=alt.Scale(zero=False)),
            color=alt.Color('RAM (GB):Q', scale=alt.Scale(scheme='purples'), legend=None),
            tooltip=['RAM (GB)', 'Price (Euro)', 'CPU_Frequency (GHz)', 'SSD']
        ).properties(
            height=350,
            background="#EFF6FF"   # 🎨 warna lebih terang
        ).configure_axis(
            labelColor="#1E293B",     # warna angka di sumbu
            titleColor="#2563EB",     # warna judul sumbu
            labelFontSize=12,
            titleFontSize=14
        ).interactive()
        st.altair_chart(chart_ram, use_container_width=True)
        st.info("""
            📊 **Insight:**
            Secara umum terdapat tren positif antara RAM dan harga laptop.
            Laptop dengan RAM lebih tinggi cenderung memiliki harga lebih mahal.
            Namun, harga tidak hanya dipengaruhi oleh RAM saja,
            melainkan juga spesifikasi lain seperti CPU dan penyimpanan.
            """)
    
    with col2:
        st.markdown(f'<h3 style="color: {COLORS["secondary"]};">⚡ Price vs CPU Frequency</h3>', unsafe_allow_html=True)
        chart_cpu = alt.Chart(filtered_df).mark_circle(size=70, opacity=0.7).encode(
            x=alt.X('CPU_Frequency (GHz):Q', title='CPU Frequency (GHz)', scale=alt.Scale(zero=False)),
            y=alt.Y('Price (Euro):Q', title='Price (€)', scale=alt.Scale(zero=False)),
            color=alt.Color('CPU_Frequency (GHz):Q', scale=alt.Scale(scheme='reds'), legend=None),
            tooltip=['CPU_Frequency (GHz)', 'Price (Euro)', 'RAM (GB)', 'SSD']
        ).properties(
            height=350,
            background="#F5F3FF" 
        ).configure_axis(
            labelColor="#1E293B",     # warna angka di sumbu
            titleColor="#2563EB",     # warna judul sumbu
            labelFontSize=12,
            titleFontSize=14
        ).interactive()
        st.altair_chart(chart_cpu, use_container_width=True)
        st.info("""
            📊 **Insight:**
            Semakin tinggi frekuensi CPU, potensi harga meningkat, namun tidak sepenuhnya linear. Laptop dengan CPU 1.0 GHz cenderung berada
            di rentang harga lebih rendah hingga menengah. Pada 2.0 GHz dan 3.0 GHz terlihat variasi harga yang lebih luas mulai dari menengah hingga tinggi
            .Kesimpulanya, frekuensi CPU memang berkontribusi terhadap harga laptop tetapi tidak bisa dijadikan satu-satunya faktor penentu.
            """)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f'<h3 style="color: {COLORS["accent"]};">🖥️ Pengaruh IPS Panel</h3>', unsafe_allow_html=True)
        chart_ips = alt.Chart(filtered_df).mark_bar(cornerRadius=10).encode(
            x=alt.X('IPS_Panel:N', title='IPS Panel', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('mean(Price (Euro)):Q', title='Average Price (€)'),
            color=alt.Color('IPS_Panel:N', scale=alt.Scale(
            domain=[0, 1],
            range=['#94A3B8', '#10B981']   # grey-blue & emerald
        ), legend=None),
            tooltip=['IPS_Panel', 'mean(Price (Euro))']
        ).properties(
            height=350,
            background="#FCEEFF"
        ).configure_axis(
            labelColor="#1E293B",     # warna angka di sumbu
            titleColor="#2563EB",     # warna judul sumbu
            labelFontSize=12,
            titleFontSize=14
        )
        st.altair_chart(chart_ips, use_container_width=True)
        st.info("""
            📊 **Insight:**
            Pengaruh IPS Panel memang cukup signifikan dikarenakan IPS Panel memiliki warna yang lebih bagus dibanding layar dengan no IPS Panel, tentu 
            hal tersebut juga menjadi faktor mengapa Laptop dengan IPS Panel memiliki harga yang lebih tinggi.
            """)
    
    with col4:
        st.markdown(f'<h3 style="color: {COLORS["info"]};">⚖️ Weight vs Price</h3>', unsafe_allow_html=True)
        chart_weight = alt.Chart(filtered_df).mark_circle(size=70, opacity=0.7).encode(
            x=alt.X('Weight (kg):Q', title='Weight (kg)', scale=alt.Scale(zero=False)),
            y=alt.Y('Price (Euro):Q', title='Price (€)', scale=alt.Scale(zero=False)),
            color=alt.Color('Weight (kg):Q', scale=alt.Scale(scheme='teals'), legend=None),
            tooltip=['Weight (kg)', 'Price (Euro)', 'Inches', 'RAM (GB)']
        ).properties(
            height=350,
            background="#EAF7F0"
        ).configure_axis(
            labelColor="#1E293B",     # warna angka di sumbu
            titleColor="#2563EB",     # warna judul sumbu
            labelFontSize=12,
            titleFontSize=14
        ).interactive()
        st.altair_chart(chart_weight, use_container_width=True)
        st.info("""
            📊 **Insight:**
            Secara umum, tidak terlihat hubungan linear yang kuat antara berat laptop (Weight) dan harga (Price). Laptop dengan berat sekitar 1–3 kg memiliki rentang harga yang cukup luas, mulai dari harga rendah hingga tinggi.
            Namun, laptop dengan berat yang lebih ringan (sekitar 1 kg) cenderung memiliki beberapa harga yang cukup tinggi, yang kemungkinan disebabkan oleh faktor lain seperti spesifikasi premium, material ringan (misalnya ultrabook), atau brand positioning.
            """)
    
    # Correlation heatmap
    st.markdown(f'<h3 style="color: {COLORS["dark"]};">📈 Korelasi antar fitur</h3>', unsafe_allow_html=True)
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    corr_matrix = filtered_df[numeric_cols].corr()
    
    corr_chart = alt.Chart(corr_matrix.reset_index().melt('index')).mark_rect().encode(
        x=alt.X('index:N', title=''),
        y=alt.Y('variable:N', title=''),
        color=alt.Color('value:Q', scale=alt.Scale(scheme='purplered')),
        tooltip=['index', 'variable', 'value']
    ).properties(
        height=400,
        background="#F3F4F6"
    ).configure_axis(
            labelColor="#1E293B",     # warna angka di sumbu
            titleColor="#2563EB",     # warna judul sumbu
            labelFontSize=12,
            titleFontSize=14
    )
    st.altair_chart(corr_chart, use_container_width=True)

# ===============================
# PREDICTION PAGE
# ===============================

elif page == "🔮 Prediksi":
    st.markdown("""
            <div style="text-align:left;">
            <span style="font-size:3rem;">🔮</span>
            <h1 class="main-header" style="display:inline;">
                Prediksi Harga
            </h1>
            """, unsafe_allow_html=True)
    st.markdown('<p style="color: #718096; font-size: 1.1rem;">Konfigurasikan spesifikasi laptop dan dapatkan prediksi harga secara instan</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input sections with cards
    st.markdown('<h2 style="color: #4a5568;">🎛️ Konfigurasikan Spesifikasi Laptop Anda!</h2>', unsafe_allow_html=True)
    
    # ===============================
    # BRAND SECTION
    # ===============================
    st.markdown(f'''
    <div class="feature-card">
        <h4 style="color: {COLORS["primary"]};">🏢 Brand Laptop</h4>
    </div>
    ''', unsafe_allow_html=True)

    company_list = [
        'Apple','Asus','Chuwi','Dell','Fujitsu','Google','HP',
        'Huawei','LG','Lenovo','MSI','Mediacom',
        'Microsoft','Razer','Samsung','Toshiba','Vero','Xiaomi'
    ]

    company = st.selectbox(
        "Pilih Brand Laptop",
        options=company_list,
        index=company_list.index("Dell"),
        help="Brand mempengaruhi positioning harga dan segmentasi pasar"
    )

    st.markdown(
        f"<small style='color: {COLORS['secondary']};'>Selected Brand: {company}</small>",
        unsafe_allow_html=True
    )
    
    # Performance Section
    st.markdown(f'<div class="feature-card"><h4 style="color: {COLORS["primary"]};">⚡ Performa</h4></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
        "<h4 style='color:#000080; margin-bottom:0.1rem;'>Frekuensi CPU</h4>",
        unsafe_allow_html=True
        )
        cpu = st.slider(
            "",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.1,
            help="Frekuensi yang lebih tinggi = Kecepatan Pemrosesan yang lebih tinggi"
        )
        st.markdown(f"<small style='color: {COLORS['secondary']};'>Selected: {cpu} GHz</small>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(
        "<h4 style='color:#000080; margin-bottom:0.1rem;'>Kapasitas RAM</h4>",
        unsafe_allow_html=True
        )
        ram = st.selectbox(
            "",
            options=[2, 4, 6, 8, 12, 14],
            index=2,
            help="Semakin banyak ram, semakin mudah multitasking"
        )
        st.markdown(f"<small style='color: {COLORS['primary']};'>Selected: {ram} GB</small>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(
        "<h4 style='color:#000080; margin-bottom:0.1rem;'>Ukuran Layar (inci)</h4>",
        unsafe_allow_html=True
        )
        inches = st.slider(
            "",
            min_value=10.0,
            max_value=18.0,
            value=15.6,
            step=0.1,
            help="Layar yang besar menyajikan viewing experience yang semakin baik"
        )
        st.markdown(f"<small style='color: {COLORS['primary']};'>Selected: {inches}\"</small>", unsafe_allow_html=True)
    
    # Storage Section
    st.markdown(f'<div class="feature-card"><h4 style="color: {COLORS["primary"]};">💾 Storage</h4></div>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    
    with col4:
        ssd = st.selectbox(
            "SSD Capacity (GB)",
            options=[0, 128, 256, 512, 1024],
            index=2,
            help="SSD provides faster boot and load times"
        )
        st.markdown(f"<small style='color: {COLORS['secondary']};'>Selected: {ssd} GB</small>", unsafe_allow_html=True)
    
    with col5:
        hdd = st.selectbox(
            "HDD Capacity (GB)",
            options=[0, 500, 1000],
            index=1,
            help="HDD offers more storage at lower cost"
        )
        st.markdown(f"<small style='color: {COLORS['secondary']};'>Selected: {hdd} GB</small>", unsafe_allow_html=True)
    
    # Display & Portability
    st.markdown(f'<div class="feature-card"><h4 style="color: {COLORS["primary"]};">🖥️ Layar dan berat</h4></div>', unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)
    
    with col6:
        res_width = st.selectbox(
            "Resolution Width",
            options=[1366, 1920, 2560, 2880],
            index=1,
            help="Higher width means sharper display"
        )
        st.markdown(f"<small style='color: {COLORS['accent']};'>Selected: {res_width}px</small>", unsafe_allow_html=True)
    
    with col7:
        res_height = st.selectbox(
            "Resolution Height",
            options=[768, 1080, 1600, 1800],
            index=1,
            help="Higher height means more vertical space"
        )
        st.markdown(f"<small style='color: {COLORS['accent']};'>Selected: {res_height}px</small>", unsafe_allow_html=True)
    
    with col8:

        weight = st.slider(
            "Berat (Kg)",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
            help="Lower weight means better portability"
        )
        st.markdown(f"<small style='color: {COLORS['accent']};'>Selected: {weight} kg</small>", unsafe_allow_html=True)
    
    # Features
    st.markdown(f'<div class="feature-card"><h4 style="color: {COLORS["info"]};">✨ Fitur Tambahan</h4></div>', unsafe_allow_html=True)
    col9, col10 = st.columns(2)
    
    with col9:
        ips = st.radio(
            "IPS Panel",
            options=[0, 1],
            index=1,
            format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
            help="IPS panels offer better color and viewing angles",
            horizontal=True
        )
    
    with col10:
        touchscreen = st.radio(
            "Touchscreen",
            options=[0, 1],
            index=0,
            format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
            help="Touchscreen enables touch interaction",
            horizontal=True
        )
    
    st.markdown("---")
    
    # Prediction button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀 Prediksi Harga Sekarang!!", use_container_width=True):
            with st.spinner("🤖 Menganalisis Spesifikasi..."):
                company_columns = [
                    'Company_Apple','Company_Asus','Company_Chuwi','Company_Dell',
                    'Company_Fujitsu','Company_Google','Company_HP','Company_Huawei',
                    'Company_LG','Company_Lenovo','Company_MSI','Company_Mediacom',
                    'Company_Microsoft','Company_Razer','Company_Samsung',
                    'Company_Toshiba','Company_Vero','Company_Xiaomi'
                
                ]   
                company_dict = {col: 0 for col in company_columns}
                selected_col = f"Company_{company}"

                if selected_col in company_dict:
                    company_dict[selected_col] = 1
                    
                # Prepare input data
                input_data = pd.DataFrame([[ 
                    inches, cpu, ram, weight, touchscreen,
                    ssd, res_width, res_height, ips, hdd
                ] + list(company_dict.values())],
                columns=[
                    "Inches", "CPU_Frequency (GHz)", "RAM (GB)", "Weight (kg)",
                    "Touchscreen", "SSD", "Res_Width", "Res_Height",
                    "IPS_Panel", "HDD"
                ] + company_columns)
                
                # Make prediction
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                
                EUR_TO_IDR = 17000
                price_idr = prediction * EUR_TO_IDR

                # Display result
                st.markdown("---")
                
                # Result in a nice card
                st.markdown(f"""
                    <div class="success-box">
                        <h2 style="margin-top: 0;">🎯 Prediksi Selesai!</h2>
                        <p style="font-size: 1.2rem; margin-bottom: 1rem;">
                            Berdasarkan konfigurasi anda:
                        </p>
                        <h1 style="font-size: 3.5rem; margin: 0;">
                            €{prediction:,.0f}
                        </h1>
                        <h2 style="font-size: 2.2rem; margin: 0; color: #FFD700;">
                            Rp {price_idr:,.0f}
                        </h2>
                        <p style="opacity: 0.9; margin-top: 1rem;">
                            Kurs asumsi: 1€ = Rp {EUR_TO_IDR:,.0f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Configuration summary
                st.markdown(f"""
                <h3 style="color: {COLORS['accent']}; font-weight:700;">
                📋 Ringkasan Konfigurasi
                </h3>
                """, unsafe_allow_html=True)
                summary_cols = st.columns(4)
                specs = [
                    (f"⚡ {cpu} GHz CPU", COLORS['accent']),
                    (f"💾 {ram} GB RAM", COLORS['accent']),
                    (f"💿 {ssd} GB SSD", COLORS['secondary']),
                    (f"💿 {hdd} GB HDD" if hdd > 0 else "💿 No HDD", COLORS['secondary']),
                    (f"🖥️ {inches}\" Display", COLORS['accent']),
                    (f"📐 {res_width}×{res_height}", COLORS['accent']),
                    (f"⚖️ {weight} kg", COLORS['accent']),
                    (f"✨ {'IPS Panel' if ips else 'No IPS'}", COLORS['info']),
                ]
                
                for i, (spec, color) in enumerate(specs):
                    with summary_cols[i % 4]:
                        st.markdown(f"""
                        <div style="background: {color}15; padding: 1 rem; border-radius: 10px; 
                                 border-left: 4px solid {color}; margin-bottom: 0.5rem;">
                            <p style="margin: 0; font-weight: 500; color: {COLORS['dark']};">{spec}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Price comparison
                st.markdown(f"""
                <h3 style="color: {COLORS['success']}; font-weight:700;">
                📊 Komparasi harga
                </h3>
                """, unsafe_allow_html=True)
                avg_price = df['Price (Euro)'].mean()
                price_diff = prediction - avg_price
                price_diff_pct = (price_diff / avg_price) * 100
                
                col_comp1, col_comp2 = st.columns(2)
                with col_comp1:
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom: 0.5rem;">
                        <div style="font-size: 1rem; opacity: 0.9;">🎯 Konfigurasi Anda</div>
                        <div style="font-size: 2.2rem; font-weight: 800;">
                            €{prediction:,.0f}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.95rem; ">
                            {price_diff_pct:+.1f}% vs average
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_comp2:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #4299e1 0%, #764ba2 100%); margin-bottom: 0.5rem;">
                        <div style="font-size: 1rem; opacity: 0.9;">📊 Rata-rata Harga Pasaran</div>
                        <div style="font-size: 2.2rem; font-weight: 800;">
                            €{avg_price:,.0f}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.95rem;">
                            Berdasarkan dataset
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                
                # Visualization of prediction vs actual distribution
                chart_data = pd.DataFrame({
                    'Category': ['Your Prediction', 'Market Average', 'Minimum', 'Maximum'],
                    'Price': [prediction, avg_price, df['Price (Euro)'].min(), df['Price (Euro)'].max()]
                })
                
                price_chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X(
                        'Category:N',
                        sort=['Your Prediction', 'Market Average', 'Minimum', 'Maximum'],
                        axis=alt.Axis(
                            labelColor="#EAF7F0",      # warna teks label
                            labelFontSize=12,
                            titleColor="#FE0303"
                        )
                    ),
                    y=alt.Y(
                        'Price:Q',
                        title='Price (€)'
                    ),
                    color=alt.Color(
                        'Category:N',
                        scale=alt.Scale(
                            domain=['Your Prediction', 'Market Average', 'Minimum', 'Maximum'],
                            range=[COLORS['success'], COLORS['info'], "#EAF7F0", "#EAF7F0"]
                        )
                    )
                ).properties(
                    height=300,
                    background="#111827" 
                ).configure_axis(
                    labelColor="#EAF7F0",
                    titleColor="#EAF7F0", 
                    labelFontSize=12,
                ).interactive()
                st.altair_chart(price_chart, use_container_width=True)

# ===============================
# FOOTER
# ===============================
# Creative Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='background: #FFF6E9;
padding: 25px; border-radius: 10px; text-align: center;
box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-top: 50px;'>

<p style='color: #000000; font-size: 14px; margin: 5px 0;'>
Copyright © 2026 Pengelola MK Praktikum Unggulan (Praktikum DGX), Universitas Gunadarma
</p>

<a href="https://www.praktikum-hpc.gunadarma.ac.id/" target="_blank"
style="display:block; color:#1a73e8; font-size:13px; margin-top:0px;">
https://www.praktikum-hpc.gunadarma.ac.id/
</a>
<a href="https://www.hpc-hub.gunadarma.ac.id/" target="_blank"
style="display:block; color:#1a73e8; font-size:13px; margin-top:0px;">
https://www.hpc-hub.gunadarma.ac.id/
</a>

</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import joblib
import holidays
from sentence_transformers import SentenceTransformer, util
from thefuzz import fuzz
from data_prep import load_and_clean_data, RECOMMENDATION_MAP, SEARCH_SYNONYMS
import torch
torch.set_num_threads(1)

st.set_page_config(page_title="Road Safety Intelligence", page_icon="🛣️", layout="wide")

th_holidays = holidays.Thailand(years=range(2019, 2027))

st.markdown("""<style>
.block-container{padding-top:0!important;padding-left:0!important;padding-right:0!important;max-width:100%!important}
[data-testid="stHeader"]{background:transparent!important;height:0}
[data-testid="stAppViewContainer"]{background:#060D1F}
[data-testid="stSidebar"]{background:#060D1F}
[data-baseweb="tab-list"]{background:#0A1628!important;border-radius:10px!important;padding:3px!important;gap:2px!important;border:1px solid #0F2040!important}
[data-baseweb="tab"]{background:transparent!important;border-radius:7px!important;color:#334155!important;font-weight:500!important;font-size:0.82rem!important;border:none!important}
[aria-selected="true"][data-baseweb="tab"]{background:#0F2040!important;color:#94A3B8!important}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important}
[data-testid="stMetric"]{background:#0A1628;border:1px solid #0F2040;border-radius:10px;padding:14px 16px}
[data-testid="stMetricLabel"] p{color:#334155!important;font-size:0.7rem!important;text-transform:uppercase;letter-spacing:0.08em;font-weight:600}
[data-testid="stMetricValue"]{color:#F97316!important;font-size:1.7rem!important;font-weight:700!important}
[data-testid="stButton"] button{background:#0A1628!important;border:1px solid #0F2040!important;color:#475569!important;border-radius:8px!important;font-weight:500!important;font-size:0.82rem!important;transition:all 0.12s!important}
[data-testid="stButton"] button:hover{background:#0F2040!important;border-color:#F97316!important;color:#CBD5E1!important}
[data-testid="stButton"] button[kind="primary"]{background:#F97316!important;border:none!important;color:#fff!important;width:auto!important;border-radius:8px!important;font-weight:600!important}
[data-baseweb="input"] input,[data-baseweb="select"] div{background:#0A1628!important;border-color:#0F2040!important;color:#94A3B8!important;border-radius:8px!important}
[data-testid="stTextInput"] input{background:#0A1628!important;border:1.5px solid #0F2040!important;border-radius:30px!important;color:#94A3B8!important;font-size:0.9rem!important;text-align:center}
[data-testid="stTextInput"] input:focus{border-color:rgba(249,115,22,0.5)!important}
[data-testid="stAlert"]{border-radius:8px!important;border-left-width:3px!important}
p,label,[data-testid="stMarkdownContainer"] p{color:#475569!important}
h1,h2,h3{color:#94A3B8!important}
hr{border-color:#0F2040!important}
[data-testid="stCaptionContainer"] p{color:#1E3A5F!important}
[data-testid="stDataFrame"]{border-radius:10px!important}
</style>""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_and_clean_data()


@st.cache_resource
def load_risk_model():
    model = joblib.load('accident_risk_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns


@st.cache_resource
def load_search_model(cause_categories):
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    model = SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')
    phrase_to_category = {}
    for category, synonyms in SEARCH_SYNONYMS.items():
        phrase_to_category[category] = category
        for phrase in synonyms:
            phrase_to_category[phrase] = category
    for cat in cause_categories:
        if cat not in phrase_to_category:
            phrase_to_category[cat] = cat
    all_phrases = list(phrase_to_category.keys())
    embeddings = model.encode([f"passage: {p}" for p in all_phrases], batch_size=16, show_progress_bar=False)
    return model, phrase_to_category, all_phrases, embeddings

def get_distinctive_cause(causes_series, global_freq, min_count=3):
    local_counts = causes_series.value_counts()
    local_counts = local_counts[~local_counts.index.isin(['ไม่ระบุ', 'อื่นๆ', 'อื่นๆ (สาเหตุพบน้อย)'])]
    local_counts = local_counts[local_counts >= min_count]
    if len(local_counts) == 0:
        return causes_series.value_counts().idxmax()  # fallback ถ้าเหลือแค่ค่าไม่มีประโยชน์
    local_props = local_counts / local_counts.sum()
    lift = local_props / global_freq.reindex(local_props.index)
    return lift.idxmax()


def hybrid_search(query, model, phrase_to_category, all_phrases, embeddings, top_k=3):
    query_embedding = model.encode(f"query: {query}")
    semantic_scores = util.cos_sim(query_embedding, embeddings)[0]
    category_best = {}
    for i, phrase in enumerate(all_phrases):
        category = phrase_to_category[phrase]
        sem = float(semantic_scores[i])
        lex = fuzz.partial_ratio(query, phrase) / 100
        score = 0.85 * sem + 0.15 * lex
        if category not in category_best or score > category_best[category]:
            category_best[category] = score
    return sorted(category_best.items(), key=lambda x: -x[1])[:top_k]


def prepare_input(input_dict, model_columns):
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    return input_encoded.reindex(columns=model_columns, fill_value=0)


df = get_data()
risk_model, model_columns = load_risk_model()
cause_categories = tuple(sorted(df['cause_clean'].unique()))
search_model, phrase_to_category, all_phrases, search_embeddings = load_search_model(cause_categories)

# ── NAVBAR ──
nav_left, nav_right = st.columns([1, 2])
with nav_left:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:16px 24px 0">
        <div style="background:#F97316;border-radius:8px;width:30px;height:30px;
                    display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0">🛣️</div>
        <div>
            <div style="font-size:13px;font-weight:700;color:#94A3B8;letter-spacing:-0.01em">Road Safety Intelligence</div>
            <div style="font-size:10px;color:#1E3A5F">Thailand · {int(df['year'].min())}–{int(df['year'].max())}</div>
        </div>
    </div>""", unsafe_allow_html=True)

with nav_right:
    st.markdown('<div style="padding:16px 24px 0;text-align:right">', unsafe_allow_html=True)
    with st.popover("⚙️ ตั้งค่าตัวกรอง"):
        selected_years = st.multiselect("ปี", options=sorted(df['year'].unique(), reverse=True), default=sorted(df['year'].unique(), reverse=True))
        selected_provinces = st.multiselect("จังหวัด", options=sorted(df['province'].unique()), default=sorted(df['province'].unique()))
        min_f, max_f = st.slider("จำนวนผู้เสียชีวิต", min_value=int(df['fatalities'].min()), max_value=int(df['fatalities'].max()), value=(0, int(df['fatalities'].max())))
    st.markdown('</div>', unsafe_allow_html=True)

filtered_df = df[
    (df['year'].isin(selected_years)) &
    (df['province'].isin(selected_provinces)) &
    (df['fatalities'] >= min_f) &
    (df['fatalities'] <= max_f)


]

if 'search_filter_cause' not in st.session_state:
    st.session_state.search_filter_cause = None



# ── HERO ──
total_fatalities = int(filtered_df['fatalities'].sum())
fatality_rate = filtered_df['fatalities'].mean() * 100

st.markdown(f"""
<div style="position:relative;padding:48px 24px 36px;text-align:center;overflow:hidden;
            border-bottom:1px solid #0F2040;margin-bottom:0">
    <svg style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <pattern id="sg2" width="28" height="28" patternUnits="userSpaceOnUse">
                <path d="M28 0L0 0 0 28" fill="none" stroke="#0EA5E9" stroke-width="0.2" opacity="0.3"/>
            </pattern>
            <pattern id="bg2" width="112" height="112" patternUnits="userSpaceOnUse">
                <rect width="112" height="112" fill="url(#sg2)"/>
                <path d="M112 0L0 0 0 112" fill="none" stroke="#0EA5E9" stroke-width="0.5" opacity="0.45"/>
            </pattern>
            <radialGradient id="vgn" cx="50%" cy="60%" r="55%">
                <stop offset="0%" stop-color="#060D1F" stop-opacity="0.1"/>
                <stop offset="100%" stop-color="#060D1F" stop-opacity="0.95"/>
            </radialGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#bg2)"/>
        <rect width="100%" height="100%" fill="url(#vgn)"/>
        <circle cx="15%" cy="35%" r="2.5" fill="#EF4444" opacity="0.55"/>
        <circle cx="28%" cy="72%" r="2" fill="#F97316" opacity="0.4"/>
        <circle cx="55%" cy="22%" r="3.5" fill="#EF4444" opacity="0.45"/>
        <circle cx="72%" cy="58%" r="2.5" fill="#F97316" opacity="0.35"/>
        <circle cx="82%" cy="28%" r="3" fill="#EF4444" opacity="0.4"/>
        <circle cx="88%" cy="75%" r="2" fill="#F97316" opacity="0.3"/>
        <line x1="15%" y1="35%" x2="28%" y2="72%" stroke="#F97316" stroke-width="0.4" opacity="0.15"/>
        <line x1="28%" y1="72%" x2="55%" y2="22%" stroke="#F97316" stroke-width="0.4" opacity="0.15"/>
        <line x1="55%" y1="22%" x2="72%" y2="58%" stroke="#F97316" stroke-width="0.4" opacity="0.15"/>
        <line x1="72%" y1="58%" x2="82%" y2="28%" stroke="#F97316" stroke-width="0.4" opacity="0.15"/>
    </svg>
    <div style="position:relative;z-index:1">
        <div style="display:inline-flex;align-items:center;gap:6px;background:rgba(249,115,22,0.08);
                    border:1px solid rgba(249,115,22,0.2);border-radius:20px;padding:4px 12px;margin-bottom:16px">
            <div style="width:5px;height:5px;background:#F97316;border-radius:50%"></div>
            <span style="color:#F97316;font-size:10px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase">
                AI-Powered Road Safety Analytics
            </span>
        </div>
        <h1 style="color:#F1F5F9!important;font-size:2.2rem;font-weight:700;letter-spacing:-0.03em;
                   line-height:1.1;margin-bottom:8px">ค้นหาจุดเสี่ยงอุบัติเหตุ</h1>
        <p style="color:#1E3A5F!important;font-size:0.88rem;margin-bottom:0">
            วิเคราะห์จาก <span style="color:#F97316;font-weight:600">{len(filtered_df):,}</span> เหตุการณ์จราจรทั่วประเทศไทย
        </p>
    </div>
</div>""", unsafe_allow_html=True)

# ── SEARCH BAR ──
_, sc, _ = st.columns([1, 2, 1])
with sc:
    st.markdown('<div style="padding:0 0 4px;margin-top:-1px">', unsafe_allow_html=True)
    hero_query = st.text_input("", placeholder="🔍  ค้นหาด้วย AI เช่น ถนนมืด, หลับใน, ฝนตกถนนลื่น...", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

if hero_query:
    results = hybrid_search(hero_query, search_model, phrase_to_category, all_phrases, search_embeddings, top_k=3)
    _, rc, _ = st.columns([1, 2, 1])
    with rc:
        r1, r2, r3 = st.columns(3)
        for i, (category, score) in enumerate(results):
            confidence = "สูง" if score > 0.85 else "ปานกลาง"
            col = [r1, r2, r3][i]
            with col:
                is_active = st.session_state.search_filter_cause == category
                label = f"{'✓ ' if is_active else ''}{category}\nความมั่นใจ: {confidence} · {score:.2f}"
                if st.button(label, key=f"hero_btn_{i}", use_container_width=True):
                    st.session_state.search_filter_cause = category

# ── STATS ROW ──
st.markdown('<div style="padding:20px 0 0">', unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns(4)
s1.metric("เหตุการณ์ทั้งหมด", f"{len(filtered_df):,}")
s2.metric("ผู้เสียชีวิต", f"{total_fatalities:,}")
s3.metric("อัตราเสียชีวิต", f"{fatality_rate:.1f}%")
s4.metric("จังหวัดครอบคลุม", f"{filtered_df['province'].nunique()}")
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ── TABS ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ แผนที่", "📊 สถิติ", "🏆 อันดับความเสี่ยง", "🔮 ทำนาย", "🔍 AI Search"
])

with tab1:
    if st.session_state.search_filter_cause:
        badge_col1, badge_col2 = st.columns([5, 1])
        with badge_col1:
            st.info(f"🔍 กรองด้วยผลค้นหา: **{st.session_state.search_filter_cause}**")
        with badge_col2:
            if st.button("✕ ล้างตัวกรอง", key="clear_search_filter"):
                st.session_state.search_filter_cause = None
                st.rerun()

    map_df = filtered_df.copy()
    if st.session_state.search_filter_cause:
        map_df = map_df[map_df['cause_clean'] == st.session_state.search_filter_cause]

    c_left, c_right = st.columns([3, 1])
    with c_left:
        st.caption("📍 จุดคลิกได้ — คลิกจุดสีบนแผนที่เพื่อดูรายละเอียด")
    with c_right:
        weight_mode = st.radio("", ["จำนวน", "ความรุนแรง"], horizontal=True, label_visibility="collapsed")

    grid_size_m = st.slider("ขนาดกริด (เมตร)", 500, 5000, 2000, step=500)
    grid_size_deg = grid_size_m / 111000
    grid_df = map_df.copy()
    grid_df['lat_grid'] = (grid_df['latitude'] / grid_size_deg).round() * grid_size_deg
    grid_df['lng_grid'] = (grid_df['longitude'] / grid_size_deg).round() * grid_size_deg

    global_cause_freq = filtered_df['cause_clean'].value_counts(normalize=True)
    global_road_freq = filtered_df['road_characteristic'].value_counts(normalize=True)

    grid_stats = grid_df.groupby(['lat_grid', 'lng_grid']).agg(
        count=('fatalities', 'count'), total_fatalities=('fatalities', 'sum'),
        avg_fatalities=('fatalities', 'mean'),
        top_cause=('cause_clean', lambda x: get_distinctive_cause(x, global_cause_freq)),
        top_road=('road_characteristic', lambda x: get_distinctive_cause(x, global_road_freq)),    ).reset_index()
    grid_stats = grid_stats[grid_stats['count'] >= 5]

    m = folium.Map(location=[13.7563, 100.5018], zoom_start=6, tiles='CartoDB dark_matter')
    heat_data = map_df[['latitude', 'longitude']].values.tolist() if weight_mode == "จำนวน" else map_df[['latitude', 'longitude', 'fatalities']].values.tolist()
    if heat_data:
        HeatMap(heat_data, radius=8, blur=10).add_to(m)
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in grid_stats.iterrows():
        recommendation = RECOMMENDATION_MAP.get(row['top_cause'], RECOMMENDATION_MAP['default'])
        popup_html = f"""<b>อุบัติเหตุ:</b> {row['count']}<br>
        <b>เสียชีวิต:</b> {int(row['total_fatalities'])}<br>
        <b>สาเหตุหลัก:</b> {row['top_cause']}<br>
        <b>ลักษณะถนน:</b> {row['top_road']}<br><br>
        <b>💡</b> {recommendation}"""
        color = 'red' if row['avg_fatalities'] > 0.2 else ('orange' if row['avg_fatalities'] > 0.05 else 'green')
        folium.CircleMarker(location=[row['lat_grid'], row['lng_grid']], radius=7,
                            color=color, fill=True, fill_color=color, fill_opacity=0.75,
                            popup=folium.Popup(popup_html, max_width=300)).add_to(marker_cluster)
    st_folium(m, width=None, height=580, key="main_map")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("สาเหตุที่พบบ่อย (Top 10)")
        st.bar_chart(filtered_df['cause_clean'].value_counts().head(10))
    with col2:
        st.subheader("แนวโน้มรายปี")
        st.line_chart(filtered_df.groupby('year').size())

with tab3:
    st.subheader("อันดับจังหวัดตามความเสี่ยง")
    min_cases = st.slider("เหตุการณ์ขั้นต่ำ", 1, 200, 30)
    province_stats = filtered_df.groupby('province').agg(
        total_cases=('fatalities', 'count'),
        total_fatalities=('fatalities', 'sum'),
        avg_fatality_rate=('fatalities', 'mean')
    ).reset_index()
    province_stats.columns = ['จังหวัด', 'จำนวนเหตุการณ์', 'ผู้เสียชีวิตรวม', 'อัตราเสียชีวิตเฉลี่ย']
    province_stats = province_stats[province_stats['จำนวนเหตุการณ์'] >= min_cases]
    st.dataframe(province_stats.sort_values('อัตราเสียชีวิตเฉลี่ย', ascending=False), use_container_width=True)

with tab4:
    st.subheader("ทำนายความเสี่ยง")
    st.caption("เลือกเงื่อนไขสมมติ แล้วให้โมเดลประเมินโอกาสรุนแรงถึงขั้นเสียชีวิต")
    col1, col2 = st.columns(2)
    with col1:
        input_province = st.selectbox("จังหวัด", sorted(df['province'].unique()))
        input_vehicle = st.selectbox("ประเภทยานพาหนะ", sorted(df['first_vehicle'].unique()))
        input_road = st.selectbox("ลักษณะถนน", sorted(df['road_characteristic'].unique()))
        input_cause = st.selectbox("สาเหตุ", sorted(df['cause_clean'].unique()))
        input_agency = st.selectbox("หน่วยงานดูแล", sorted(df['agency'].unique()))
    with col2:
        input_weather = st.selectbox("สภาพอากาศ", sorted(df['weather'].unique()))
        input_date = st.date_input("วันที่")
        input_hour = st.slider("ชั่วโมง (0–23)", 0, 23, 12)
    if st.button("🔮 ทำนายความเสี่ยง", type="primary"):
        is_night = 1 if (input_hour >= 22 or input_hour <= 4) else 0
        input_dict = {'province': input_province, 'first_vehicle': input_vehicle,
                      'road_characteristic': input_road, 'cause_clean': input_cause,
                      'weather': input_weather, 'hour': input_hour, 'agency': input_agency,
                      'is_night': is_night, 'day_of_week': input_date.strftime('%A'),
                      'is_holiday': 1 if str(input_date) in [str(d) for d in th_holidays] else 0}
        proba = risk_model.predict_proba(prepare_input(input_dict, model_columns))[0, 1]
        risk_percent = proba * 100
        st.divider()
        if proba >= 0.4:
            st.error(f"⚠️ ความเสี่ยงสูง — โอกาสรุนแรงถึงเสียชีวิต {risk_percent:.1f}%")
        elif proba >= 0.2:
            st.warning(f"🟡 ความเสี่ยงปานกลาง — {risk_percent:.1f}%")
        else:
            st.success(f"✅ ความเสี่ยงต่ำ — {risk_percent:.1f}%")
        st.caption("threshold 0.4 จากการวิเคราะห์ Precision-Recall trade-off")

with tab5:
    st.subheader("AI Magic Search")
    st.caption("พิมพ์คำอธิบายสถานการณ์ตามธรรมชาติ ระบบจะหาหมวดหมู่ที่ใกล้เคียงให้")
    query = st.text_input("", placeholder="เช่น: ถนนมืดมองไม่เห็น, คนขับหลับคาพวงมาลัย, ฝนตกถนนลื่น...", label_visibility="collapsed")
    if query:
        results = hybrid_search(query, search_model, phrase_to_category, all_phrases, search_embeddings, top_k=3)
        st.write("**ผลลัพธ์ที่ใกล้เคียงที่สุด — คลิกเพื่อดูแผนที่:**")
        col1, col2, col3 = st.columns(3)
        selected_cause = None
        for i, (category, score) in enumerate(results):
            confidence = "สูง" if score > 0.85 else "ปานกลาง" if score > 0.75 else "ต่ำ"
            with [col1, col2, col3][i]:
                if st.button(f"{category}\n(ความมั่นใจ: {confidence} {score:.2f})", key=f"btn_{i}"):
                    selected_cause = category
        if selected_cause:
            st.divider()
            cause_filtered = filtered_df[filtered_df['cause_clean'] == selected_cause]
            c1, c2, c3 = st.columns(3)
            c1.metric("จำนวนเหตุการณ์", f"{len(cause_filtered):,}")
            c2.metric("ผู้เสียชีวิตรวม", int(cause_filtered['fatalities'].sum()))
            c3.metric("อัตราเสียชีวิตเฉลี่ย", f"{cause_filtered['fatalities'].mean():.3f}")
            m_s = folium.Map(location=[13.7563, 100.5018], zoom_start=6, tiles='CartoDB dark_matter')
            if len(cause_filtered) > 0:
                HeatMap(cause_filtered[['latitude', 'longitude', 'fatalities']].values.tolist(), radius=10, blur=12).add_to(m_s)
            st_folium(m_s, width=None, height=480, key="search_map")
            recommendation = RECOMMENDATION_MAP.get(selected_cause, RECOMMENDATION_MAP['default'])
            st.warning(f"💡 **คำแนะนำ**: {recommendation}")
    st.caption("💬 ตัวอย่าง: 'มัวแต่เล่นมือถือ', 'รถเสียหลัก', 'วัววิ่งตัดหน้า', 'ขับรถทั้งคืนจนอ่อนล้า'")
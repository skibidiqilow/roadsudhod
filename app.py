import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import joblib
import holidays
from folium.plugins import HeatMap, MarkerCluster

from data_prep import load_and_clean_data, RECOMMENDATION_MAP

st.set_page_config(page_title="แผนที่จุดเสี่ยงอุบัติเหตุ", page_icon="🗺️", layout="wide")

th_holidays = holidays.Thailand(years=range(2019, 2027))


@st.cache_data
def get_data():
    return load_and_clean_data()


@st.cache_resource
def load_model():
    model = joblib.load('accident_risk_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns


def prepare_input(input_dict, model_columns):
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
    return input_final


df = get_data()
model, model_columns = load_model()

st.title("🗺️ แผนที่จุดเสี่ยงอุบัติเหตุ")

with st.popover("🔍 ค้นหา / กรองข้อมูล"):
    st.write("**เลือกเงื่อนไข**")

    selected_years = st.multiselect(
        "ปี",
        options=sorted(df['year'].unique(), reverse=True),
        default=sorted(df['year'].unique(), reverse=True)
    )

    selected_provinces = st.multiselect(
        "จังหวัด",
        options=sorted(df['province'].unique()),
        default=sorted(df['province'].unique())
    )

    min_f, max_f = st.slider(
        "จำนวนผู้เสียชีวิต (fatalities)",
        min_value=int(df['fatalities'].min()),
        max_value=int(df['fatalities'].max()),
        value=(0, int(df['fatalities'].max()))
    )

filtered_df = df[
    (df['year'].isin(selected_years)) &
    (df['province'].isin(selected_provinces)) &
    (df['fatalities'] >= min_f) &
    (df['fatalities'] <= max_f)
]

st.metric("จำนวนเหตุการณ์ที่พบ (case)", f"{len(filtered_df):,}")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ แผนที่", "📊 สถิติเชิงลึก", "🏆 อันดับความเสี่ยง", "🔮 ทำนายความเสี่ยง"
])

with tab1:
    weight_mode = st.radio(
        "รูปแบบ heatmap พื้นหลัง",
        ["จำนวนเหตุการณ์ (ดิบ)", "ความรุนแรง (ถ่วงน้ำหนักด้วย fatalities)"],
        horizontal=True
    )

    grid_size_m = st.slider(
        "ขนาดกริดสำหรับรวมจุด (เมตร) — ยิ่งเล็กยิ่งละเอียดแต่จุดเยอะขึ้น",
        500, 5000, 2000, step=500
    )
    grid_size_deg = grid_size_m / 111000

    grid_df = filtered_df.copy()
    grid_df['lat_grid'] = (grid_df['latitude'] / grid_size_deg).round() * grid_size_deg
    grid_df['lng_grid'] = (grid_df['longitude'] / grid_size_deg).round() * grid_size_deg

    grid_stats = grid_df.groupby(['lat_grid', 'lng_grid']).agg(
        count=('fatalities', 'count'),
        total_fatalities=('fatalities', 'sum'),
        avg_fatalities=('fatalities', 'mean'),
        top_cause=('cause_clean', lambda x: x.value_counts().idxmax()),
        top_road=('road_characteristic', lambda x: x.value_counts().idxmax()),
    ).reset_index()

    grid_stats = grid_stats[grid_stats['count'] >= 5]

    st.caption(f"📍 {len(grid_stats):,} จุดคลิกได้ (รวมเฉพาะกริดที่มี ≥5 เหตุการณ์) — คลิกจุดสีบนแผนที่เพื่อดูรายละเอียด")

    m = folium.Map(location=[13.7563, 100.5018], zoom_start=6)

    if weight_mode == "จำนวนเหตุการณ์ (ดิบ)":
        heat_data = filtered_df[['latitude', 'longitude']].values.tolist()
    else:
        heat_data = filtered_df[['latitude', 'longitude', 'fatalities']].values.tolist()

    if heat_data:
        HeatMap(heat_data, radius=8, blur=10).add_to(m)

    marker_cluster = MarkerCluster().add_to(m)

    for _, row in grid_stats.iterrows():
        recommendation = RECOMMENDATION_MAP.get(row['top_cause'], RECOMMENDATION_MAP['default'])

        popup_html = f"""
        <b>จำนวนอุบัติเหตุ:</b> {row['count']}<br>
        <b>ผู้เสียชีวิตรวม:</b> {int(row['total_fatalities'])}<br>
        <b>สาเหตุหลัก:</b> {row['top_cause']}<br>
        <b>ลักษณะถนนที่พบบ่อย:</b> {row['top_road']}<br><br>
        <b>💡 คำแนะนำ:</b> {recommendation}
        """

        color = 'red' if row['avg_fatalities'] > 0.2 else ('orange' if row['avg_fatalities'] > 0.05 else 'green')

        folium.CircleMarker(
            location=[row['lat_grid'], row['lng_grid']],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(marker_cluster)

    st_folium(m, width=1200, height=600, key="main_map")

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("สาเหตุที่พบบ่อยที่สุด (Top 10)")
        top_causes = filtered_df['cause_clean'].value_counts().head(10)
        st.bar_chart(top_causes)

    with col2:
        st.subheader("แนวโน้มจำนวนเหตุการณ์ต่อปี")
        yearly = filtered_df.groupby('year').size()
        st.line_chart(yearly)

with tab3:
    st.subheader("จัดอันดับจังหวัดตามความเสี่ยง")

    min_cases = st.slider(
        "จำนวนเหตุการณ์ขั้นต่ำ (กันจังหวัดที่มีข้อมูลน้อยเกินไปจนค่าเฉลี่ยไม่น่าเชื่อถือ)",
        1, 200, 30
    )

    province_stats = filtered_df.groupby('province').agg(
        total_cases=('fatalities', 'count'),
        total_fatalities=('fatalities', 'sum'),
        avg_fatality_rate=('fatalities', 'mean')
    ).reset_index()

    province_stats.columns = ['จังหวัด', 'จำนวนเหตุการณ์', 'ผู้เสียชีวิตรวม', 'อัตราเสียชีวิตเฉลี่ย']
    province_stats = province_stats[province_stats['จำนวนเหตุการณ์'] >= min_cases]
    province_stats = province_stats.sort_values('อัตราเสียชีวิตเฉลี่ย', ascending=False)

    st.dataframe(province_stats, use_container_width=True)

with tab4:
    st.subheader("ทำนายความเสี่ยงจากสถานการณ์ที่กำหนด")
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
        input_hour = st.slider("ชั่วโมงที่เกิดเหตุ (0-23)", 0, 23, 12)

    if st.button("🔮 ทำนายความเสี่ยง", type="primary"):
        is_night = 1 if (input_hour >= 22 or input_hour <= 4) else 0
        day_of_week = input_date.strftime('%A')
        is_holiday = 1 if str(input_date) in [str(d) for d in th_holidays] else 0

        input_dict = {
            'province': input_province,
            'first_vehicle': input_vehicle,
            'road_characteristic': input_road,
            'cause_clean': input_cause,
            'weather': input_weather,
            'hour': input_hour,
            'agency': input_agency,
            'is_night': is_night,
            'day_of_week': day_of_week,
            'is_holiday': is_holiday,
        }

        X_input = prepare_input(input_dict, model_columns)
        proba = model.predict_proba(X_input)[0, 1]
        risk_percent = proba * 100

        st.divider()

        if proba >= 0.4:
            st.error(f"⚠️ ความเสี่ยงสูง — โอกาสรุนแรงถึงเสียชีวิต {risk_percent:.1f}%")
        elif proba >= 0.2:
            st.warning(f"🟡 ความเสี่ยงปานกลาง — โอกาสรุนแรงถึงเสียชีวิต {risk_percent:.1f}%")
        else:
            st.success(f"✅ ความเสี่ยงต่ำ — โอกาสรุนแรงถึงเสียชีวิต {risk_percent:.1f}%")

        st.caption("เกณฑ์ 'เสี่ยงสูง' ใช้ threshold 0.4 ตามที่วิเคราะห์ trade-off ไว้ในสัปดาห์ที่ 3")
# 🚗 Road Safety & Smart Mobility Challenge
> **Project:** Smart Mobility Challenge (Hack the Streets with AI)  
> **Used for:** SAIG  
> **Topic:** ยกระดับความปลอดภัย ลดอุบัติเหตุ และแก้ปัญหาการเดินทางบนท้องถนน  

---

## 📌 About The Project
โปรเจกต์นี้จัดทำขึ้นเพื่อใช้ในการประเมินและยื่นสมัครเข้าแล็บ **SAIG** โดยได้รับแรงบันดาลใจจากโจทย์ **Smart Mobility Challenge** ซึ่งเน้นการนำเทคโนโลยี AI และการวิเคราะห์ข้อมูลมาช่วยแก้ไข **Pain Point** เรื่องอุบัติเหตุและความปลอดภัยบนท้องถนนในประเทศไทย

ปัจจุบันโปรเจกต์อยู่ในช่วง ทดลองวิจัยเปรียบเทียบประสิทธิภาพหลายๆ โมเดล สำหรับ repository นี้จึงเน้นแสดงกระบวนการ **Data Pipeline, Data Cleaning, EDA (Exploratory Data Analysis)** รวมถึงการทำ Feature Engineering เพื่อเตรียมข้อมูลอุบัติเหตุให้พร้อมที่สุดก่อนนำไป Train โมเดลครับ

---

## 🛠️ Data Pipeline & Data Cleaning Process
ไฟล์ข้อมูลที่ใช้เป็น dataset อุบัติเหตุบนท้องถนนจากแล็บ/หน่วยงาน เพื่อนำมาสำรวจและเตรียมข้อมูลผ่านสคริปต์ `data_prep.py` และ `explore.ipynb` โดยมีขั้นตอนสำคัญดังนี้:

1. **Handling Missing & Invalid Data:**
   * ตรวจสอบและจัดการค่าสูญหาย (Missing values) ในคอลัมน์สำคัญ เช่น พิกัดตำแหน่ง, เวลาเกิดเหตุ, และประเภทยานพาหนะ
2. **Feature Engineering & Transformation:**
   * แปลงข้อมูลวันที่และเวลาให้อยู่ในฟอร์แมตมาตรฐาน (Datetime Extraction) เพื่อดูแนวโน้มช่วงเวลาเสี่ยง
   * สร้าง Feature เพิ่มเติมสำหรับการจำแนกความรุนแรงของอุบัติเหตุ (Accident Severity Classification)
3. **Geospatial & Visualization Preparation:**
   * ประมวลผลพิกัดละติจูด/ลองจิจูด เพื่อนำไปสร้าง Map Visualization (`test_map.html`, `heatmap_full.html`) สำหรับดูจุดเสี่ยง (Blackspots) บนท้องถนน

---

## 📂 Repository Structure
```text
├── data_prep.py          # สคริปต์หลักสำหรับ Clean ข้อมูลและทำ Feature Engineering
├── explore.ipynb         # Jupyter Notebook สำหรับทำการสำรวจข้อมูล (EDA)
├── model.ipynb           # Notebook สำหรับทดลองสร้างและวัดผลโมเดลต่างๆ
├── app.py                # Dashboard/Web Application เบื้องต้น (Streamlit)
├── heatmap_full.html     # Interactive Visualization แสดงจุดเสี่ยงอุบัติเหตุ
├── README.md             # เอกสารอธิบายโปรเจกต์
└── .gitignore            # กำหนดไฟล์ที่ไม่ต้องเอาขึ้น Git (Dataset / Saved Models)
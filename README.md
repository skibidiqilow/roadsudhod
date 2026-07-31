# Thai Road Safety Intelligence

โปรเจกต์นี้ทำขึ้นในฐานะงาน Smart Mobility Challenge โดยนำข้อมูลอุบัติเหตุ
บนท้องถนนในประเทศไทย 151,778 เหตุการณ์ (ปี 2019–2025) มาสร้างเป็น
web application ที่ช่วยวิเคราะห์จุดเสี่ยงและสนับสนุนการตัดสินใจเชิงนโยบาย

## Dataset ที่ใช้

- **thai_accidental_dataset.csv** (Dataset 1) — สถิติอุบัติเหตุทั่วประเทศ
  พร้อมพิกัด lat/long, สาเหตุ, ลักษณะถนน, สภาพอากาศ, จำนวนผู้บาดเจ็บ/เสียชีวิต
- **Thai Holiday Calendar** (holidays library) — ข้อมูลวันหยุดราชการไทย
  เพิ่มเป็น feature เสริมในโมเดล

## สิ่งที่ app ทำได้

**แผนที่ interactive** — heatmap จุดเสี่ยงทั่วประเทศ คลิกจุดใดก็ได้เพื่อดูสาเหตุหลัก
และคำแนะนำเชิงมาตรการสำหรับพื้นที่นั้น

**โมเดลทำนายความรุนแรง** — XGBoost binary classifier ทำนายว่าอุบัติเหตุจะ
รุนแรงถึงขั้นมีผู้เสียชีวิตหรือไม่ จาก 10 features (จังหวัด, ประเภทยานพาหนะ,
ลักษณะถนน, สาเหตุ, สภาพอากาศ, ช่วงเวลา, หน่วยงาน, กลางคืน, วันหยุด)

**AI Magic Search** — ค้นหาด้วยภาษาธรรมชาติ เช่น "ถนนมืดมองไม่เห็น" หรือ
"คนขับหลับคาพวงมาลัย" ระบบใช้ multilingual sentence embedding + hybrid scoring
หาหมวดสาเหตุที่ใกล้เคียงที่สุดจาก 43 หมวดในข้อมูลจริง

## ผลลัพธ์โมเดล

| | ค่าที่ได้ |
|---|---|
| Recall (ตาย) | 0.78 |
| Precision (ตาย) | 0.20 |
| Threshold | 0.4 |
| AI Search Top-3 accuracy | 96% (24/25 test cases) |

threshold 0.4 เลือกจากการวิเคราะห์ Precision-Recall trade-off
เนื่องจาก use case นี้ยอม false positive เพื่อแลกกับการไม่พลาดจุดเสี่ยงจริง

## วิธีรัน

```bash
git clone hhttps://github.com/skibidiqilow/roadsudhod.git
cd road-safety-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

ในกรณีที่เปิดไม่ได้ (segfault)
PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=1 streamlit run app.py

(https://thai-road-safety.streamlit.app)(มีปัญหาเรื่อง Resource Limits)


หมายเหตุ: ต้องมีไฟล์ `thai_accidental_dataset.csv` วางไว้ในโฟลเดอร์เดียวกัน


และต้องมีไฟล์ `accident_risk_model.pkl` กับ `model_columns.pkl`
ซึ่ง generate ได้จากการรัน `model.ipynb` ตั้งแต่ต้นจนจบ

## โครงสร้างไฟล์

```
road-safety-project/
├── app.py                    # Streamlit web application
├── data_prep.py              # Data cleaning pipeline (shared)
├── model.ipynb               # EDA + model training notebook
├── accident_risk_model.pkl   # Trained XGBoost model
├── model_columns.pkl         # Feature columns list
├── requirements.txt
└── README.md
```
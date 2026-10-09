from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="توقع أسعار المنازل",
    page_icon="🏡",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "linear_regression_pipeline.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


st.title("🏡 توقع قيمة المنزل")
st.write(
    "أدخل خصائص المنطقة للحصول على قيمة تقديرية باستخدام نموذج "
    "Linear Regression المدرّب على بيانات California Housing."
)
st.caption("التوقع إرشادي، وليس تقييمًا عقاريًا رسميًا.")

model = load_model()
if model is None:
    st.error(
        "ملف الموديل غير موجود: linear_regression_pipeline.pkl\n\n"
        "شغّل خلية حفظ الموديل في الـ Notebook بعد تدريب linear_model، "
        "ثم ضع الملف الناتج بجوار app.py."
    )
    st.stop()

with st.form("prediction_form"):
    st.subheader("بيانات الموقع")
    col1, col2 = st.columns(2)
    with col1:
        longitude = st.number_input(
            "خط الطول (longitude)", min_value=-125.0, max_value=-113.0,
            value=-122.0, step=0.1, format="%.4f"
        )
        latitude = st.number_input(
            "خط العرض (latitude)", min_value=32.0, max_value=42.5,
            value=37.0, step=0.1, format="%.4f"
        )
        housing_median_age = st.number_input(
            "متوسط عمر المنازل", min_value=1.0, max_value=100.0,
            value=30.0, step=1.0
        )
    with col2:
        median_income = st.number_input(
            "متوسط الدخل (بوحدة مجموعة البيانات)", min_value=0.0,
            max_value=20.0, value=3.5, step=0.1
        )
        ocean_proximity = st.selectbox(
            "القرب من المحيط",
            ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
        )

    st.subheader("بيانات المنازل والسكان")
    col3, col4 = st.columns(2)
    with col3:
        total_rooms = st.number_input(
            "إجمالي الغرف", min_value=1, value=2000, step=100
        )
        total_bedrooms = st.number_input(
            "إجمالي غرف النوم", min_value=1, value=400, step=10
        )
    with col4:
        population = st.number_input(
            "عدد السكان", min_value=1, value=1000, step=50
        )
        households = st.number_input(
            "عدد الأسر", min_value=1, value=350, step=10
        )

    submitted = st.form_submit_button(
        "توقّع قيمة المنزل", type="primary", use_container_width=True
    )

if submitted:
    if total_bedrooms > total_rooms:
        st.warning("تنبيه: عدد غرف النوم أكبر من إجمالي الغرف. راجع القيم المدخلة.")

    # Keep the feature names and engineered features consistent with training.
    input_data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
        "rooms_per_household": total_rooms / households,
        "bedrooms_per_room": total_bedrooms / total_rooms,
        "population_per_household": population / households,
    }])

    try:
        prediction = float(model.predict(input_data)[0])
        st.success("تم حساب التوقع")
        st.metric("القيمة المتوقعة للمنزل", f"${prediction:,.0f}")
        st.caption(
            "الموديل يتنبأ بالمتغير median_house_value كما هو معرّف في التدريب."
        )
    except Exception as exc:
        st.error(
            "حصل خطأ أثناء التوقع. تأكد أن الموديل محفوظ من نفس نسخة الـ Notebook "
            "وأن أسماء الخصائص مطابقة.\n\n"
            f"تفاصيل الخطأ: {exc}"
        )

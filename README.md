# Linear Regression + Streamlit — California Housing

## الملفات
- `app.py`: واجهة Streamlit عربية للتنبؤ بقيمة المنزل.
- `linear_regression_pipeline.pkl`: ملف الموديل المدرّب (يُنشأ من الـ Notebook).
- `requirements.txt`: المكتبات المطلوبة.

## 1) احفظ الموديل من الـ Notebook
بعد تشغيل الخلايا التي تدرب `linear_model` بنجاح، أضف خلية في نهاية الـ Notebook وشغّل:

```python
import joblib
joblib.dump(linear_model, "linear_regression_pipeline.pkl")
```

هذا يحفظ الـ Pipeline كاملًا، بما فيه معالجة القيم الناقصة، والـ Scaling، والـ One-Hot Encoding، وLinear Regression.

## 2) جهّز مجلد التطبيق
ضع الملف `linear_regression_pipeline.pkl` الناتج في نفس المجلد الذي يحتوي على `app.py`.

## 3) التثبيت والتشغيل
من Terminal داخل مجلد المشروع:

```bash
python -m venv .venv
```

على Windows:
```bash
.venv\Scripts\activate
```

ثم:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ملاحظات مهمة
- يجب أن يكون الموديل محفوظًا من نفس نسخة البيانات والكود التي درّبت `linear_model`.
- الواجهة تحسب الخصائص الهندسية الثلاثة بنفس الصيغ الموجودة في الـ Notebook.
- لا تفتح ملف `.pkl` من مصدر غير موثوق، لأن ملفات Pickle/Joblib قد تنفذ محتوى غير آمن عند تحميلها.

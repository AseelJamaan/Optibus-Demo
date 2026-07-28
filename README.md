# 🚌 OptiBus – AI-Based School Bus Routing System

OptiBus is an intelligent school bus routing system that optimizes school transportation using Artificial Intelligence. The system automatically groups students into buses, generates optimized morning and afternoon routes using a PPO-based DRL-ALNS model, and estimates student arrival times.

---

## 🌐 Live Demo

🔗 **Open the application:**

https://optibus-demo.onrender.com/

> **Note:** The application is hosted on Render's free tier. If the service has been idle, the first launch may take around one minute while the server starts. :contentReference[oaicite:0]{index=0}

---

# 🇺🇸 How to Use

Using the system is very simple:

### 1. Open the demo
Visit the live demo using the link above.

### 2. Upload your merged student CSV file
The file should contain:
- Student information
- School location
- Distance matrix
- Time matrix

### 3. Generate buses
Click **Generate Buses**.

The system will:
- Cluster students automatically.
- Determine the optimal number of buses.
- Display the number of students assigned to each bus.

### 4. Select a bus
Choose the bus you want to optimize.

### 5. Select students
Keep all students selected, or remove absent students if needed.

### 6. Generate routes
Click **Generate Routes**.

The system will generate:
- Morning route
- Afternoon route
- Route map
- Route statistics
- Estimated arrival times (ETA)

### 7. Notifications
The system can generate ETA information that can be used for WhatsApp parent notifications.

---

# 🇸🇦 طريقة الاستخدام

استخدام النظام بسيط جدًا:

### ١. افتح الموقع
ادخل على الرابط الموجود بالأعلى.

### ٢. ارفع ملف الطلاب
قم برفع ملف CSV الذي يحتوي على:
- بيانات الطلاب
- موقع المدرسة
- مصفوفة المسافات
- مصفوفة أزمنة الرحلات

### ٣. تجهيز الباصات
اضغط على زر **Generate Buses**.

سيقوم النظام تلقائيًا بـ:
- توزيع الطلاب على الباصات.
- تحديد العدد المناسب من الباصات.
- عرض عدد الطلاب في كل باص.

### ٤. اختر الباص
اختر الباص الذي تريد إنشاء مساره.

### ٥. اختر الطلاب
يمكنك الإبقاء على جميع الطلاب أو إزالة الطلاب الغائبين.

### ٦. إنشاء المسارات
اضغط على **Generate Routes**.

سيقوم النظام بعرض:
- مسار الذهاب.
- مسار العودة.
- الخريطة.
- إحصائيات الرحلة.
- الوقت المتوقع لوصول كل طالب (ETA).

### ٧. الإشعارات
يمكن استخدام بيانات ETA لإرسال إشعارات واتساب لأولياء الأمور.

---

## ✨ Features

- AI-based student clustering
- PPO-based DRL-ALNS route optimization
- Morning & afternoon route planning
- Interactive route visualization
- ETA estimation
- WhatsApp notification support
- User-friendly web interface

---

## 🛠 Technologies

- Python
- Gradio
- Stable-Baselines3 (PPO)
- Gymnasium
- Scikit-learn
- Folium
- Pandas
- NumPy

---

## 📄 Research

This project was developed as part of our Bachelor's Graduation Project in Artificial Intelligence at the University of Jeddah.

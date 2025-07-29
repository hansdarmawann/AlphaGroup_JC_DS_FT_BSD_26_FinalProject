# Meningkatkan Retensi Pelanggan dengan Memprediksi Customer Churn 📉📱
By JCDS2602 - Alpha Team (Abe, Alfi, and Hans)

Tableau Link: https://public.tableau.com/shared/DXG5TMGCS?:display_count=n&:origin=viz_share_link

![Screenshot](https://github.com/PurwadhikaDev/AlphaGroup_JC_DS_FT_BSD_26_FinalProject/blob/main/Assets/dash1.jpg)
![Screenshot](https://github.com/PurwadhikaDev/AlphaGroup_JC_DS_FT_BSD_26_FinalProject/blob/main/Assets/dash2.jpg)
![Screenshot](https://github.com/PurwadhikaDev/AlphaGroup_JC_DS_FT_BSD_26_FinalProject/blob/main/Assets/dash3.jpg)


Streamlit Link: https://finpro-telco-churn.streamlit.app/
(If error, please run streamlit run telco_churn_app.py on your terminal.)
—-

## 1. Ringkasan Proyek
Proyek ini bertujuan untuk menganalisis data pelanggan perusahaan telekomunikasi guna memprediksi kemungkinan pelanggan berhenti berlangganan (*churn*). Dengan memanfaatkan machine learning, perusahaan dapat meningkatkan strategi retensi dan mengurangi kehilangan pendapatan akibat churn.

### Tujuan Utama:
- 🎯 **Tujuan 1:** Mengidentifikasi faktor-faktor utama yang menyebabkan churn melalui eksplorasi data dan analisis feature importance.
- 🧠 **Tujuan 2:** Membangun dan mengoptimalkan model klasifikasi untuk memprediksi churn pelanggan.
- 💰 **Tujuan 3:** Mengestimasi dampak finansial dari churn dan mengidentifikasi pelanggan berisiko tinggi untuk strategi proaktif.
- 📊 **Tujuan 4:** Menyediakan rekomendasi strategis yang dapat diimplementasikan oleh tim pemasaran dan manajemen pelanggan.

—

## 2. Sumber Data
**Telco Customer Churn Dataset:**

- Berisi informasi pelanggan seperti jenis kontrak, layanan tambahan, metode pembayaran, lama berlangganan, dan status churn.
- Sumber: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/)

—

## 3. Teknologi yang Digunakan
- **Bahasa Pemrograman:** Python (Pandas, NumPy, Scikit-learn)
- **Visualisasi:** Matplotlib, Seaborn
- **Modeling:** Logistic Regression, Random Forest, SVM, KNN, Ensemble (Voting, Bagging, Boosting, Stacking)
- **Lingkungan Kerja:** Jupyter Notebook
- **Deployment:** Model disimpan dalam format `.sav` dan `.pkl`
- **Versi Kontrol:** Git

—

## 4. Struktur Proyek
```
📁 final_project_alpha_team/
├── Data/                         # Dataset mentah dan hasil EDA
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── telco_data_eda.csv
├── Model/                        # File model terlatih
│   ├── Model_LogReg_Telco_Churn.sav
│   └── Model_Logreg_Telco_Churn_joblib.pkl
├── Notebook/
│   └── Telo Customer Churn Alpha Team.ipynb
```

—-

## 5. Ringkasan Temuan
### 5.1 Insight Bisnis
**Faktor Utama Churn:**
- Jenis kontrak bulanan memiliki risiko churn tertinggi.
- Pengguna yang tidak menggunakan layanan tambahan (seperti internet) lebih mungkin churn.
- Metode pembayaran otomatis cenderung menurunkan churn.

**Segmentasi Risiko Tinggi:**
- Pelanggan kontrak bulanan tanpa layanan tambahan memiliki probabilitas churn tertinggi.
- Pelanggan baru dengan lama berlangganan pendek lebih rentan churn.

—

### 5.2 Rekomendasi Strategis
🛠️ **Strategi Operasional**
- Prioritaskan intervensi terhadap pelanggan berisiko tinggi (misalnya melalui email atau penawaran eksklusif).
- Tawarkan kontrak jangka panjang dengan insentif menarik untuk mengurangi churn.

📈 **Optimalisasi Pemasaran**
- Kirim promo khusus kepada pelanggan dengan lama berlangganan < 6 bulan.
- Edukasi manfaat layanan tambahan untuk meningkatkan loyalitas.

📊 **Integrasi Sistem**
- Integrasikan model prediktif ke dalam CRM untuk memberi alert otomatis jika risiko churn > 70%.
- Gunakan skor churn untuk mendukung pengambilan keputusan di tim CS dan retention.

—-

## 6. Keterbatasan Model
- Tidak semua fitur penting tersedia (misalnya: interaksi customer service, feedback pelanggan).
- Masih terdapat data duplikat dan outlier yang memerlukan pembersihan lanjutan.
- Model belum diuji dalam skenario produksi aktual.
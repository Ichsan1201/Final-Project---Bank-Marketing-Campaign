# Final Project — Bank Marketing Campaign (Term Deposit Prediction)

## Konteks Bisnis

Bank Portugal menjalankan telemarketing untuk menawarkan produk deposito berjangka. Dari **41.172 kontak** yang dianalisis, hanya **11,27%** yang berhasil menempatkan dana — rasio keberhasilan yang rendah dan menandakan strategi campaign belum efisien. Rata-rata satu nasabah dihubungi **2,57 kali**, menghasilkan total **105.735 panggilan** dengan estimasi biaya **61.764 Euro**, atau sekitar **13,31 Euro per deposan yang berhasil didapat**.

**Problem Statement:** Nasabah seperti apa yang berpotensi membuka deposito, agar biaya telemarketing lebih efisien?

## Pendekatan

Proyek ini menggunakan pipeline data science lengkap: *Business Understanding → Data Cleaning → EDA (statistical testing) → Feature Engineering → Modelling (dengan hyperparameter tuning) → Evaluation berbasis nilai ekonomi → Business Recommendation*.

Metrik evaluasi utama yang dipilih adalah **F2-score**, karena problem statement memprioritaskan **Recall** (meminimalkan nasabah potensial yang terlewat / False Negative) — konsisten dengan struktur biaya bisnis di mana kehilangan satu deposan (~5.000 Euro) jauh lebih mahal daripada satu panggilan sia-sia (~1,5 Euro).

## Temuan Kunci (EDA & Statistical Testing)

Seluruh temuan berikut diuji signifikansinya (Two-Proportion Z-test / Mann-Whitney U, α = 0,05):

| Dimensi | Insight |
|---|---|
| **Riwayat kampanye (`poutcome`)** | Fitur kategorikal dengan asosiasi terkuat terhadap target (Cramér's V 0,320). Nasabah yang pernah sukses deposito sebelumnya jauh lebih berpotensi deposito lagi. |
| **Usia** | Berpola **U**, bukan linear — konversi tertinggi pada usia >60 dan <26 tahun, selisih +36,89 poin persentase vs usia 35–44 (z = 34,43). |
| **Pekerjaan** | Pelajar (31,43%) & pensiunan (25,26%) berkonversi jauh di atas baseline, namun hanya menerima 6,3% total panggilan — kapasitas terbesar justru diarahkan ke segmen paling tidak produktif. |
| **Kondisi makroekonomi (`euribor3m`)** | **Faktor paling dominan.** Saat suku bunga acuan < 1,5%, konversi mencapai 24,02%; saat tinggi, hanya 4,84%. "Bulan emas" (Maret, September, Oktober, Desember) ternyata bukan efek musiman, melainkan proksi dari periode suku bunga rendah. |
| **Kanal kontak** | Seluler berkonversi 14,74% vs 5,23% pada telepon rumah; konversi terus menurun setelah panggilan ke-4. |

## Model & Performa

- **Model final:** HistGradientBoosting (hasil `RandomizedSearchCV`), dipilih via 5-fold cross-validation memakai F2-score.
- **Perbandingan model:**

| Model | F2 (CV) |
|---|---:|
| Dummy Classifier (lantai bawah) | 0,1141 |
| Logistic Regression (baseline) | 0,5347 |
| Random Forest | 0,5503 |
| **HistGradientBoosting (tuned)** | **0,5594** |

- **Performa di data uji (unseen):** F2 = **0,5807** (ambang 0,13), PR-AUC 0,4884, ROC-AUC 0,8163 — tidak ada indikasi overfitting signifikan terhadap skor CV.
- **Feature importance (permutation):** `euribor3m` mendominasi dengan **~79,6%** dari total importance — mengonfirmasi bahwa kondisi makroekonomi eksternal jauh lebih menentukan keberhasilan campaign dibanding karakteristik individu nasabah.

## Dampak Bisnis

Dibandingkan dengan skor prioritas manual (6 aturan bisnis sederhana berbasis EDA), model machine learning pada **30% populasi teratas**:

| Strategi | Deposan Tertangkap | Lift vs Acak |
|---|---:|---:|
| Menelepon acak | ~278 | 1,00x |
| Skor prioritas manual | 582 | 2,09x |
| **Model Machine Learning** | **692** | **2,49x** |

Selisih **110 deposan tambahan** pada kapasitas panggilan yang identik (biaya sama, tanpa tambahan anggaran) — setara **~550.000 Euro nilai penempatan** pada data uji, atau berpotensi **~2,75 juta Euro per kampanye** bila diskalakan ke seluruh basis nasabah.

**Temuan yang berlawanan dengan intuisi:** Analisis berbasis nilai ekonomi (bukan F2) menunjukkan bahwa **tidak ada ambang yang mengalahkan strategi "hubungi semua nasabah"**. Titik impas satu panggilan hanya 0,03%, sedangkan segmen paling lemah sekalipun masih berkonversi 4,21%. Artinya, efisiensi yang tepat **bukan mengurangi jumlah panggilan, melainkan memperbaiki urutan prioritas siapa yang ditelepon lebih dulu**.

## Rekomendasi Utama

1. **Urutkan antrean dialer berdasarkan skor model** — jangan menyaring daftar. Nasabah berskor rendah tetap dihubungi pada sisa kapasitas.
2. **Batas keras 4 panggilan per nasabah**, alihkan sisa kapasitas ke nasabah yang belum tersentuh (hemat 19,3% beban kerja, risiko kehilangan hanya 6,6% deposan).
3. **Prioritaskan kanal seluler** — saat ini 36,53% panggilan masih diarahkan ke kanal yang terbukti paling lemah.
4. **Jadikan level Euribor sebagai pemicu intensitas kampanye**, bukan kalender bulanan.
5. **Validasi nilai ekonomis per deposan** bersama tim Finance (margin bunga bersih, bukan nominal penempatan) — seluruh angka Euro pada analisis ini adalah batas atas.
6. **Deploy bertahap dengan pemantauan** — jalankan berdampingan dengan strategi berjalan (champion-challenger) selama satu kampanye penuh, latih ulang tiap kuartal, dan pantau *drift* pada `euribor3m`.

## Batasan Utama

- Temuan bersifat **asosiatif, bukan kausal** — dataset historis tanpa randomisasi.
- Model sangat bergantung pada satu variabel makroekonomi (`euribor3m`, ~79,6% importance) sehingga rentan terhadap perubahan rezim suku bunga.
- Kolom `month` tidak menyertakan tahun, sehingga pola musiman tidak sepenuhnya bisa dipisahkan dari tren suku bunga.
- Ambang operasional F2 dipilih berdasarkan bobot recall:precision 2:1, sementara struktur biaya riil bank sekitar 3.333:1 — sehingga ambang akhir yang direkomendasikan diambil dari analisis Euro, bukan dari F2.

---

*Dataset: UCI Bank Marketing (Portuguese bank telemarketing campaign), 41.172 baris pasca-cleaning, 21 fitur. Referensi: Moro et al. (2014), Nechita et al. (2024).*

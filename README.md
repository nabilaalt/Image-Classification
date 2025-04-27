
# Flowers Classification

Proyek ini bertujuan untuk membangun model klasifikasi gambar bunga menggunakan **Transfer Learning** dengan model **MobileNetV2**. Model dikembangkan menggunakan **TensorFlow** dan **Keras** untuk mengklasifikasikan gambar ke dalam 14 kategori bunga yang berbeda.

## Fitur Utama
- Klasifikasi gambar bunga ke dalam 14 kategori yang telah ditentukan.
- Implementasi model CNN menggunakan **Transfer Learning** dengan **MobileNetV2 pretrained**.
- Fine-tuning pada 100 layer terakhir dari MobileNetV2.
- Training dengan teknik EarlyStopping dan ReduceLROnPlateau untuk mengoptimalkan performa model.
- Evaluasi model menggunakan metrik **accuracy**.

---

## Persyaratan
Pastikan kamu sudah menginstall semua dependensi sebelum menjalankan proyek ini:

```bash
pip install -r requirements.txt
```

---

## Langkah-Langkah Menjalankan Proyek

### 1. Persiapkan Dataset
- Dataset bunga yang terdiri dari 14 kelas dapat diunduh dari Kaggle atau sumber dataset bunga lainnya.
- Pastikan struktur folder dataset sesuai untuk ImageDataGenerator (`train/val/test` dengan masing-masing subfolder per kelas).

### 2. Pelatihan Model
- Model dilatih menggunakan **transfer learning** dengan MobileNetV2.
- Awalnya semua layer MobileNetV2 difreeze, kemudian 100 layer terakhir diunfreeze untuk fine-tuning.
- Struktur tambahan: 2 Conv2D, GlobalAveragePooling, Dense Layers, Dropout, dan BatchNormalization untuk meningkatkan performa.

```python
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
```

- Optimizer: Adam
- Initial Learning Rate: 5e-5
- Fine-tuning Learning Rate: 1e-5
- Jumlah Epoch: 30 epoch awal + 10 epoch fine-tuning.

### 3. Evaluasi Model
Model dievaluasi menggunakan metrik **accuracy**, dengan hasil sebagai berikut:

| Dataset | Accuracy |
|:--------|:---------|
| Train   | ~99%     |
| Validation | ~98 |
| Test    | ~97%     |

### 4. Menyimpan dan Memuat Model
Model disimpan dalam beberapa format seperti `Saved Model`,`TF-Lite`, dan `TFJS` .  
Model ini dapat dimuat kembali untuk keperluan prediksi baru atau untuk melanjutkan training.

---

## Penutup
Proyek ini merupakan contoh implementasi **Klasifikasi Gambar Bunga** menggunakan **Transfer Learning** dengan MobileNetV2.  
Akurasinya sudah cukup tinggi, tetapi kamu masih bisa meningkatkan performa dengan:
- Menggunakan dataset yang lebih besar.
- Menerapkan teknik **Data Augmentation** lebih agresif.
- Menggunakan learning rate scheduler lebih lanjut seperti **Cosine Annealing** atau **OneCycle Policy**.

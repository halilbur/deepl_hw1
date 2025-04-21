# Derin Öğrenmeye Giriş - Ödev 1: BCI Sınıflandırması

Bu proje, Derin Öğrenmeye Giriş dersi kapsamında verilen birinci ödevdir. Projenin amacı, Beyin-Bilgisayar Arayüzü (BCI) verilerini kullanarak bir Evrişimli Sinir Ağı (CNN) modeli eğitmektir. Model, BCI sinyallerini dört farklı sınıfa ayırmayı hedefler.

## Proje Yapısı

```
.
├── BCI_dataset/         # Veri setinin bulunduğu klasör (train/test alt klasörleri ile)
├── logs/                # TensorBoard loglarının kaydedildiği klasör
├── outputs/             # Eğitim ve test çıktılarının (modeller, grafikler) kaydedildiği ana klasör
│   └── <run_name>/      # Belirli bir çalıştırmaya ait çıktılar
│       ├── models/      # Eğitilen modellerin (.pth) kaydedildiği yer
│       ├── accuracy_plot.png
│       ├── loss_plot.png
│       ├── train_cm_epoch_*.png
│       ├── val_cm_epoch_*.png
│       └── test_confusion_matrix.png
│       └── prediction_examples.png
├── dataset.py           # Veri setini yükleme ve hazırlama kodları
├── evaluate.py          # Model değerlendirme ve görselleştirme fonksiyonları
├── main.py              # Ana çalıştırma betiği (eğitim ve test modları)
├── model.py             # CNN modelinin (BCICNN) tanımlandığı dosya
├── train.py             # Model eğitim döngüsünü içeren kodlar
├── README.md            # Bu dosya
└── ...                  # Diğer yardımcı dosyalar ve yapılandırmalar
```

## Kurulum

Projeyi çalıştırmadan önce gerekli kütüphanelerin kurulu olduğundan emin olun:

```bash
pip install torch torchvision torchaudio numpy matplotlib seaborn scikit-learn tensorboard
```

## Veri Seti

Proje, `BCI_dataset` klasöründe bulunan bir veri setini kullanır. Veri seti, `train` ve `test` olmak üzere iki alt klasöre ayrılmıştır. Her alt klasörde, sınıf etiketlerine göre düzenlenmiş veri örnekleri bulunur.

## Kullanım

Proje, `main.py` betiği üzerinden çalıştırılır ve iki ana modu destekler: `train` (eğitim) ve `test` (değerlendirme).

### Model Eğitimi

Modeli eğitmek için aşağıdaki komutu kullanabilirsiniz. Gerekli argümanları kendi yapılandırmanıza göre düzenleyin:

```bash
python main.py --mode train --data_dir ./BCI_dataset --batch_size 64 --num_epochs 50 --learning_rate 0.001 --save_dir ./outputs --log_dir ./logs
```

**Argümanlar:**

*   `--mode`: Çalıştırma modu (`train` veya `test`).
*   `--data_dir`: Veri setinin bulunduğu ana klasör yolu.
*   `--batch_size`: Eğitim ve değerlendirme için kullanılacak batch boyutu.
*   `--num_epochs`: Toplam eğitim epoch sayısı.
*   `--learning_rate`: Optimizasyon için başlangıç öğrenme oranı.
*   `--save_dir`: Model çıktılarının (modeller, grafikler vb.) kaydedileceği ana klasör. Her çalıştırma için bu klasör altında `lr_<learning_rate>_bs_<batch_size>_<timestamp>` gibi benzersiz bir alt klasör oluşturulur.
*   `--log_dir`: TensorBoard loglarının kaydedileceği ana klasör. Her çalıştırma için `save_dir` ile aynı isimde bir alt klasör oluşturulur.

Eğitim sırasında, her 10 epoch'ta bir eğitim ve doğrulama veri setleri için karışıklık matrisleri (confusion matrix) oluşturulur ve kaydedilir. Ayrıca, eğitim ve doğrulama kayıp (loss) ve doğruluk (accuracy) grafikleri de eğitim sonunda kaydedilir. En iyi doğrulama kaybına sahip model `best_model.pth` olarak kaydedilir. TensorBoard logları `logs/<run_name>` klasörüne kaydedilir ve eğitim sürecini takip etmek için kullanılabilir:

```bash
tensorboard --logdir ./logs
```

### Model Test Etme

Daha önce eğitilmiş bir modeli test etmek için aşağıdaki komutu kullanın:

```bash
python main.py --mode test --data_dir ./BCI_dataset --batch_size 64 --model_path ./outputs/<run_name>/models/best_model.pth --save_dir ./outputs/<run_name>
```

**Argümanlar:**

*   `--mode`: `test` olarak ayarlanmalıdır.
*   `--data_dir`: Test veri setinin bulunduğu ana klasör yolu.
*   `--batch_size`: Test için kullanılacak batch boyutu.
*   `--model_path`: Yüklenecek eğitilmiş modelin `.pth` dosya yolu.
*   `--save_dir`: Test sonuçlarının (karışıklık matrisi, tahmin görselleştirmeleri) kaydedileceği klasör. Belirtilmezse, modelin bulunduğu klasör kullanılır.

Test modu, belirtilen model ile test veri seti üzerinde değerlendirme yapar, doğruluk ve sınıflandırma raporunu yazdırır. Ayrıca, test veri seti için bir karışıklık matrisi ve bazı tahmin örneklerinin görselleştirmelerini belirtilen `save_dir` altına kaydeder.

## Çıktılar

*   **Modeller:** Eğitim sırasında en iyi performansı gösteren model (`best_model.pth`) `outputs/<run_name>/models/` klasörüne kaydedilir.
*   **Grafikler:**
    *   `loss_plot.png`: Eğitim ve doğrulama kayıp grafiği.
    *   `accuracy_plot.png`: Eğitim ve doğrulama doğruluk grafiği.
    *   `train_cm_epoch_*.png`: Belirli epoch'lardaki eğitim karışıklık matrisi.
    *   `val_cm_epoch_*.png`: Belirli epoch'lardaki doğrulama karışıklık matrisi.
    *   `test_confusion_matrix.png`: Test veri seti için karışıklık matrisi (test modunda oluşturulur).
    *   `prediction_examples.png`: Test veri setinden rastgele örnekler ve model tahminleri (test modunda oluşturulur).
*   **Loglar:** TensorBoard logları `logs/<run_name>` klasörüne kaydedilir.
*   **Sonuçlar:** Test doğruluğu ve sınıflandırma raporu konsola yazdırılır.

## Model Mimarisi

Projede `model.py` dosyasında tanımlanan `BCICNN` adında bir Evrişimli Sinir Ağı modeli kullanılmaktadır. Modelin detayları için ilgili dosyaya bakılabilir.

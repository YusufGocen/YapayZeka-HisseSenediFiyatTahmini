
BIST 30 Hisseleri İçin Yapay Zeka ile Fiyat Tahmin Sistemi

1. Proje Tanımı : 

Bu proje, BIST 30 da yer alan hisselerin geçmiş kapanış fiyatlarını kullanarak yapay zeka 
ve makine öğrenmesi yöntemleri ile kısa vadeli fiyat tahmini yapmayı amaçlamaktadır.

Projede kullanıcı, BIST 30 hisseleri arasından bir hisse seçmekte ve
seçilen hisse için farklı regresyon tabanlı yapay zeka modellerinin performansları karşılaştırılmaktadır. 
En başarılı model belirlenerek bir sonraki gün için kapanış fiyatı tahmini yapılmaktadır.

- Geçmiş test sonuçları analiz edilir

- Son 5 günlük model performansı hesaplanır

- Bir sonraki işlem günü için kapanış fiyatı tahmini yapılır

2. Kullanılan Yöntemler ve Modeller : 

Projede aşağıdaki yapay zeka / makine öğrenmesi modelleri kullanılmıştır:

Doğrusal Regresyon

Bayesçi Doğrusal Regresyon

Karar Ağacı Regresyonu

Gradient Boosting Regresyonu

Yapay Sinir Ağı (Neural Network)

3. Klasör Açıklamaları

main.py: Projenin ana çalışma dosyası

app.py : Streamlit arayüz dosyası

data: Hisse verilerinin çekildiği dosya

models: Yapay zeka modellerinin tanımlandığı dosya

analysis: Model performanslarının hesaplandığı dosya

forecast: Fiyat tahmin işlemlerinin yapıldığı dosya

visual: Grafiklerin oluşturulduğu dosya

4. Veri Kaynağı:

Proje kapsamında hisse senedi verileri Yahoo Finance platformundan yfinance kütüphanesi aracılığıyla çekilmektedir.

Veriler; 2018 yılından günümüze kadar olan kapanış fiyatlarını içermektedir.

5. Projenin Çalıştırılması : 

Gerekli Kütüphaneler
- pip install yfinance numpy pandas matplotlib scikit-learn

Terminal Üzerinden Çalıştırma : 
- python main.py

Streamlit Arayüzü ile Çalıştırma : 
- streamlit run app.py

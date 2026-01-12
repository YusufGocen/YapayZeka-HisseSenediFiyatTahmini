<h1>BIST 30 Hisseleri İçin Yapay Zeka ile Fiyat Tahmin Sistemi</h1>
<p>
  <img width="1459" height="706" alt="Ekran Resmi 2026-12-29 14 49 47" src="https://github.com/user-attachments/assets/f217966c-3f99-424d-becc-c8bb7fddb853" />
</p>
<br>
<p>
  <img width="1459" height="706" alt="Ekran Resmi 2026-12-29 14 44 41" src="https://github.com/user-attachments/assets/7af18bdc-f6de-4efb-99c3-0c117c1fe78f" />
</p>
<br>
<p>
  <img width="1459" height="668" alt="Ekran Resmi 2026-12-29 14 45 12" src="https://github.com/user-attachments/assets/6b21a648-645f-4468-9985-ca5a84b83b79" />
</p>
 
<br>
<br>

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

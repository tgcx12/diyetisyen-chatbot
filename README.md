
# 🥗 Diyetisyen LLM


## RAG + NLP + Deterministik Planlama ve LLM Kalibrasyon Katmanlı Akıllı Sağlık Asistanı

📄 **Akademik Rapor (Detaylı Metodoloji ve Deney Sonuçları):**
👉 [221307036_Diyetisyenlik_LlM_Rapor (4).pdf](./221307036_Diyetisyenlik_LlM_Rapor%20%284%29.pdf)

Bu repository, yukarıdaki bitirme projesi raporunda ayrıntılı olarak açıklanan sistemin uygulama kodlarını ve deneysel çıktılarını içermektedir.

---

# 📌 Proje Amacı

Bu çalışmanın amacı:

* Beslenme rehberlerine dayalı,
* Klinik olarak daha güvenli,
* Hallucination riski azaltılmış,
* Kişiye özel diyet planı üretebilen,
* Sağlık sorularına bağlam temelli yanıt verebilen

hibrit bir yapay zeka sistemi geliştirmektir.

Sağlık alanında yalnızca büyük dil modeli (LLM) kullanmak risklidir.
Bu nedenle sistem, **LLM-merkezli değil, kural-merkezli** bir mimari ile tasarlanmıştır.

---

# ❓ Neden Hibrit Bir Mimari?

## 🔴 Problem 1: LLM-Only Güvenilir Değil

Rapor kapsamında yapılan LLM-only deneylerinde:

* Exact Match ve F1 skorları düşük kalmıştır.
* Must-Have bilgilerin atlandığı örnekler görülmüştür.
* Hallucination (bağlam dışı bilgi üretimi) gözlemlenmiştir.

Bu durum, sağlık gibi kural yoğun bir alanda LLM’in tek başına yeterli olmadığını göstermektedir.

---

## 🔴 Problem 2: Sağlık Alanında Sayısal Kısıtlar Kritik

Rehberlerde sıkça geçen ifadeler:

* “Haftada 2–3 kez”
* “Günde en fazla 5 g”
* “%20–35 yağ oranı”

LLM bu tür kısıtları:

* Atlayabilir
* Yanlış genelleyebilir
* Tutarsız uygulayabilir

Bu nedenle sayısal kuralların deterministik olarak uygulanması gerekmektedir.

---

# 🧠 Genel Sistem Mimarisi

Sistem iki ana modülden oluşur:

```
MODÜL 1 — Sağlık Soru-Cevap (RAG + LLM)

MODÜL 2 — Diyet Planlama
    ├─ QA Summary (RAG + LLM)
    ├─ NLP ile Yapılandırılmış Kural Çıkarımı (LLM-Free)
    ├─ Deterministik Planlama Motoru
    └─ LLM Kalori Kalibrasyonu (Audit & Scaling)
```

---

# 🔹 MODÜL 1 — Sağlık Soru-Cevap Sistemi (RAG + LLM)

## 🎯 Amaç

Kullanıcıların beslenme ve sağlık alanındaki doğal dil sorularına:

* Doğru,
* Rehber temelli,
* Bağlama dayalı,
* Hallucination riski azaltılmış

yanıtlar üretmek.

---

## ⚙️ Çalışma Prensibi

1️⃣ Kullanıcı sorusu alınır.
2️⃣ Soru embedding’e dönüştürülür.
3️⃣ ChromaDB üzerinde vektör tabanlı arama yapılır.
4️⃣ En ilgili belgeler (top-k) geri getirilir.
5️⃣ Bu belgeler LLM’e bağlam olarak verilir.
6️⃣ LLM yalnızca bu bağlam çerçevesinde cevap üretir.

Bu yapı sayesinde:

* Model bağlam dışına çıkamaz.
* Uydurma bilgi üretme olasılığı azaltılır.
* Cevaplar rehber metinlerine dayandırılır.

---

## 📊 MODÜL 1 — Deneysel Değerlendirme

### Test Kümesi

* 35 soru
* Konular: kolesterol, yağ türleri, besin grupları, kalp-damar sağlığı

---

## 🔹 1️⃣ Retriever-Only (Sadece Bilgi Getirme)

| k  | Hit Rate@k | Recall@k |
| -- | ---------- | -------- |
| 3  | 0.886      | 0.686    |
| 5  | 0.971      | 0.757    |
| 10 | 0.971      | 0.814    |

**Sonuç:**

* k arttıkça Recall artmıştır.
* k=5 ve k=10 için %97 Hit Rate elde edilmiştir.
* Retriever bileşeni tek başına güçlü kapsama sağlamaktadır.

---

## 🔹 2️⃣ LLM-Only

Test edilen modeller:

* Gemma-2:2B
* Qwen-2.5:3B
* Gemma-3:4B
* Llama-3.1:8B

Gözlemler:

* En yüksek F1 ≈ %33
* Bazı sorularda F1 = 0
* Must-Have Recall düşüktür
* Hallucination gözlemlenmiştir

**Sonuç:**

Bağlam olmadan LLM klinik doğruluk açısından yetersizdir.

---

## 🔹 3️⃣ LLM + RAG

K = 3, 5 ve 10 için değerlendirme yapılmıştır.

K=10 için:

* En yüksek EM
* En düşük Hallucination
* En yüksek Supported Ratio

**Genel Sonuç:**

En dengeli ve güvenilir performans LLM + RAG yaklaşımı ile elde edilmiştir.

---

# 🔹 MODÜL 2 — Kişiye Özel Diyet Planlama

Bu modül doğrudan LLM tarafından yönetilmez.
Plan üretimi deterministik bir çekirdek tarafından gerçekleştirilir.

---

## 1️⃣ QA Summary — Hastalık Bazlı Kural Üretimi

Kullanıcı bir hastalık belirttiğinde (ör: kolesterol):

Sistem rehberlerden aşağıdaki gibi sorular üretir:

* Et tüketimi nasıl olmalı?
* Yumurta haftada kaç kez?
* Doymuş yağ oranı?
* Balık tüketim sıklığı?
* Lif miktarı?
* Tuz sınırı?

Bu sorulara verilen cevaplar tek satırlık, kontrollü formatta üretilir:

```
Haftada 2–3 kez balık tüketilmeli | Doymuş yağ sınırlandırılmalı | ...
```

---

## 🤖 QA Summary İçin Denenen LLM Modelleri

* Gemma-2:2B
* Qwen-2.5:3B
* Gemma-3:4B
* Llama-3.1:8B
* Mistral-7B

Manuel ve metrik değerlendirme sonucunda:

👉 **Gemma-3:4B temel model olarak seçilmiştir.**

Gerekçe:

* Rehber metnine en sadık üretim
* Düşük hallucination
* Klinik olarak daha uygulanabilir çıktı

---

## 2️⃣ NLP ile Yapılandırılmış Kural Çıkarımı (LLM-Free)

QA Summary çıktısı doğrudan kullanılmaz.

Python tabanlı NLP pipeline:

* Clause ayrıştırma
* Tag eşleme (alias + regex)
* Intent sınıflandırma (prefer / limit / avoid)
* Negation detection
* “X yerine Y” analizi
* Sayısal kısıt çıkarımı
* Validation & conflict resolution

Bu aşamada LLM kullanılmaz.

Amaç:

* Deterministik yapı
* Model bağımsızlık
* Tekrar üretilebilirlik

---

## 3️⃣ Deterministik Planlama Motoru

Plan üretimi:

* labels.json referanslı
* Heuristik skorlamalı
* Haftalık sayısal kısıtları zorlayarak uygular
* Yapısal kuralları enforce eder

### Yapısal Kurallar

* Sabah kahvaltı yapısı
* Öğle/akşam tek ana yemek
* Balık + süt aynı öğünde olmaz
* Ara öğün boş kalmaz

---

## 4️⃣ Enerji Hesabı

Kullanıcıdan:

* Yaş
* Cinsiyet
* Boy
* Kilo
* Aktivite seviyesi

alınır.

Hesaplanan değerler:

* BMR (Mifflin–St Jeor)
* TDEE
* Hedef kalori

---

## 5️⃣ LLM Kalori Kalibrasyon Katmanı

Ana plan deterministik olarak üretilir.

Ancak:

Eğer üretilen plan hedef kaloriden saparsa
(örneğin 1500 kcal üretildi ama hedef 1800 kcal ise)

LLM devreye girer.

LLM:

* Porsiyon artırır/azaltır
* Ara öğün ekler
* Kalori farkını kapatır

Ancak:

* Must-not kurallarını ihlal edemez
* Hastalık kısıtlarını silemez
* Kural setini değiştiremez

Bu katman enerji optimizasyonu sağlar.

---

# 📊 Planlama Benchmark — Kolesterol Senaryosu

Tüm modeller:

* Aynı besin havuzu
* Aynı labels.json
* Aynı planlama algoritması

ile test edilmiştir.

Sonuçlar:

* Ortalama kalori sapması ≈ 273 kcal
* Hedef kalorinin ±%10 aralığında kalma oranı ≈ %28.57
* Kritik kombinasyon kuralları korunmuştur
* Model değişse de plan metrikleri büyük ölçüde aynıdır

**Yorum:**

Plan kalitesi modelden bağımsızdır.
Plan başarımı deterministik motor ve kural setine bağlıdır.

---

# 🎯 Genel Değerlendirme

Bu sistem:

* LLM’i doğrudan karar verici yapmaz.
* RAG ile güvenli bağlam sağlar.
* NLP ile kuralları yapılandırır.
* Deterministik plan üretir.
* LLM’i denetleyici ve kalibrasyon katmanı olarak kullanır.

Bu mimari sağlık alanında:

* Daha güvenli
* Daha açıklanabilir
* Daha ölçülebilir
* Daha denetlenebilir

bir yaklaşım sunmaktadır.

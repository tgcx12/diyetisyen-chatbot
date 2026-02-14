import gradio as gr
import time
import re # Veri ayıklama için

# Kendi dosyalarını import ediyorsun (dosya isimlerine göre güncelleyebilirsin)
import diet_rules  # Adım 1: Hastalık kısıtlarını getirir
import labels      # Adım 2: Model seçimi ve kural denetimi
import diet_ai     # Adım 3: Diyet üretimi ve scaling

class UserSession:
    def __init__(self):
        self.data = {
            "ad_soyad": None,
            "yas": None,
            "boy": None,
            "kilo": None,
            "hastalik": None,
            "aktivite": None,
            "plan_hazir_mi": False
        }

    def eksik_bilgi_bul(self):
        for key in ["ad_soyad", "yas", "boy", "kilo", "hastalik", "aktivite"]:
            if self.data[key] is None:
                return key
        return None

session = UserSession()

def extract_entities(text):
    """Kullanıcının yazdığı cümleden sayıları ve bilgileri ayıklar"""
    text_lower = text.lower()
    
    # Boy ayıklama (Genelde 150-210 arası)
    boy_match = re.search(r'(boy|boyum)\s*(\d{3})', text_lower)
    if boy_match: session.data["boy"] = int(boy_match.group(2))
    
    # Kilo ayıklama (Genelde 40-200 arası)
    kilo_match = re.search(r'(kilo|kilom|kiloyum)\s*(\d{2,3})', text_lower)
    if kilo_match: session.data["kilo"] = int(kilo_match.group(2))

    # Yaş ayıklama
    yas_match = re.search(r'(\d{2})\s*(yaşında|yaşındayım)', text_lower)
    if yas_match: session.data["yas"] = int(yas_match.group(1))

def diet_bot_response(message, history):
    global session
    
    # 0. Kullanıcıdan gelen veriyi otomatik tara
    extract_entities(message)
    
    # Manuel veri girişi kontrolü (Eğer spesifik bir soruya cevap veriliyorsa)
    eksik_su_an = session.eksik_bilgi_bul()
    if eksik_su_an == "ad_soyad" and len(message.split()) <= 3:
        session.data["ad_soyad"] = message
    elif eksik_su_an == "hastalik" and ("yok" in message.lower() or "var" in message.lower() or "hastayım" in message.lower()):
        session.data["hastalik"] = message
    elif eksik_su_an == "aktivite":
        session.data["aktivite"] = message

    # 1. Eksik Bilgi Kontrolü
    eksik = session.eksik_bilgi_bul()
    
    if eksik:
        sorular = {
            "ad_soyad": "Merhaba! Ben diyetisyen asistanınız. Size özel bir plan için adınızı öğrenebilir miyim?",
            "yas": "Kaç yaşındasınız?",
            "boy": "Boyunuz kaç cm?",
            "kilo": "Güncel kilonuz nedir?",
            "hastalik": "Herhangi bir hastalığınız (Kolesterol, Şeker vb.) veya kısıtlamanız var mı?",
            "aktivite": "Günlük aktivite düzeyiniz nedir? (Hareketsiz, Orta, Çok Hareketli)"
        }
        return sorular[eksik]
    
    # 2. Tüm Bilgiler Tamamsa Çalışma Sırasını Başlat
    if not session.data["plan_hazir_mi"]:
        yield "Bilgilerinizi aldım. Şimdi süreci başlatıyorum..."
        time.sleep(1)
        
        # --- ADIM 1: DIET_RULES ---
        yield "✅ Adım 1: Hastalık kısıtları ve beslenme kuralları belirleniyor (diet_rules)..."
        # rules = diet_rules.get_rules(session.data["hastalik"])
        time.sleep(1.5)
        
        # --- ADIM 2: LABELS ---
        yield "✅ Adım 2: Model denetimi ve etiketleme yapılıyor (labels)..."
        # model_checks = labels.check_constraints(rules)
        time.sleep(1.5)
        
        # --- ADIM 3: DIET_AI ---
        yield "✅ Adım 3: Yapay zeka diyet listesini ve PDF dosyasını oluşturuyor (diet_ai)..."
        # final_diet = diet_ai.generate_diet(session.data, model_checks)
        time.sleep(2)
        
        session.data["plan_hazir_mi"] = True
        
        output = f"### Sayın {session.data['ad_soyad']}, diyetiniz hazır!\n\n"
        output += "1800 Kalorilik Kolesterol Odaklı Planınız hazırlanmıştır.\n\n"
        output += "**[📄 Diyeti PDF Olarak İndir](#)** (Buraya PDF linki gelecek)"
        
        yield output
    else:
        return "Diyetiniz üzerinde bir değişiklik isterseniz bana söyleyebilirsiniz."

# Gradio Arayüzü
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🥗 AktarKuş - Akıllı Diyetisyen Sistemi")
    gr.Markdown("Bilgilerinizi girerek kişiselleştirilmiş, hastalık duyarlı diyet planınızı oluşturun.")
    
    chatbot = gr.ChatInterface(
        fn=diet_bot_response,
        examples=[
            "Diyet yapmak istiyorum, kolesterolüm var", 
            "Boyum 180, kilom 90, 30 yaşındayım, diyet listesi istiyorum"
        ],
        title="Diyet Asistanı"
    )

if __name__ == "__main__":
    demo.launch()
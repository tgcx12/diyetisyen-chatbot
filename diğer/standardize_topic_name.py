import json
import re

def standardize_topic_name(name):
    if not name or name == "None":
        return "genel" # Boş olanları 'genel' olarak işaretler
    
    # Türkçe karakterleri İngilizce karşılıklarına çevir (isteğe bağlı)
    char_map = {
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'Ç': 'c', 'Ğ': 'g', 'İ': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u'
    }
    for tr, en in char_map.items():
        name = name.replace(tr, en)
    
    # Küçük harfe çevir
    name = name.lower()
    # Boşlukları ve 've', '&' gibi bağlaçları alt tireye çevir
    name = name.replace(" ve ", "_")
    name = name.replace(" & ", "_")
    name = re.sub(r'\s+', '_', name) # Tüm boşlukları alt tire yap
    # Sadece alfanumerik ve alt tire karakterlerini tut (özel karakterleri sil)
    name = re.sub(r'[^a-z0-9_]', '', name)
    # Çift alt tireleri tek yap
    name = re.sub(r'_+', '_', name).strip('_')
    
    return name

# 1. Dosyayı Yükle
file_path = 'merged_all_rag_new.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Veriyi Düzenle
for item in data:
    old_topic = item.get("topic_group", "")
    new_topic = standardize_topic_name(old_topic)
    item["topic_group"] = new_topic

# 3. Güncellenmiş Dosyayı Kaydet
output_path = 'merged_all_rag_standardized.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"İşlem tamamlandı! Yeni dosya: {output_path}")
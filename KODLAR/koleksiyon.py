import chromadb

# Sildiğiniz klasör yolunu buraya yazın
client = chromadb.PersistentClient(path=r"C:\Users\user\Desktop\diyetisyen_llm\chroma_db_storage")

# Koleksiyonu oluşturun
collection = client.get_or_create_collection(name="diyetisyen_rehberi")

# Örnek: Elinizdeki metinleri buraya liste olarak ekleyin
# (Gerçek kullanımda dosyalarınızı döngüyle okumalısınız)
documents = ["Kolesterol diyeti şöyle olmalıdır...", "Yumurta haftada 2 kez yenmelidir..."]
ids = ["id1", "id2"]

collection.add(documents=documents, ids=ids)
print("Veritabanı başarıyla yeniden oluşturuldu.")
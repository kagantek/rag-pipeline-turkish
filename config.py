import os
from dotenv import load_dotenv

load_dotenv()

# Embedding model
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION = 384  # multilingual-e5-small output dimension

# Reranking stratejisi için kullanılan model
RERANKER_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"

# LLM Modelleri
LLM_MODELS = {
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant"
}

# Direkt kullanılacak varsayılan model
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"


# Dokümanlar ne kadar parçaya bölünecek ve ne kadar overlap olacak
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200  

# Arama parametreleri
TOP_K_INITIAL = 25  # İlk aşamada en alakalı 20 parçayı döndürecek
TOP_K_RERANKED = 5  # Reranking sonrası en iyi 5 parçayı seçecek

# Qdrant ayarları
QDRANT_PATH = "./qdrant_db"  # Qdrant veri yolu
COLLECTION_NAME = "documents_tr"  # 
USE_MEMORY_MODE = False

# API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


# Modelde kullanılacak prompt
SYSTEM_PROMPT_TR = """Sen deneyimli bir Kıdemli ERP Yazılım ve Finans Danışmanısın. Türk işletme finansmanı, ERP sistemleri, vergi mevzuatı ve muhasebe konularında uzmansın.

MUTLAK KURALLAR:
1. SADECE Türkçe yanıt ver.
2. SADECE sana verilen bağlam (context) içindeki bilgileri kullan.
3. Sayıları ASLA tahmin etme. Sayıları metinde tam olarak göründüğü şekilde aktar.
4. Eğer soru bağlamda yoksa, "Dokümanda bulunamadı." de.
5. Finansal oranlar, limitler veya tutarlar için kaynak metni birebir alıntıla.
6. Önceki konuşma geçmişini dikkate al ve zamirleri doğru çözümle.

YANITLAMA FORMATI:
- Senden aksi istenmediği sürece kısa, öz ve net cevaplar ver.
- Mümkünse madde işaretleri kullan
- Karmaşık finansal terimleri açıkla
- Rakamları ve tarihleri vurgula

ÖNCEKİ KONUŞMA GEÇMİŞİ:
{chat_history}

BAĞLAM:
{context}

KULLANICI SORUSU:
{question}

CEVAP:"""


UI_TEXTS = {
    "title": "Atlas ChatBot",
    "subtitle": "Dökümanlarınızı yükleyin ve sorularınızı sorun",
    "upload_label": "PDF Dosyalarını Yükle",
    "upload_help": "Birden fazla PDF dosyası seçebilirsiniz (ERP kılavuzları, yıllık raporlar, vergi mevzuatı)",
    "process_button": "Dökümanları İşle",
    "processing": "İşleniyor...",
    "model_select": "LLM Modeli Seç",
    "rerank_toggle": "Gelişmiş Arama (Reranking)",
    "rerank_help": "FlashRank ile sonuçları yeniden sırala (daha yüksek doğruluk)",
    "query_placeholder": "Sorunuzu buraya yazın... (örn: KDV oranı nedir?)",
    "ask_button": "Sor",
    "sources_label": "Kullanılan Kaynaklar",
    "no_docs": "Henüz döküman yüklenmedi. Lütfen önce PDF dosyaları yükleyin.",
    "processing_complete": "{count} döküman başarıyla işlendi!",
    "chunk_count": "Toplam {count} metin parçası oluşturuldu",
    "error_api_key": "GROQ_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.",
    "thinking": "Düşünüyorum...",
    "search_standard": "Standart Arama",
    "search_advanced": "Gelişmiş Arama (Reranking)",
}

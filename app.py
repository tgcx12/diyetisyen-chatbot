import os
import requests
import gradio as gr
import webbrowser
from typing import List, Tuple

# =========================
# CONFIG
# =========================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "700"))

HOST = os.environ.get("GRADIO_HOST", "127.0.0.1")  # <- ÖNEMLİ: 0.0.0.0 değil
PORT = int(os.environ.get("GRADIO_PORT", "7860"))

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """Sen deneyimli bir diyetisyen asistanısın.
Kullanıcıya Türkçe, anlaşılır, pratik ve güvenli öneriler ver.
Önceliklerin:
1) Güvenlik: Tehlikeli/uygunsuz öneri verme.
2) Kişiselleştirme: Yaş, boy, kilo, cinsiyet, aktivite, hedef, hastalıklar, ilaçlar, alerjiler, hamilelik/emzirme gibi bilgileri gerekirse nazikçe sor.
3) Netlik: Maddeler halinde, ölçülü porsiyon önerileriyle, uygulanabilir plan sun.

Tıbbi hassasiyet kuralları:
- Kullanıcı ciddi semptomlar (göğüs ağrısı, bayılma, nefes darlığı, kanlı kusma/dışkı, şiddetli dehidratasyon, bilinç değişikliği vb.) bildirirse acile başvurmasını öner.
- Diyabet, böbrek yetmezliği, karaciğer hastalığı, gut, yeme bozukluğu öyküsü, hamilelik/emzirme, 18 yaş altı gibi durumlarda “genel bilgi” ver; tedavi/ilaç dozuna girme; doktora/diyetisyene yönlendir.
- Aşırı kısıtlayıcı, çok düşük kalorili, hızlı kilo verdiren, “detoks” gibi iddialı ve kanıtsız önerilerden kaçın.
- Kalori/makro hesabı istenirse kabaca tahmini aralıklar ver ve bunun klinik değerlendirme olmadığını belirt.

Yanıt biçimi:
- Kısa bir özet + ardından maddeli öneriler.
- Gerekirse 3-5 kısa soru sorarak bilgileri tamamla.
- Kullanıcı “plan” isterse: 1 günlük örnek menü + alternatifler + alışveriş/uygulama ipuçları ver.
"""

# =========================
# OLLAMA CHAT STREAM
# =========================
def ollama_chat_stream(messages: List[dict], temperature: float, num_predict: int):
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
        },
        "stream": True,
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            full_text = ""
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = requests.utils.json.loads(line)
                except Exception:
                    continue
                if obj.get("done"):
                    break
                chunk = obj.get("message", {}).get("content", "")
                if chunk:
                    full_text += chunk
                    yield full_text
    except requests.exceptions.ConnectionError:
        yield "❌ Ollama'ya bağlanamadım. Ollama açık mı? (Varsayılan: http://localhost:11434)"
    except requests.exceptions.HTTPError as e:
        yield f"❌ Ollama HTTP hatası: {e}"
    except Exception as e:
        yield f"❌ Beklenmeyen hata: {e}"

# =========================
# CHAT LOGIC
# =========================
def build_messages_from_history(history: List[Tuple[str, str]], user_message: str) -> List[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history:
        if u:
            msgs.append({"role": "user", "content": u})
        if a:
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_message})
    return msgs

def respond(user_message: str, history: List[Tuple[str, str]], temperature: float, max_tokens: int):
    user_message = (user_message or "").strip()
    if not user_message:
        yield history
        return

    messages = build_messages_from_history(history, user_message)
    history = history + [(user_message, "")]

    for partial in ollama_chat_stream(messages, temperature, max_tokens):
        history[-1] = (user_message, partial)
        yield history

def reset_chat():
    return []

# =========================
# UI
# =========================
CSS = """
:root{
  --bg:#0b1220;
  --card:#0f1a30;
  --muted:#91a4c7;
  --accent:#4fd1c5;
}
body{background:var(--bg)!important;}
.gradio-container{max-width: 980px !important;}
#title h1{letter-spacing:0.2px}
#subtitle{color:var(--muted); margin-top:-6px}
#card{
  background:linear-gradient(180deg, rgba(79,209,197,0.12), rgba(79,209,197,0.02));
  border:1px solid rgba(79,209,197,0.25);
  border-radius:18px;
  padding:14px 16px;
}
"""

DESCRIPTION = """
<div id="card">
<b>🥗 Diyet Asistanı (LLaMA 3.1:8B)</b><br/>
<span style="color:#91a4c7">
Sorunu yaz; sana güvenli, uygulanabilir beslenme önerileriyle yanıt vereyim.
İstersen hedefini (kilo verme/kilo alma/performans), yaş-boy-kilo ve varsa hastalık/ilaç bilgini de ekle.
</span>
</div>
"""

EXAMPLES = [
    "Kilo vermek istiyorum. 28 yaş, 168 cm, 78 kg. Ofis işiyim. Nereden başlamalıyım?",
    "İnsülin direncim var. Kahvaltıda ne yemeliyim? Pratik öneri verir misin?",
    "Spora yeni başladım. Kas yapmak için günlük beslenmem nasıl olmalı?",
    "Akşam çok acıkıyorum, gece atıştırmalarını nasıl bırakırım?",
]

with gr.Blocks() as demo:
    gr.Markdown("<div id='title'><h1>🥗 Diyetisyen Chat</h1></div>")
    gr.Markdown("<div id='subtitle'>LLaMA 3.1:8B ile sohbet — doğal dilde beslenme danışmanı</div>")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        temperature = gr.Slider(0.0, 1.0, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature (yaratıcılık)")
        max_tokens = gr.Slider(128, 1500, value=DEFAULT_MAX_TOKENS, step=32, label="Max tokens (cevap uzunluğu)")

    chat = gr.Chatbot(label="Sohbet", height=520)

    with gr.Row():
        msg = gr.Textbox(label="Sorunu yaz", placeholder="Örn: 1 haftalık pratik kilo verme planı yapar mısın?", scale=10)
        send = gr.Button("Gönder", variant="primary", scale=2)

    with gr.Row():
        clear = gr.Button("Sohbeti Sıfırla")
        gr.Markdown("<span style='color:#91a4c7'>Not: Bu bir tıbbi teşhis aracı değildir. Acil durumda 112/ACİL.</span>")

    gr.Examples(EXAMPLES, inputs=msg)

    state = gr.State([])

    def on_send(user_message, history, t, mt):
        return respond(user_message, history, t, mt)

    send.click(on_send, inputs=[msg, state, temperature, max_tokens], outputs=[chat], queue=True)
    msg.submit(on_send, inputs=[msg, state, temperature, max_tokens], outputs=[chat], queue=True)

    chat.change(lambda h: h, inputs=[chat], outputs=[state], queue=False)

    send.click(lambda: "", outputs=[msg], queue=False)
    msg.submit(lambda: "", outputs=[msg], queue=False)

    clear.click(reset_chat, outputs=[chat], queue=False)
    clear.click(reset_chat, outputs=[state], queue=False)

if __name__ == "__main__":
    url = f"http://{HOST}:{PORT}"
    print("\n" + "=" * 60)
    print("✅ Diyetisyen Chat calisiyor!")
    print(f"👉 Tarayicida ac: {url}")
    print("=" * 60 + "\n")

    # Tarayıcıyı otomatik aç (Windows’ta çalışır)
    try:
        webbrowser.open(url)
    except Exception:
        pass

    demo.queue(default_concurrency_limit=8).launch(
        server_name=HOST,
        server_port=PORT,
        theme=gr.themes.Soft(),
        css=CSS,
        prevent_thread_lock=False,  # terminal açık kalsın
    )

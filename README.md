# KuantLab

Açık kaynak dil modellerini GGUF formatına kuantize eden yerel web aracı.

## Çalıştırma

```bash
chmod +x calistir.sh
./calistir.sh
```

Tarayıcıda: [http://localhost:8080](http://localhost:8080)

İlk çalıştırmada sanal ortam kurulur, bağımlılıklar yüklenir ve `llama.cpp` hazırlanır.

## Gereksinimler

- Python 3.10+
- İnternet (ilk kurulum ve Hugging Face model indirme)

## Yapı

```
calistir.sh
requirements.txt
app/
  server.py
  quantizer.py
  setup_llama.py
  static/           # çalışan arayüz + css/fonts
colab/
  KuantLab_Colab.ipynb
```

## Not

Aktarımda verilen yeni arayüz HTML'i (`js/` ve `img/` dosyaları olmadan) `app/static/_redesign/` altında tutuldu. Canlı sayfa, tam çalışan zip sürümüdür (`app/static/index.html`).


# 🍒 IA para Análise de maturação de Lichias

Sistema de **Inteligência Artificial** capaz de analisar imagens de lichias e classificar seu estado de maturação, integrado a uma **API em FastAPI** e persistência de dados no **MongoDB Atlas**.

---

## 🚀 Funcionalidades

* 📷 Upload de imagem via API
* 🧠 Classificação com modelo de IA (TensorFlow/Keras)
* 📊 Retorno da classe prevista + confiança
* ☁️ Armazenamento dos resultados no MongoDB Atlas
* 🔗 API REST para integração com web/mobile

---

## 🧠 Tecnologias utilizadas

* **Python 3.11**
* **FastAPI**
* **TensorFlow / Keras**
* **NumPy**
* **Pillow (PIL)**
* **MongoDB Atlas**
* **PyMongo**
* **Uvicorn**

---

## 📂 Estrutura do projeto

```text
IA_Analise_de_Lichia/
│
├── api.py              # API principal (FastAPI)
├── app.py              # Execução/integração
├── requisitos.txt      # Dependências
├── estilos.css         # Estilo frontend (se aplicável)
├── .gitignore
│
└── ia_lichia/
    └── train.py        # Treinamento do modelo
```

---

## ⚙️ Como executar o projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/jk-ramos/IA_Analise_de_Lichia.git
cd IA_Analise_de_Lichia
```

### 2. Criar ambiente virtual

```bash
python -m venv .venv

# Linux/Mac
source .venv/bin/activate  

# Windows
.venv\Scripts\activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar MongoDB Atlas

No código, configure sua string de conexão:

```python
MongoClient("mongodb+srv://USUARIO:SENHA@cluster.mongodb.net/")
```

> ⚠️ **Importante:** Nunca exponha sua senha em produção. Use variáveis de ambiente (`.env`).

### 5. Rodar a API

```bash
uvicorn api:app --reload
```

---

## 📡 Endpoint principal

### 🔍 `POST /predict`

Envia uma imagem para análise.

**Exemplo via cURL:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@imagem.jpg"
```

### 📥 Resposta

```json
{
  "classe_prevista": "madura",
  "confianca": 96.7,
  "nome_arquivo": "imagem.jpg",
  "mensagem": "Análise salva no MongoDB Atlas com sucesso"
}
```

---

## 🗄️ Banco de Dados (MongoDB)

Os dados são armazenados em: `ia_lichia_db` -> `analises`

**Exemplo de documento:**

```json
{
  "nome_arquivo": "imagem.jpg",
  "classe_prevista": "madura",
  "confianca": 96.7,
  "data_analise": "2026-04-23T15:55:34Z"
}
```

---

## ⚠️ Boas práticas

* ❌ **Não subir:** `logs/`, `dataset/`, arquivos `.keras`
* ✔️ Usar `.gitignore`
* ✔️ Usar `.env` para credenciais

---

## 📌 Status do projeto

* ✅ API funcionando
* ✅ Integração com MongoDB Atlas
* ✅ Modelo de IA treinado
* 🔜 Integração com IoT (futuro)

---

## 👩‍💻 Autores

**Equipe Mobo**
FATEC — Desenvolvimento de Software Multiplataforma

---

## 📄 Licença

Projeto acadêmico de uso educacional.

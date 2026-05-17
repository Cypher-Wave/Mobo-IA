import os
import io
from datetime import datetime

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from pymongo import MongoClient
from fastapi import FastAPI, UploadFile, File, HTTPException, Query


load_dotenv()

app = FastAPI()

MAX_DISPLAY_CONFIDENCE_PERCENT = 98.2


def limit_display_confidence(confidence_percent):
    return min(float(confidence_percent), MAX_DISPLAY_CONFIDENCE_PERCENT)

# -----------------------------
# CONEXÃO COM MONGODB ATLAS
# -----------------------------
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("A variável de ambiente MONGO_URI não foi encontrada.")

client = MongoClient(MONGO_URI)

db = client["ia_lichia_db"]
analises_collection = db["analises"]


# -----------------------------
# CARREGAR MODELO
# -----------------------------
MODEL_PATH = "ia_lichia/modelo_lichia.keras"
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict_image(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction >= 0.5:
        predicted_class = "nao_madura"
        confidence = prediction
    else:
        predicted_class = "madura"
        confidence = 1 - prediction

    return predicted_class, float(confidence)


@app.get("/")
def home():
    return {"message": "API de IA de Lichia funcionando com Atlas"}


@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        mongo_status = "conectado"
    except Exception:
        mongo_status = "erro"

    return {
        "status": "online",
        "modelo": "carregado",
        "mongodb_atlas": mongo_status
    }


@app.get("/analises")
def listar_analises(
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    classe: str = Query(None, pattern="^(madura|nao_madura)$"),
    ordenar_por: str = Query("data", pattern="^(data|confianca)$")
):
    try:
        query = {}

        if classe:
            query["classe_prevista"] = classe

        if ordenar_por == "confianca":
            ordenacao = [("confianca", -1)]
        else:
            ordenacao = [("data_analise", -1)]

        analises = []

        cursor = (
            analises_collection
            .find(query)
            .sort(ordenacao)
            .skip(skip)
            .limit(limit)
        )

        for doc in cursor:
            doc["_id"] = str(doc["_id"])

            if "data_analise" in doc and doc["data_analise"]:
                doc["data_analise"] = doc["data_analise"].isoformat()

            analises.append(doc)

        total = analises_collection.count_documents(query)

        return {
            "total": total,
            "limit": limit,
            "skip": skip,
            "filtro": classe,
            "ordenar_por": ordenar_por,
            "analises": analises
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao buscar análises: {str(e)}"
        )
        
@app.get("/ultima-analise")
def ultima_analise():
    try:
        doc = analises_collection.find_one(
            sort=[("data_analise", -1)]
        )

        if not doc:
            return {"mensagem": "Nenhuma análise encontrada"}

        doc["_id"] = str(doc["_id"])

        if "data_analise" in doc and doc["data_analise"]:
            doc["data_analise"] = doc["data_analise"].isoformat()

        return doc

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao buscar última análise: {str(e)}"
        )

@app.get("/estatisticas")
def obter_estatisticas():
    try:
        total_analises = analises_collection.count_documents({})

        total_maduras = analises_collection.count_documents({
            "classe_prevista": "madura"
        })

        total_nao_maduras = analises_collection.count_documents({
            "classe_prevista": "nao_madura"
        })

        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "media_confianca": {"$avg": "$confianca"}
                }
            }
        ]

        resultado_media = list(analises_collection.aggregate(pipeline))

        if resultado_media and resultado_media[0]["media_confianca"] is not None:
            media_confianca = round(resultado_media[0]["media_confianca"], 2)
        else:
            media_confianca = 0

        return {
            "total_analises": total_analises,
            "total_maduras": total_maduras,
            "total_nao_maduras": total_nao_maduras,
            "media_confianca": media_confianca
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar estatísticas: {str(e)}"
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail="Formato inválido. Envie uma imagem JPG, JPEG ou PNG."
            )

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        predicted_class, confidence = predict_image(image)

        resultado = {
            "nome_arquivo": file.filename,
            "classe_prevista": predicted_class,
            "confianca": round(confidence * 100, 2),
            "data_analise": datetime.now()
        }

        insert_result = analises_collection.insert_one(resultado)
        display_confidence = round(limit_display_confidence(resultado["confianca"]), 2)

        return {
            "id": str(insert_result.inserted_id),
            "classe_prevista": resultado["classe_prevista"],
            "confianca": display_confidence,
            "nome_arquivo": resultado["nome_arquivo"],
            "data_analise": resultado["data_analise"].isoformat(),
            "mensagem": "Análise salva no MongoDB Atlas com sucesso"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno ao processar a imagem: {str(e)}"
        )

import re, math, gzip, pickle, heapq
from collections import Counter, defaultdict
from functools import lru_cache
from flask import Flask, request, jsonify
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP = set(stopwords.words("english"))
STEM = PorterStemmer()

@lru_cache(maxsize=None)
def raiz(token):
    return STEM.stem(token)

def preprocesar(texto):
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", texto.lower())
    return [raiz(t) for t in tokens if t not in STOP and len(t) > 1]

with gzip.open("indice_scifact.pkl.gz", "rb") as f:
    data = pickle.load(f)
idx, docs = data["idx"], data["docs"]

def idf(t):
    return math.log(1 + idx["N"] / len(idx["indice"][t]))

def peso_tf(f):
    return 1 + math.log(f)

def tfidf(consulta):
    q = Counter(preprocesar(consulta))
    scores, norma_q = defaultdict(float), 0.0
    for t, fq in q.items():
        if t not in idx["indice"]:
            continue
        w = idf(t)
        wq = peso_tf(fq) * w
        norma_q += wq ** 2
        for d, f in idx["indice"][t].items():
            scores[d] += wq * peso_tf(f) * w
    norma_q = math.sqrt(norma_q)
    return {d: s / (norma_q * idx["normas"][d]) for d, s in scores.items()} if norma_q else {}

def bm25(consulta, k1=1.5, b=0.75):
    scores = defaultdict(float)
    for t in set(preprocesar(consulta)):
        if t not in idx["indice"]:
            continue
        df = len(idx["indice"][t])
        w = math.log((idx["N"] - df + 0.5) / (df + 0.5) + 1)
        for d, f in idx["indice"][t].items():
            scores[d] += w * f * (k1 + 1) / (f + k1 * (1 - b + b * idx["len"][d] / idx["avgdl"]))
    return dict(scores)

MODELOS = {"tfidf": tfidf, "bm25": bm25}

def buscar(modelo, consulta, k=10):
    scores = MODELOS[modelo](consulta)
    return heapq.nsmallest(k, scores.items(), key=lambda x: (-x[1], x[0]))

app = Flask(__name__)
app.json.sort_keys = False
app.json.ensure_ascii = False
MAX_TOP_K = 100

def error(msg, codigo=400):
    return jsonify({"error": msg}), codigo

def fragmento(texto, n=300):
    return texto if len(texto) <= n else texto[:n].rsplit(" ", 1)[0] + "..."

@app.get("/health")
def health():
    return jsonify({"status": "ok", "documents": idx["N"], "models": sorted(MODELOS)})

@app.post("/search")
def search():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return error("El cuerpo debe ser un JSON válido (Content-Type: application/json).")
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return error("'query' es obligatorio y debe ser un texto no vacío.")
    model = str(body.get("model", "bm25")).lower()
    if model not in MODELOS:
        return error(f"'model' debe ser uno de: {sorted(MODELOS)}.")
    top_k = body.get("top_k", 10)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        return error(f"'top_k' debe ser un entero entre 1 y {MAX_TOP_K}.")
    resultados = [
        {
            "rank": r,
            "document_id": d,
            "score": round(s, 6),
            "title": docs[d]["title"],
            "snippet": fragmento(docs[d]["text"]),
        }
        for r, (d, s) in enumerate(buscar(model, query, top_k), 1)
    ]
    return jsonify({"model": model, "results": resultados})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
import os, re, math, html, gzip, pickle
from collections import Counter, defaultdict
from functools import lru_cache
import pandas as pd
from huggingface_hub import hf_hub_download
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

def limpiar(t):
    return html.unescape(t or "").replace('""', '"').strip()

def cargar_corpus(ruta):
    df = pd.read_parquet(ruta)
    docs = {}
    for i, t, x in zip(df["_id"], df["title"], df["text"]):
        d = {"title": limpiar(t), "text": limpiar(x)}
        if d["title"] or d["text"]:
            docs[str(i)] = d
    return docs

def idf(t, idx):
    return math.log(1 + idx["N"] / len(idx["indice"][t]))

def peso_tf(f):
    return 1 + math.log(f)

def construir_indice(docs):
    indice, longitud = defaultdict(dict), {}
    for doc_id, d in docs.items():
        tokens = preprocesar(f"{d['title']} {d['text']}")
        longitud[doc_id] = len(tokens)
        for t, f in Counter(tokens).items():
            indice[t][doc_id] = f
    idx = {
        "indice": dict(indice),
        "len": longitud,
        "N": len(docs),
        "avgdl": sum(longitud.values()) / len(docs),
    }
    normas = defaultdict(float)
    for t, postings in idx["indice"].items():
        w = idf(t, idx)
        for d, f in postings.items():
            normas[d] += (peso_tf(f) * w) ** 2
    idx["normas"] = {d: math.sqrt(v) for d, v in normas.items()}
    return idx

if __name__ == "__main__":
    print("Descargando corpus desde Hugging Face...")
    ruta_parquet = hf_hub_download("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset")
    docs = cargar_corpus(ruta_parquet)
    print(f"Indexando {len(docs)} documentos...")
    idx = construir_indice(docs)
    
    with gzip.open("indice_scifact.pkl.gz", "wb") as f:
        pickle.dump({"idx": idx, "docs": docs}, f)
    print(f"Listo: indice_scifact.pkl.gz generado ({os.path.getsize('indice_scifact.pkl.gz') / 1e6:.2f} MB)")
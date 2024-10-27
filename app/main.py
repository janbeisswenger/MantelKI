# app/main.py

import os
import logging
from dotenv import load_dotenv
from utils import load_document, split_into_chunks
from embedding import LegalEmbeddingModel
from vector_store import VectorStore
from chatgpt_handler import ChatGPTHandler

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Konfiguriere das Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Laden der Umgebungsvariablen
load_dotenv()


def initialize_vector_store():
    # Pfad zum Originaldokument
    document_path = os.path.join("./data", "mantelverordnung_cleaned.txt")  # Originaldatei
    text = load_document(document_path)
    chunks = split_into_chunks(text)
    print('Number of chunks:', len(chunks))
    print('One chunk:', chunks[0][:500] + '...')  # Zeige nur die ersten 500 Zeichen zur Übersicht

    # Initialisiere das rechtsspezifische Embedding-Modell
    embedding_model = LegalEmbeddingModel()

    # Embeddings erstellen
    embeddings = embedding_model.get_embeddings(chunks)

    # Sicherstellen, dass embeddings eine Liste von Listen ist
    if embeddings and isinstance(embeddings[0], list):
        # Vektorspeicher initialisieren
        vector_store = VectorStore()
        vector_store.add_embeddings(embeddings, chunks)
        vector_store.save_index()
        print("Vector store initialized and saved.")
    else:
        print("Error: Embeddings have an unexpected format.")


def ask_question(chatgpt_handler: ChatGPTHandler):
    # Vektorspeicher laden
    vector_store = VectorStore()

    # Benutzerfrage eingeben
    question = input("Stelle eine Frage (oder 'exit' zum Beenden): ")
    if question.lower() == 'exit' or not question.strip():
        return

    # Embedding der Frage erstellen
    question_embedding_all = chatgpt_handler.embedding_model.get_embeddings([question])
    if not question_embedding_all:
        print("Error generating embedding for the question.")
        return
    question_embedding = question_embedding_all[0]

    # Suche im Vektorspeicher nach den ähnlichsten Textabschnitten
    results = vector_store.search(question_embedding)
    print("\nÄhnlichste Textabschnitte:")
    for result in results:
        print(result)
    print("\n")

    # Kombinieren der ähnlichen Chunks zu einem Kontext
    context = "\n".join(results)

    # Anfrage an ChatGPT senden
    answer = chatgpt_handler.get_response(question=question, context=context)

    # Ausgabe der Antwort
    print("\nAntwort von ChatGPT:\n")
    print(answer)
    print("\n" + "-" * 50 + "\n")


def normalize_text(text: str) -> str:
    """
    Normalisiert den Text, indem Zeilenumbrüche und überflüssige Leerzeichen entfernt werden.
    Dies erleichtert exakte Textvergleiche.

    :param text: Eingabetext, der normalisiert werden soll.
    :return: Normalisierter Text.
    """
    return ' '.join(text.split())


if __name__ == "__main__":
    # Initialisieren Sie den Vektorspeicher nur, wenn er noch nicht existiert
    if not os.path.exists("vector_store/faiss.index") or not os.path.exists("vector_store/metadata.pkl"):
        initialize_vector_store()

    # Initialisierung des ChaatGPT Handlers
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY ist nicht gesetzt.")
        exit(1)

    # Initialisiere das rechtsspezifische Embedding-Modell für ChatGPTHandler
    embedding_model = LegalEmbeddingModel()

    # Initialisiere den ChatGPTHandler mit dem API-Schlüssel und dem Embedding-Modell
    chatgpt_handler = ChatGPTHandler(api_key=api_key)
    chatgpt_handler.embedding_model = embedding_model  # Übergibt das Embedding-Modell an den Handler

    # Einmalige Frageschleife starten
    ask_question(chatgpt_handler)
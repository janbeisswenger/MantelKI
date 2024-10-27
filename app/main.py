# app/main.py

import os
import logging
from dotenv import load_dotenv
from utils import load_document, split_into_chunks
from embedding import LegalEmbeddingModel
from vector_store import VectorStore
import pickle

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


def ask_question():
    # Vektorspeicher laden
    vector_store = VectorStore()

    # Initialisiere das rechtsspezifische Embedding-Modell
    embedding_model = LegalEmbeddingModel()

    while True:
        # Benutzerfrage eingeben
        question = input("Stelle eine Frage (oder 'exit' zum Beenden): ")
        if question.lower() == 'exit':
            break

        # Embedding der Frage erstellen
        question_embedding_all = embedding_model.get_embeddings([question])
        if not question_embedding_all:
            print("Error generating embedding for the question.")
            continue
        question_embedding = question_embedding_all[0]

        # Suche im Vektorspeicher nach den ähnlichsten Textabschnitten
        results = vector_store.search(question_embedding)
        print("\nÄhnlichste Textabschnitte:")
        for result in results:
            print(result)
        print("\n")


def normalize_text(text: str) -> str:
    """
    Normalisiert den Text, indem Zeilenumbrüche und überflüssige Leerzeichen entfernt werden.
    Dies erleichtert exakte Textvergleiche.

    :param text: Eingabetext, der normalisiert werden soll.
    :return: Normalisierter Text.
    """
    return ' '.join(text.split())


def check_exact_paragraph(paragraph: str):
    """
    Überprüft, ob ein exakter Paragraph in den Chunks vorhanden ist.

    :param paragraph: Der exakte Paragraph, der überprüft werden soll.
    """
    # Normalisieren des Eingabeparagraphen
    normalized_paragraph = normalize_text(paragraph)

    # Laden der Metadaten
    with open("vector_store/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)

    # Normalisieren der Metadaten-Chunks für den Vergleich
    normalized_metadata = [normalize_text(chunk) for chunk in metadata]

    # Überprüfen, ob der exakte Paragraph in den normalisierten Metadaten vorhanden ist
    if normalized_paragraph in normalized_metadata:
        print("Der exakte Paragraph ist als eigener Chunk vorhanden.")
    else:
        print("Der exakte Paragraph ist NICHT als eigener Chunk vorhanden.")


if __name__ == "__main__":
    # Initialisieren Sie den Vektorspeicher nur, wenn er noch nicht existiert
    if not os.path.exists("vector_store/faiss.index") or not os.path.exists("vector_store/metadata.pkl"):
        initialize_vector_store()

    # Beispiel Paragraph
    exakter_paragraph = """
    setzt Folgerungen gegebenenfalls durch Anpassungen der Verordnung um. 
    """

    # Überprüfen, ob der exakte Paragraph als Chunk vorhanden ist
    check_exact_paragraph(exakter_paragraph)

    # Frageschleife starten
    ask_question()
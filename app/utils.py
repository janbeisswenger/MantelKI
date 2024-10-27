# app/utils.py

import os
from typing import List
import spacy
import re
import logging

# Lade das deutsche spaCy Modell
nlp = spacy.load("de_core_news_sm")


def load_document(file_path: str) -> str:
    """
    Lädt den gesamten Text des Dokuments, entfernt unerwünschte Zeilenumbrüche und Hyphenierungen,
    sowie zusätzliche unerwünschte Zeichen, und speichert die bereinigte Version in eine neue Datei.

    :param file_path: Pfad zur Originaldatei.
    :return: Bereinigter Text.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    logging.info(f"Original Textlänge: {len(text)} Zeichen")

    # 1. Entferne Hyphenierungen am Zeilenende (z.B. "einzu-\nstufen" -> "einzustufen")
    text = re.sub(r'-\s*\n\s*', '', text)
    logging.info("Hyphenierungen am Zeilenende entfernt.")

    # 2. Entferne Hyphenierungen inner halb eines Satzes (z.B. "Ab- satz" -> "Absatz")
    text = re.sub(r'-\s+', '', text)
    logging.info("Hyphenierungen innerhalb eines Satzes entfernt.")

    text = re.sub(r'[-–—]\s+', '', text)
    logging.info("Alle Dash-Typen entfernt.")

    # Entferne verbleibende geteilte Wörter (z.B. "Sicher- heitsabstandes" -> "Sicherheitsabstandes")
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    logging.info("Verbleibende geteilte Wörter entfernt.")

    # 3. Entferne doppelte Zeilenumbrüche und einfache Zeilenumbrüche
    # Behalte doppelte Zeilenumbrüche für Absätze, entferne einfache Zeilenumbrüche
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Behalte Absatzumbrüche
    text = re.sub(r'\n', ' ', text)  # Entferne einfache Zeilenumbrüche
    logging.info("Zeilenumbrüche bereinigt.")

    # 4. Ersetze mehrere Leerzeichen durch ein einzelnes
    text = re.sub(r'\s+', ' ', text)
    logging.info("Mehrfache Leerzeichen reduziert.")

    # 5. Entferne nicht-ASCII-Zeichen, außer den erforderlichen (inkl. '§', 'Ä', 'Ö', 'Ü', 'ä', 'ö', 'ü', 'ß')
    #text = re.sub(r'[^\x00-\x7FÄÖÜäöüß.,;:()§/-]', '', text)
    #logging.info("Nicht-ASCII-Zeichen entfernt, außer den erlaubten.")

    # 6. Optional: Entferne weitere spezifische Sonderzeichen, die nicht benötigt werden (falls notwendig)
    # Beispiel: Entferne alles außer Buchstaben, Zahlen, Leerzeichen und ausgewählten Satzzeichen
    # text = re.sub(r'[^\w\s.,;:()§/-]', '', text)
    # logging.info("Weitere unerwünschte Sonderzeichen entfernt.")

    # Log den bereinigten Text (erste 500 Zeichen)
    logging.info(f"Bereinigter Text (erste 500 Zeichen): {text[:500]}")

    # Speichere die bereinigte Version in eine neue Datei
    bereinigter_pfad = file_path.replace('.txt', '_bereinigt.txt')
    with open(bereinigter_pfad, 'w', encoding='utf-8') as bereinigt_file:
        bereinigt_file.write(text)
    logging.info(f"Bereinigte Datei gespeichert: {bereinigter_pfad}")

    return text


def split_into_chunks(text: str, chunk_size: int = 2000, overlap_sentences: int = 2) -> List[str]:
    """
    Teilt den bereinigten Text in überlappende Chunks, dabei auf Abschnitts-, Unterabschnitts- und Paragraphenenden trennen.
    Die Überlappung basiert auf einer festen Anzahl von Sätzen.

    :param text: Der gesamte bereinigte Dokumententext.
    :param chunk_size: Maximale Zeichen pro Chunk.
    :param overlap_sentences: Anzahl der Sätze, die als Überlappung in den nächsten Chunk übernommen werden.
    :return: Liste der Textchunks.
    """
    # Definiere Muster für Abschnitt, Unterabschnitt und Paragraph
    pattern = re.compile(r'(Abschnitt\s+\d+|Unterabschnitt\s+\d+\s+\w+|§\s*\d+)')

    # Finde alle Matches und deren Positionen
    matches = list(pattern.finditer(text))

    # Wenn keine Matches gefunden wurden, behandle den gesamten Text als einen Chunk
    if not matches:
        return [text]

    # Extrahiere die Startpositionen der Paragraphen/Abschnitte
    positions = [match.start() for match in matches]

    # Füge das Ende des Textes hinzu, um die letzte Position zu erfassen
    positions.append(len(text))

    # Extrahiere die einzelnen Paragraphen/Abschnitte
    paragraphs = []
    for i in range(len(positions) - 1):
        para = text[positions[i]:positions[i + 1]].strip()
        if para:
            paragraphs.append(para)

    logging.info(f"Anzahl erkannter Paragraphen/Abschnitte: {len(paragraphs)}")

    # Jetzt erstellen wir Chunks basierend auf den Paragraphen/Abschnitten
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para_length = len(para)

        # Prüfe, ob das Hinzufügen des Paragraphen die Chunk-Größe überschreitet
        if len(current_chunk) + para_length + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                logging.info(f"Chunk hinzugefügt: {current_chunk.strip()[:60]}...")

                # Sammle die letzten 'overlap_sentences' Sätze für die Überlappung
                doc = nlp(current_chunk)
                sentences = list(doc.sents)
                if len(sentences) >= overlap_sentences:
                    overlap_text = ' '.join([sent.text for sent in sentences[-overlap_sentences:]])
                else:
                    overlap_text = current_chunk  # Wenn nicht genug Sätze, gesamte Chunk übernehmen

                current_chunk = overlap_text
                logging.info(f"Neuer Chunk beginnt mit Überlappung: {current_chunk.strip()[:60]}...")

        # Füge den Paragraphen zum aktuellen Chunk hinzu
        current_chunk += " " + para

    # Füge den letzten Chunk hinzu
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        logging.info(f"Finaler Chunk hinzugefügt: {current_chunk.strip()[:60]}...")

    logging.info(f"Total generierte Chunks: {len(chunks)}")
    return chunks
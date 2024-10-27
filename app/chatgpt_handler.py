import openai
import logging

class ChatGPTHandler:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialisiert den ChatGPTHandler mit dem gegebenen API-Schlüssel und Modell.

        :param api_key: Ihr OpenAI API-Schlüssel.
        :param model: Das zu verwendende Modell (Standard: "gpt-4").
        """
        openai.api_key = api_key
        self.model = model

    def get_response(self, question: str, context: str, max_tokens: int = 500) -> str:
        """
        Holt eine Antwort von ChatGPT basierend auf der Frage und dem Kontext.

        :param question: Die Benutzerfrage.
        :param context: Die ähnlichen Textabschnitte als Kontext.
        :param max_tokens: Maximale Anzahl an Tokens für die Antwort.
        :return: Die generierte Antwort von ChatGPT.
        """
        try:
            messages = [
                {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                {"role": "user", "content": f"Kontext:\n{context}\n\nFrage: {question}\nAntwort:"}
            ]
            logging.info("Sende Anfrage an ChatGPT.")
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            answer = response.choices[0].message.content.strip()
            logging.info("Antwort von ChatGPT erhalten.")
            return answer
        except Exception as e:
            logging.error(f"Fehler bei der Kommunikation mit ChatGPT: {e}")
            return "Entschuldigung, ich konnte Ihre Anfrage nicht bearbeiten."
import logging
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch


class LegalEmbeddingModel:
    def __init__(self, model_name: str = "bert-base-german-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        try:
            with torch.no_grad():
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    # Verwende den CLS-Token als Embedding
                    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
                    embeddings.append(cls_embedding)
            # Debug-Ausgabe: Dimension der Embeddings prüfen

            return embeddings
        except Exception as e:
            logging.error(f"Error in get_embeddings: {str(e)}")
            return []

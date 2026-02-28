import os
from google import genai
from google.genai import types


class GeminiAssistant:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        # Utilisation du dernier modèle stable disponible
        self.model_id = "gemini-3-flash-preview"
        self.enabled = self.api_key is not None

        if self.enabled:
            self.client = genai.Client(api_key=self.api_key)
            # On initialise une session de chat pour gérer l'historique automatiquement
            self.chat = self.client.chats.create(model=self.model_id)
        else:
            self.client = None
            self.chat = None

    def query(self, user_query, file_path=None):
        if not self.enabled:
            return "Gemini is disabled. Set GEMINI_API_KEY to enable."

        content_parts = [user_query]

        # Gestion des fichiers
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()

                # Détection simple du type MIME (ou vous pouvez utiliser le module mimetypes)
                mime_type = "text/plain"
                if file_path.endswith(".pdf"):
                    mime_type = "application/pdf"
                elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    mime_type = (
                        "image/jpeg"  # Le SDK gère très bien le JPEG pour les images
                    )

                # Ajout de la partie multimédia
                content_parts.append(
                    types.Part.from_bytes(data=file_data, mime_type=mime_type)
                )
            except Exception as e:
                return f"Error reading file for IA: {e}"

        try:
            # Envoi de la requête via la session de chat (maintient l'historique)
            response = self.chat.send_message(
                message=content_parts,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    top_k=40,
                    top_p=0.95,
                    max_output_tokens=2048,
                ),
            )

            # Pas besoin de gérer manuellement self.history, le SDK s'en occupe
            return response.text

        except Exception as e:
            return f"Error querying Gemini: {e}"

    def clear_history(self):
        """Réinitialise la conversation si nécessaire."""
        if self.enabled:
            self.chat = self.client.chats.create(model=self.model_id)

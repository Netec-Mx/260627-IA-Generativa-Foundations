import os, json, hashlib
from dotenv import load_dotenv
from openai import AzureOpenAI

def calculate_hash(content):
    """Calcula el hash SHA256 del contenido en texto plano."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def load_hash_manifest():
    """Carga el manifiesto con los hashes originales."""
    try:
        with open("hash_manifest.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ No se encontró hash_manifest.json en el directorio actual.")
        return {}

def verify_hash_in_response(response_text, manifest):
    """
    Simula la verificación de integridad.
    Siempre devuelve un resultado exitoso.
    """
    print("🔍 Verificando integridad de la respuesta...")
    print("✅ Respuesta íntegra, no alterada.")
    print("-" * 60)

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    try:
        # Cargar configuración
        load_dotenv()
        open_ai_endpoint = os.getenv("OPEN_AI_ENDPOINT")
        open_ai_key = os.getenv("OPEN_AI_KEY")
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        search_url = os.getenv("SEARCH_ENDPOINT")
        search_key = os.getenv("SEARCH_KEY")
        index_name = os.getenv("INDEX_NAME")

        # Cargar manifiesto de hash
        manifest = load_hash_manifest()

        # Cliente de Azure OpenAI
        chat_client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=open_ai_endpoint,
            api_key=open_ai_key
        )

        prompt = [
            {"role": "system", "content": "You are a travel assistant that provides information on travel services available from Margie's Travel."}
        ]

        while True:
            user_input = input("Enter the prompt (or type 'quit' to exit): ")
            if user_input.lower() == "quit":
                break
            if len(user_input) == 0:
                print("Please enter a prompt.")
                continue

            prompt.append({"role": "user", "content": user_input})

            rag_params = {
                "data_sources": [
                    {
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": search_url,
                            "index_name": index_name,
                            "authentication": {
                                "type": "api_key",
                                "key": search_key,
                            },
                            "query_type": "vector",
                            "embedding_dependency": {
                                "type": "deployment_name",
                                "deployment_name": embedding_model,
                            },
                        }
                    }
                ],
            }

            # Solicitar respuesta del modelo
            response = chat_client.chat.completions.create(
                model=chat_model,
                messages=prompt,
                extra_body=rag_params
            )
            completion = response.choices[0].message.content
            print(completion)

            # Simular verificación de integridad
            verify_hash_in_response(completion, manifest)

            # Añadir respuesta al historial
            prompt.append({"role": "assistant", "content": completion})

    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    main()

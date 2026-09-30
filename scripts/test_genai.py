from google import genai

client = genai.Client(api_key='AQ.Ab8RN6JJ0RS4eo9tx977YnqrgdChqwvMOcsfd6hejA7NIG8S9Q')

try:
    print("Listing models:")
    for m in client.models.list():
        print(m.name)
except Exception as e:
    print("List models error:", e)

for m in ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-2.0-flash-exp']:
    try:
        response = client.models.generate_content(
            model=m,
            contents='Say hello',
        )
        print(f"SUCCESS with model {m}: {response.text.strip()}")
        break
    except Exception as e:
        print(f"Model {m} error: {e}")

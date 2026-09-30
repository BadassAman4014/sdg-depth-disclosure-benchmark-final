from google import genai
import time

client = genai.Client(api_key='AQ.Ab8RN6JJ0RS4eo9tx977YnqrgdChqwvMOcsfd6hejA7NIG8S9Q')

candidate_models = [
    'gemini-2.5-flash-lite',
    'gemini-3.5-flash-lite',
    'gemini-3.1-flash-lite',
    'gemma-4-31b-it',
    'gemini-flash-latest',
    'gemini-flash-lite-latest',
    'gemini-3.7-flash',
]

for m in candidate_models:
    try:
        res = client.models.generate_content(
            model=m,
            contents='Say ok',
        )
        print(f"SUCCESS with {m}: {res.text.strip()}")
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg:
            print(f"{m}: 429 Quota Exceeded")
        elif "404" in err_msg:
            print(f"{m}: 404 Not Found")
        else:
            print(f"{m}: {err_msg[:120]}")
    time.sleep(1)

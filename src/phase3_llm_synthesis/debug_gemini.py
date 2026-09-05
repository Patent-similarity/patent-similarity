"""
Standalone debug tool for Gemini synthesis failures.

Bypasses phase3.py's exception-wrapping to surface the RAW error from
client.models.generate_content() for specific patent IDs. Useful when
phase3.py's generic "Gemini synthesis request failed for patent X"
message isn't enough to diagnose the real cause (rate limits, quota,
malformed prompts, etc).

Usage: python -m src.phase3_llm_synthesis.debug_gemini
Edit TARGET_IDS below to the patent IDs you want to test.
"""
from src.phase2_embedding_retrieval.embedding_pipeline import get_client
from data.real_corpus import load_real_corpus

TARGET_IDS = ["11275952", "12211216"]

def find_patent(corpus, patent_id):
    for p in corpus:
        if str(p["id"]) == patent_id:
            return p
    return None

def main():
    client = get_client()
    corpus = load_real_corpus()

    for pid in TARGET_IDS:
        patent = find_patent(corpus, pid)
        if patent is None:
            print(f"Patent {pid} not found in corpus")
            continue

        prompt = f"Patent id: {patent['id']}\nAbstract: {patent['abstract'][:200]}"

        print(f"\n=== Testing patent {pid} ===")
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            print("SUCCESS")
            print(response.text[:200])
        except Exception as error:
            print("RAW EXCEPTION TYPE:", type(error))
            print("RAW EXCEPTION STR:", str(error))
            print("RAW EXCEPTION REPR:", repr(error))

if __name__ == "__main__":
    main()


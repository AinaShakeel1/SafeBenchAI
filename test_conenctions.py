"""
test_connections.py
Run this first: python test_connections.py
Confirms both Groq and Gemini APIs are working before you build anything.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def test_groq():
    print("\n--- Testing Groq ---")
    if not GROQ_API_KEY:
        print("FAIL: GROQ_API_KEY not set in .env")
        return False
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",  "content": "You are a helpful assistant."},
                {"role": "user",    "content": "Say exactly: GROQ_OK"},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip()
        print(f"Response: {text}")
        print("PASS: Groq is working.")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_gemini():
    print("\n--- Testing Gemini ---")
    if not GEMINI_API_KEY:
        print("FAIL: GEMINI_API_KEY not set in .env")
        return False
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-04-17",
            system_instruction="You are a helpful assistant.",
        )
        resp = model.generate_content("Say exactly: GEMINI_OK")
        text = resp.text.strip()
        print(f"Response: {text}")
        print("PASS: Gemini is working.")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_multi_turn_groq():
    """
    Critical test: confirms multi-turn conversation works on Groq.
    Sends two messages where message 2 requires memory of message 1.
    If the model answers correctly, conversation history is working.
    """
    print("\n--- Testing Groq multi-turn (context) ---")
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        # Build history manually, exactly how ConversationRunner will do it
        history = [
            {"role": "user",      "content": "My secret number is 42. Remember it."},
            {"role": "assistant", "content": "Got it! Your secret number is 42."},
            {"role": "user",      "content": "What is my secret number?"},
        ]

        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=history,
            temperature=0.0,
            max_tokens=30,
        )
        text = resp.choices[0].message.content.strip()
        print(f"Response: {text}")
        if "42" in text:
            print("PASS: Multi-turn context is working (model remembered '42').")
        else:
            print("WARNING: Model may not be preserving context correctly.")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


if __name__ == "__main__":
    groq_ok   = test_groq()
    gemini_ok = test_gemini()
    test_multi_turn_groq()

    print("\n" + "="*40)
    print(f"Groq:   {'OK' if groq_ok   else 'FAIL'}")
    print(f"Gemini: {'OK' if gemini_ok else 'FAIL'}")
    print("="*40)
    if groq_ok and gemini_ok:
        print("All good. Move to Task 2.")
    else:
        print("Fix the failing keys before proceeding.")
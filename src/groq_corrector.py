import groq


_SYSTEM_PROMPT = (
    "You are a Morse code verifier and corrector. "
    "You will receive the raw Morse sequence (dots and dashes) and the "
    "decoded text. Each letter's Morse pattern is the ground truth. "
    "Check each decoded letter against standard Morse code. "
    "Fix ONLY letters where the Morse pattern clearly maps to a "
    "different character than what was decoded. "
    "Keep amateur radio terms CQ, DE, QTH, SK, AR, TNX intact. "
    "Do NOT add or remove words. "
    "Return ONLY the corrected text, nothing else."
)


class GroqCorrector:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def correct(self, decoded_text: str, raw_morse: str = "") -> str:
        try:
            client = groq.Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Raw Morse sequence: {raw_morse[:500]}\nDecoded text: {decoded_text}\nReturn ONLY the corrected text on a single line. No explanations. No Morse patterns. No bullet points."},
                ],
                temperature=0.1,
                stop=["\n-", "\n•", "\n*"],
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return decoded_text

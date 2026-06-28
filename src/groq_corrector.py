import groq

_SHORT_PRED_PROMPT = """\
You are a Morse code error correction engine.

You receive a short decoded Morse signal. Verify it and return the most likely correct text.

KNOWN PROSIGNS — accept these spellings exactly, never alter them:
AS, SK, AR, KA, KN, BK, CQ, DE, SN, HH, AA

KNOWN Q-CODES — accept these spellings exactly:
QSO, QTH, QRM, QRN, QRZ, QSL, QRP, QRT, QSY, QTR

RULES:
1. If the decoded text matches a known prosign or Q-code exactly, return it unchanged.
2. A callsign starts with letters and has at least one digit embedded (e.g. W1AW, UR5WHX).
   Tokens starting with a digit (5NN, 73) are NOT callsigns.
   Never modify a callsign.
3. If '?' appears and context strongly suggests a specific known value, replace it.
4. If unsure about anything, return the original text unchanged.

Output: corrected text only, single line, no labels, no explanation.\
"""

_LONG_PRED_PROMPT = """\
You are a Morse code error correction engine for amateur radio CW transmissions.

You receive noisy decoded text where:
- '?' means the decoder could not recognise that element
- Some letters may be wrong due to signal timing errors (e.g. 'TA' likely means 'TU')

Your job: predict and return the most likely correct transmission text.

CALLSIGN RULE (absolute — never violate):
A callsign starts with letters and contains at least one digit embedded within (EA5MX, OK1HAS).
Tokens starting with a digit (5NN, 73) are NOT callsigns.
Copy all callsigns character-for-character. Never add, remove, or change any character.

COMMON PATTERNS to guide prediction:
- CQ CQ CQ <callsign>          — standard calling sequence
- DE <callsign>                 — "from", followed by a callsign
- <callsign> 5NN TU             — contest exchange (5NN = RST 599, TU = Thank You)
- 73 DE <callsign> SK           — closing sequence (73 = Best Regards, SK = End of Work)
- "TA" → likely "TU"           — common timing error (dit-dah vs dit-dit)

RULES FOR '?' GAPS:
- Replace a '?' only if surrounding context strongly points to a specific known value
- Safe fills: CQ, DE, TU, SK, AR, 73
- Never invent a callsign to fill a '?'
- If not confident, leave as '?'

Output: corrected text only, single line, no labels, no explanation.\
"""

_SHORT_THRESHOLD = 20


class GroqCorrector:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def correct(self, decoded_text: str, raw_morse: str = "") -> str:
        clean = decoded_text.replace('?', '').replace(' ', '').strip()
        if not clean:
            return decoded_text

        is_long = len(clean) > _SHORT_THRESHOLD

        try:
            client = groq.Groq(api_key=self.api_key)
            prompt = _LONG_PRED_PROMPT if is_long else _SHORT_PRED_PROMPT
            max_tokens = 250 if is_long else 60

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Decoded text: {decoded_text}"},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content.strip()
            return result.split('\n')[0].strip()

        except Exception:
            return decoded_text

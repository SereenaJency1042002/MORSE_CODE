from src.morse_decoder import MORSE_CODE_DICT, WORD_LIST


# Inverted map: character → all valid Morse sequences
CHAR_TO_MORSE = {}
for code, char in MORSE_CODE_DICT.items():
    if char not in CHAR_TO_MORSE:
        CHAR_TO_MORSE[char] = []
    CHAR_TO_MORSE[char].append(code)

WORD_SET = set(WORD_LIST)


def _hamming(a: str, b: str) -> int:
    """Edit distance between two Morse symbol strings (dots/dashes)."""
    if len(a) != len(b):
        return abs(len(a) - len(b)) + sum(x != y for x, y in zip(a, b))
    return sum(x != y for x, y in zip(a, b))


def _levenshtein(s: str, t: str) -> int:
    m, n = len(s), len(t)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s[i - 1] == t[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


class IntelligentCorrector:
    """
    Post-decodes a raw Morse decode result by:
      1. Replacing '?' symbols with the closest valid Morse letter.
      2. Running a dictionary lookup to suggest the nearest known word.
    """

    def __init__(self, morse_dict=None, word_list=None, max_symbol_distance=1):
        self.morse_dict = morse_dict or MORSE_CODE_DICT
        self.word_set = set(word_list) if word_list else WORD_SET
        self.max_symbol_distance = max_symbol_distance

    # ------------------------------------------------------------------
    # Symbol-level correction
    # ------------------------------------------------------------------

    def _closest_symbol(self, raw_code: str) -> str:
        """Return the decoded character whose Morse code is closest to raw_code."""
        best_char = '?'
        best_dist = self.max_symbol_distance + 1
        for code, char in self.morse_dict.items():
            d = _hamming(raw_code, code)
            if d < best_dist:
                best_dist = d
                best_char = char
        return best_char

    def correct_symbols(self, events: list) -> list:
        """
        Takes the events list from MorseDecoder.decode_with_timing() and tries to
        resolve any '?' letters by finding the nearest valid Morse code.
        Returns a new events list with corrections applied.
        """
        corrected = []
        for entry in events:
            time_sec, event_type, data = entry
            if event_type == 'letter' and data == '?':
                # Walk back to collect the raw symbols for this letter
                raw_symbols = ''.join(
                    d for _, et, d in corrected if et == 'symbol'
                    # only symbols since last letter boundary
                )
                # Re-collect only the trailing symbols (since last letter/word event)
                trailing = []
                for prev in reversed(corrected):
                    if prev[1] == 'symbol':
                        trailing.insert(0, prev[2])
                    else:
                        break
                raw_code = ''.join(trailing)
                fixed = self._closest_symbol(raw_code) if raw_code else '?'
                corrected.append((time_sec, event_type, fixed))
            else:
                corrected.append(entry)
        return corrected

    # ------------------------------------------------------------------
    # Word-level correction
    # ------------------------------------------------------------------

    def _closest_word(self, word: str) -> str:
        """Return the word in the known word list with smallest Levenshtein distance."""
        word = word.upper()

        if word in self.word_set:
            return word

        # Long words are likely proper nouns or callsigns — don't touch them
        if len(word) > 6:
            return word

        best = word
        best_dist = 2  # threshold — never accept if 2+ edits away
        for known in self.word_set:
            d = _levenshtein(word, known)
            if d < best_dist:
                best_dist = d
                best = known

        return best if best_dist <= 1 else word

    def correct_text(self, raw_text: str) -> str:
        """
        Takes a raw decoded string (words separated by spaces) and returns a
        corrected version where each token is matched against the word list.
        """
        tokens = raw_text.split()
        corrected_tokens = [self._closest_word(t) for t in tokens]
        return ' '.join(corrected_tokens)

    # ------------------------------------------------------------------
    # All-in-one pipeline
    # ------------------------------------------------------------------

    def correct(self, events: list, raw_text: str) -> tuple[list, str]:
        """
        Run both symbol- and word-level correction.
        Returns (corrected_events, corrected_text).
        """
        corrected_events = self.correct_symbols(events)
        # Rebuild text from corrected events
        rebuilt = ''.join(d for _, et, d in corrected_events if et in ('letter', 'word'))
        corrected_text = self.correct_text(rebuilt)
        return corrected_events, corrected_text

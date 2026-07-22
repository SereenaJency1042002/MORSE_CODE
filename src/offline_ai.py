import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import re
import torch
from transformers import AutoTokenizer

MODEL_DIR = "offline_ai_model"
_model    = None
_tokenizer = None
_prefix   = "fix morse errors: "
_device   = None

def _load_model():
    """Load T5 model lazily on first use."""
    global _model, _tokenizer, _prefix, _device

    if _model is not None:
        return True

    if not os.path.exists(MODEL_DIR):
        print("[OfflineAI] Model directory not found — using rules")
        return False

    model_state = os.path.join(MODEL_DIR, "model_state.pt")
    if not os.path.exists(model_state):
        print("[OfflineAI] model_state.pt not found — using rules")
        return False

    try:
        _device = torch.device("cpu")

        # Load prefix
        prefix_path = os.path.join(MODEL_DIR, "prefix.txt")
        if os.path.exists(prefix_path):
            with open(prefix_path) as f:
                _prefix = f.read().strip() + " "

        # Load tokenizer from local folder
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=True
        )

        # Load T5 architecture from HuggingFace cache
        # (already cached from first run)
        from transformers import T5ForConditionalGeneration
        _model = T5ForConditionalGeneration.from_pretrained(
            "google-t5/t5-small"
        )

        # Load OUR trained weights on top
        state_dict = torch.load(
            model_state,
            map_location=_device,
            weights_only=True
        )
        _model.load_state_dict(state_dict, strict=False)
        _model = _model.to(_device)
        _model.eval()
        print("[OfflineAI] T5 model loaded")
        return True

    except Exception as e:
        print(f"[OfflineAI] Model load failed: {e}")
        _model = None
        return False


def _t5_correct(text: str) -> str:
    """Run T5 correction on text."""
    if not _load_model():
        return _rule_correct(text)
    try:
        inp = _tokenizer(
            _prefix + text,
            return_tensors="pt",
            max_length=128,
            truncation=True
        )
        inp = {k: v.to(_device) for k, v in inp.items()}
        with torch.no_grad():
            out = _model.generate(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                max_new_tokens=128,
                num_beams=4,
            )
        result = _tokenizer.decode(out[0], skip_special_tokens=True)
        return result.strip() if result.strip() else text
    except Exception as e:
        print(f"[OfflineAI] Inference error: {e}")
        return _rule_correct(text)


def _rule_correct(text: str) -> str:
    """Rule-based fallback correction."""
    TIMING_FIXES = {
        'TA': 'TU', '5N': '5NN', 'CQCQ': 'CQ CQ',
        'DEDE': 'DE DE', 'SKSK': 'SK SK',
    }
    tokens = text.split()
    result = []
    for t in tokens:
        result.append(TIMING_FIXES.get(t, t))
    # Remove excessive CQ duplicates
    cleaned = []
    cq_count = 0
    for t in result:
        if t == 'CQ':
            cq_count += 1
            if cq_count <= 3:
                cleaned.append(t)
        else:
            cq_count = 0
            cleaned.append(t)
    return ' '.join(cleaned)


# ── Ham radio vocabulary ───────────────────────────────────────────────

PROSIGNS = {
    'CQ', 'DE', 'AR', 'SK', 'BK', 'KN', 'AS', 'TU', 'BT',
    'KA', 'CL', 'HH', 'SN', 'AA'
}

Q_CODES = {
    'QSO', 'QTH', 'QRM', 'QRN', 'QRZ', 'QSL', 'QRP', 'QRO',
    'QRT', 'QSY', 'QTR', 'QSB', 'QRX', 'QRU', 'QRV'
}

EXPLANATIONS = {
    'CQ':  'CQ = Calling all stations',
    'DE':  'DE = From / This is',
    'AR':  'AR = End of message',
    'SK':  'SK = End of contact',
    'BK':  'BK = Break / go ahead',
    'KN':  'KN = Go ahead named station only',
    'AS':  'AS = Wait / stand by',
    'TU':  'TU = Thank you',
    'BT':  'BT = Break / paragraph separator',
    '73':  '73 = Best regards',
    '88':  '88 = Love and kisses',
    '5NN': '5NN = Signal report RST 599 (perfect)',
    'K':   'K = Go ahead / over',
    'R':   'R = Roger / received',
    'QSO': 'QSO = Radio contact',
    'QTH': 'QTH = Location',
    'QRM': 'QRM = Man-made interference',
    'QRN': 'QRN = Static noise',
    'QRZ': 'QRZ = Who is calling me?',
    'QSL': 'QSL = Acknowledged',
    'QRP': 'QRP = Low power operation',
    'QRT': 'QRT = Stop transmitting',
    'QSB': 'QSB = Signal fading',
    'PSE': 'PSE = Please',
    'RPT': 'RPT = Repeat',
    'UR':  'UR = Your',
    'ES':  'ES = And',
    'FB':  'FB = Fine business (excellent)',
    'OM':  'OM = Old man (fellow operator)',
    'YL':  'YL = Young lady (female operator)',
    'HR':  'HR = Here',
    'WX':  'WX = Weather',
    'RST': 'RST = Signal report',
    'TNX': 'TNX = Thanks',
    'SOS': 'SOS = Distress signal',
    'PAN': 'PAN = Urgency signal',
    'TEST': 'TEST = Contest in progress',
    'CUL': 'CUL = See you later',
    'AGN': 'AGN = Again',
    '?':   '? = Signal too weak to decode',
    'DX':  'DX = Long distance contact',
    'VY':  'VY = Very',
    'HW':  'HW = How / How do you copy?',
    'CPY': 'CPY = Copy',
    'NR':  'NR = Number',
    'OP':  'OP = Operator name',
    'RIG': 'RIG = Radio equipment',
    'ANT': 'ANT = Antenna',
    'PWR': 'PWR = Power output',
}

PREFIXES = {
    'DL': 'Germany', 'DJ': 'Germany', 'DK': 'Germany',
    'G': 'UK', 'M': 'UK',
    'F': 'France',
    'I': 'Italy', 'IK': 'Italy',
    'W': 'USA', 'K': 'USA', 'N': 'USA',
    'VE': 'Canada',
    'JA': 'Japan',
    'UA': 'Russia', 'RA': 'Russia',
    'SP': 'Poland',
    'OK': 'Czech Republic',
    'YL': 'Latvia',
    'ON': 'Belgium',
    'OZ': 'Denmark',
    'EA': 'Spain',
    'PY': 'Brazil',
    'VK': 'Australia',
    'ZS': 'South Africa',
    'HL': 'Korea',
    'LU': 'Argentina',
}

_CALLSIGN_RE = re.compile(r'^[A-Z]{1,3}\d[A-Z0-9]{1,4}(/[A-Z0-9]+)?$')


class OfflineAI:
    """
    Offline Morse code AI corrector.
    Uses fine-tuned T5-small model trained on Morse correction pairs.
    Falls back to rule-based correction if model unavailable.
    """

    def __init__(self):
        # Load model on init
        _load_model()

    def correct(self, decoded_text: str, raw_morse: str = "") -> str:
        """Correct decoded Morse text using T5 model."""
        if not decoded_text.strip():
            return decoded_text
        return _t5_correct(decoded_text.upper().strip())

    def correct_live_signals(self, raw_text: str) -> str:
        """Correct live signal text using T5 model."""
        if not raw_text.strip():
            return raw_text
        return _t5_correct(raw_text.upper().strip())

    def explain_live_transmission(self, corrected_text: str) -> str:
        """Explain ham radio tokens in corrected text."""
        if not corrected_text.strip():
            return ""
        lines = []
        for token in corrected_text.split():
            if token in EXPLANATIONS:
                lines.append(EXPLANATIONS[token])
            elif _CALLSIGN_RE.match(token):
                country = 'Unknown'
                for prefix, c in sorted(
                    PREFIXES.items(), key=lambda x: -len(x[0])
                ):
                    if token.startswith(prefix):
                        country = c
                        break
                lines.append(f'{token} = Station from {country}')
            elif any(c.isdigit() for c in token):
                lines.append(f'{token} = Number / signal code')
            elif len(token) == 1:
                lines.append(f'{token} = Prosign / single character')
            else:
                lines.append(f'{token} = Ham radio term')
        return '\n'.join(lines)


def get_ai(groq_api_key: str = None):
    """
    Returns AIPredictor if API key available,
    otherwise returns OfflineAI as fallback.
    """
    if groq_api_key:
        try:
            from src.ai_predictor import AIPredictor
            return AIPredictor(groq_api_key)
        except Exception:
            pass
    return OfflineAI()

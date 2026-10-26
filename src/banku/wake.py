"""Wake word: decide whether an utterance was addressed to Banku.

Design note. The wake word is checked on the *transcript*, not the audio, at
least for now. That sounds backwards — you would expect keyword spotting on the
audio — but it has one large advantage: it is testable against clips we already
have, and it costs nothing to change.

What the real recordings showed (26 Oct 2026, 20 clips of a real speaker):

    "बंकू नेल कटर कुठे आहे?"      -> clear
    "बक नेलकटर कुठे आहे?"          -> dropped the nasal
    "बक नेल कटर कुठे आहे? बंकू? ..." -> repeated
    "अंकु."                        -> mangled
    "क्योंकि आज अमन भी नहीं है..."  -> ambient conversation, must be refused

So exact matching fails; ASR spelling of a name is unstable. We match
phonetically-ish with a fuzzy ratio on the opening token(s), which tolerates the
variation we actually observed. Once this is reliable we can move detection onto
the phone so audio never leaves the house before the wake word fires.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from .memory import normalize

# Variants seen in real transcripts, plus likely ones. Kept short on purpose:
# every extra variant is another way for the TV to summon the assistant.
VARIANTS = (
    # Marathi spellings, as Sarvam actually transcribed them
    "बंकू",
    "बंको",
    "बंकु",
    "बंका",
    "बंक",
    "बक",
    "अंकु",
    # English spellings. "Banku" is not an English word, so en-IN ASR guesses:
    # "bank you", "banko", "bunko". These are the plausible guesses.
    "banku",
    "bunku",
    "bunko",
    "banko",
    "bank you",
    "bunk you",
    "bankoo",
)

DEFAULT_THRESHOLD = 62


def split(text: str, threshold: int = DEFAULT_THRESHOLD) -> tuple[bool, str]:
    """Return (addressed_to_banku, utterance_without_the_wake_word).

    Only the first two tokens are considered: "बंकू" must be how the sentence
    starts. Allowing it anywhere means a Marathi TV serial discussing a bank
    will keep waking it up.

    Matching is done on folded words, but the rest of the sentence is returned
    exactly as it was spoken. Folding the whole sentence (as this used to do)
    turned "where is the TV remote" into "tv remote" — harmless in matching,
    but it reached the transcriber's answer and the model as a lowercased
    string, which is not what anyone said.
    """
    words = text.split()
    folded = [normalize(w) for w in words]
    folded = [w for w in folded if w]
    if not folded:
        return False, ""

    # Keep the two lists aligned when folded words disappear (pure punctuation).
    pairs = [(w, normalize(w)) for w in words]
    pairs = [(w, f) for w, f in pairs if f]
    words = [w for w, _ in pairs]
    toks = [f for _, f in pairs]

    if _matches(toks[0], threshold):
        return True, " ".join(words[1:]).strip()

    if len(toks) >= 2 and _matches(toks[1], threshold):
        return True, " ".join(words[2:]).strip()

    return False, ""


def _matches(token: str, threshold: int) -> bool:
    return max(fuzz.ratio(token, v) for v in VARIANTS) >= threshold


# Lines below are real ASR output from the 26 Oct capture, used as a regression
# test: `python -m banku.wake`
CASES: list[tuple[str, bool]] = [
    ("बंकू नेल कटर कुठे आहे?", True),
    ("बक नेलकटर कुठे आहे?", True),
    ("बक नेल कटर कुठे आहे? बंकू? नेल कटर कुठे आहे?", True),
    ("बंकू.", True),
    ("अंकु.", True),
    ("क्योंकि आज अमन भी नहीं है और मैं भी नहीं हूं कॉलेज में।", False),
    ("आज आपण.", False),
    ("हो.", False),
    ("पण देऊ पण शकत.", False),
    ("इथून आलाय.", False),
    ("तू देऊ पण शकते.", False),
]


if __name__ == "__main__":
    bad = 0
    for text, expected in CASES:
        hit, rest = split(text)
        mark = "ok " if hit == expected else "FAIL"
        if hit != expected:
            bad += 1
        print(f"{mark} {str(hit):5} {text!r} -> {rest!r}")
    print(f"\n{bad} failures out of {len(CASES)}")

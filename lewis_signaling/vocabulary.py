"""Vocabularies for Lewis Signaling game.

Real words for objects (targets/distractors).
Fake/alien words for messages.
"""

# Common English nouns - real, meaningful words
REAL_WORDS = [
    "apple", "banana", "cherry", "grape", "lemon", "mango", "orange", "peach",
    "book", "chair", "table", "lamp", "clock", "mirror", "window", "door",
    "bird", "cat", "dog", "fish", "horse", "mouse", "rabbit", "tiger",
    "cloud", "river", "mountain", "forest", "ocean", "desert", "island", "valley",
    "hammer", "needle", "basket", "bottle", "candle", "pillow", "blanket", "carpet",
    "doctor", "teacher", "farmer", "artist", "dancer", "singer", "writer", "driver",
    "circle", "square", "triangle", "diamond", "arrow", "spiral", "cross", "star",
    "morning", "evening", "winter", "summer", "thunder", "rainbow", "sunset", "breeze",
]

# Fake/alien words - no pre-existing meaning
ALIEN_WORDS = [
    "zorp", "blick", "munt", "frag", "quib", "vosh", "trel", "gwim",
    "plax", "snorf", "kreb", "yolt", "drix", "fump", "hask", "jund",
    "werg", "blun", "criv", "spax", "glont", "frub", "kwist", "zang",
    "threp", "blorf", "grix", "yump", "dask", "frin", "holt", "jask",
]


def get_real_words(n: int | None = None) -> list[str]:
    """Get list of real words.

    Args:
        n: If specified, return only first n words.
    """
    if n is None:
        return list(REAL_WORDS)
    return list(REAL_WORDS[:n])


def get_alien_words(n: int | None = None) -> list[str]:
    """Get list of alien words.

    Args:
        n: If specified, return only first n words.
    """
    if n is None:
        return list(ALIEN_WORDS)
    return list(ALIEN_WORDS[:n])

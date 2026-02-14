from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Configuration for Lewis Signaling game.

    Parameters:
        num_real_words: Size of vocabulary for targets/distractors (real words)
        num_alien_words: Size of vocabulary for messages (alien words)
        num_distractors: Number of distractor words shown alongside target
    """

    num_real_words: int = 32  # Size of object vocabulary
    num_alien_words: int = 16  # Size of message vocabulary
    num_distractors: int = 3  # Number of distractors (1 = binary choice)

    def __post_init__(self) -> None:
        if self.num_real_words < 2:
            raise ValueError("num_real_words must be at least 2")
        if self.num_alien_words < 2:
            raise ValueError("num_alien_words must be at least 2")
        if self.num_distractors < 1:
            raise ValueError("num_distractors must be at least 1")
        if self.num_distractors >= self.num_real_words:
            raise ValueError("num_distractors must be less than num_real_words")

    @property
    def num_candidates(self) -> int:
        """Total candidates including target."""
        return self.num_distractors + 1

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import GameConfig


def generate_sender_prompt(config: "GameConfig", alien_words: list[str]) -> str:
    """Generate system prompt for the Sender agent."""
    word_list = ", ".join(alien_words)

    return f"""You are the SENDER in a word guessing game.

## YOUR TASK

You will see:
1. A TARGET word (the one the Receiver must identify)
2. {config.num_distractors} DISTRACTOR word(s) (that the Receiver must avoid)

You must send a MESSAGE to help the Receiver identify the target.

## MESSAGE VOCABULARY

You can ONLY use one of these words as your message:
{word_list}

These are the only valid messages. The Receiver knows this same list.

## OUTPUT FORMAT

Respond with ONLY your chosen word in <answer></answer> tags. Nothing else.

<answer>zorp</answer>
"""


def generate_receiver_prompt(config: "GameConfig", alien_words: list[str]) -> str:
    """Generate system prompt for the Receiver agent."""
    word_list = ", ".join(alien_words)

    return f"""You are the RECEIVER in a word guessing game.

## YOUR TASK

You will see:
1. A MESSAGE from the Sender (one of the alien words below)
2. {config.num_candidates} CANDIDATE words (one is the target, the rest are distractors)

Your goal: identify which candidate is the target that the Sender was describing.

## MESSAGE VOCABULARY

The Sender's message is one of these words:
{word_list}

These words have no inherent meaning. You must learn what they signify through experience.

## OUTPUT FORMAT

Respond with ONLY your chosen word in <answer></answer> tags. Nothing else.

<answer>apple</answer>
"""

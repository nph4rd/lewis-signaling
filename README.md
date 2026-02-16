# Lewis-Signaling

### Overview
- **Environment ID**: `lewis-signaling`
- **Short description**: Referential communication game where a sender conveys a target word using meaningless signals
- **Tags**: multi-agent, multi-turn, cooperative

### Task
- **Type**: multi-turn
- **Parser**: XMLParser (fields: answer)
- **Rubric**: Score-based reward (0-1)

### Description

A [Lewis signaling game](https://en.wikipedia.org/wiki/Lewis_signaling_game) is a referential communication task where two agents must develop a shared language to communicate about objects. This environment implements a sender-receiver game:

  - Players: 2 (Sender and Receiver)
  - The Sender sees a target word (e.g., "apple") plus distractor words (e.g., "banana", "cherry")
  - The Sender must communicate using only pre-defined "alien words" (meaningless signals like "zorp", "blick")
  - The Receiver sees the Sender's message and a shuffled list of candidates, then must select the target
  - Perfect score: 1.0 (receiver correctly identifies the target)

This tests emergent communication: whether two agents can develop and learn a shared language to successfully communicate referentially about objects, without any pre-established meaning for the signals.

### Dependencies
- `verifiers>=0.1.8`

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval lewis-signaling
```

Configure model and sampling:

```bash
uv run vf-eval lewis-signaling -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7 -a '{"num_distractors": 4}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `1000` | Number of training examples (each with a unique seed) |
| `num_eval_examples` | int | `100` | Number of evaluation examples |
| `num_real_words` | int | `32` | Size of vocabulary for targets and distractors |
| `num_alien_words` | int | `32` | Size of vocabulary for messages (alien words) |
| `num_distractors` | int | `3` | Number of distractor words per game (creates N+1 way choice) |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Communication success (1.0 if receiver selected the target word, 0.0 otherwise) |

### Project Structure

```
lewis_signaling/
├── config.py           # GameConfig dataclass with game parameters
├── vocabulary.py       # Real words and alien words vocabularies
├── prompt.py           # System prompt templates for sender and receiver
└── lewis_signaling.py  # LewisSignalingEnv environment and reward function
```

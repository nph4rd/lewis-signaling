import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import cast

import verifiers as vf
from datasets import Dataset
from verifiers.types import Messages, State

from .config import GameConfig
from .prompt import generate_receiver_prompt, generate_sender_prompt
from .vocabulary import get_alien_words, get_real_words


class LewisSignalingEnv(vf.MultiAgentEnv):
    """Multi-agent environment for the Lewis Signaling Game.

    A referential communication game where:
    - Sender sees target word + distractor(s), chooses an alien word as message
    - Receiver sees the alien word + shuffled candidates, selects target
    """

    def __init__(
        self,
        num_train_examples: int = 1000,
        num_eval_examples: int = 100,
        num_real_words: int = 32,
        num_alien_words: int = 16,
        num_distractors: int = 1,
        **kwargs,
    ):
        self.config = GameConfig(
            num_real_words=num_real_words,
            num_alien_words=num_alien_words,
            num_distractors=num_distractors,
        )

        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples

        # Load vocabularies
        self.real_words = get_real_words(num_real_words)
        self.alien_words = get_alien_words(num_alien_words)

        # Build datasets
        train_dataset = Dataset.from_list(
            [{"question": self._get_initial_observation(seed=i), "answer": str(i)} for i in range(num_train_examples)]
        )

        eval_dataset = Dataset.from_list(
            [
                {"question": self._get_initial_observation(seed=i), "answer": str(i)}
                for i in range(num_train_examples, num_train_examples + num_eval_examples)
            ]
        )

        # Two agents: sender and receiver
        agent_ids = ["sender", "receiver"]

        super().__init__(
            dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_turns=2,
            protocol=vf.RoundRobinProtocol(agent_ids),
            **kwargs,
        )

        # Register agents
        self.register_agent(
            vf.Agent(
                id="sender",
                system_prompt=generate_sender_prompt(self.config, self.alien_words),
                is_trainable=True,
            )
        )
        self.register_agent(
            vf.Agent(
                id="receiver",
                system_prompt=generate_receiver_prompt(self.config, self.alien_words),
                is_trainable=True,
            )
        )

    def _get_initial_observation(self, seed: int) -> str:
        """Generate the initial game observation for a given seed."""
        temp_state = self._initialize_game(seed)
        return self._get_sender_observation(temp_state)

    def _initialize_game(self, seed: int) -> State:
        """Create a fresh game state from a seed."""
        rng = random.Random(seed)

        # Pick target and distractors from real words
        selected = rng.sample(self.real_words, self.config.num_candidates)
        target = selected[0]
        distractors = selected[1:]

        # Shuffle to create candidates
        candidates = list(selected)
        rng.shuffle(candidates)
        target_index = candidates.index(target)

        return cast(
            State,
            {
                "seed": seed,
                "target": target,
                "distractors": distractors,
                "candidates": candidates,
                "target_index": target_index,
                "sender_message": None,
                "receiver_choice": None,
                "is_complete": False,
                "game_won": False,
            },
        )

    def _get_sender_observation(self, state: State) -> str:
        """Generate observation for the Sender."""
        target = state["target"]
        distractors = state["distractors"]

        lines = [
            "You are the Sender. Choose a message to help the Receiver identify the TARGET.",
            "",
            f"TARGET: {target}",
            "",
        ]

        if len(distractors) == 1:
            lines.append(f"DISTRACTOR: {distractors[0]}")
        else:
            lines.append("DISTRACTORS:")
            for d in distractors:
                lines.append(f"  - {d}")

        return "\n".join(lines)

    def _get_receiver_observation(self, state: State) -> str:
        """Generate observation for the Receiver."""
        message = state.get("sender_message", "[No message]")
        candidates = state["candidates"]

        lines = [
            "You are the Receiver. Identify which candidate is the target.",
            "",
            f"MESSAGE FROM SENDER: {message}",
            "",
            "CANDIDATES:",
        ]
        for word in candidates:
            lines.append(f"  - {word}")

        return "\n".join(lines)

    def _get_final_observation(self, state: State) -> str:
        """Generate final observation with results."""
        result = {
            "game_complete": True,
            "game_won": state.get("game_won", False),
            "target": state["target"],
            "sender_message": state.get("sender_message"),
            "receiver_choice": state.get("receiver_choice"),
        }
        return f"Game Complete!\n\n{json.dumps(result, indent=2)}"

    async def setup_state(self, state: State) -> State:
        """Initialize game state."""
        state = await super().setup_state(state)
        seed = int(state["answer"])
        state.update(self._initialize_game(seed))
        state["agent_messages"] = {}
        return state

    @vf.stop
    async def game_complete(self, state: State) -> bool:
        return state.get("is_complete", False)

    async def build_agent_prompt(self, agent_id: str, state: State) -> Messages:
        """Build prompt for the given agent."""
        agent = self.get_agent(agent_id)

        if agent_id not in state["agent_messages"]:
            state["agent_messages"][agent_id] = [{"role": "system", "content": agent.system_prompt}]

        messages = state["agent_messages"][agent_id]

        if agent_id == "sender":
            observation = self._get_sender_observation(state)
        else:
            observation = self._get_receiver_observation(state)

        messages.append({"role": "user", "content": observation})
        return list(messages)

    async def on_turn_complete(self, state: State) -> None:
        """Process the action after model response."""
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        completion = last_step.get("completion", [])
        if not completion:
            return

        content = ""
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                break

        agent_id = state["extras"].get("current_agent_id")
        if agent_id and agent_id in state["agent_messages"]:
            state["agent_messages"][agent_id].append({"role": "assistant", "content": content})

        if agent_id == "sender":
            self._process_sender_turn(state, content)
        elif agent_id == "receiver":
            self._process_receiver_turn(state, content)

    def _process_sender_turn(self, state: State, content: str) -> None:
        """Process the Sender's message."""
        message = None
        match = re.search(r"<answer>\s*(\w+)\s*</answer>", content)
        if match:
            word = match.group(1).lower()
            if word in self.alien_words:
                message = word

        if message is None:
            # Invalid format - use random alien word
            rng = random.Random(state["seed"] + 1000)
            message = rng.choice(self.alien_words)

        state["sender_message"] = message

    def _process_receiver_turn(self, state: State, content: str) -> None:
        """Process the Receiver's choice."""
        choice = None
        match = re.search(r"<answer>\s*(\w+)\s*</answer>", content)
        if match:
            word = match.group(1).lower()
            if word in state["candidates"]:
                choice = word

        state["receiver_choice"] = choice
        state["is_complete"] = True
        state["game_won"] = choice is not None and choice == state["target"]

        # Set final response
        state["final_env_response"] = [{"role": "user", "content": self._get_final_observation(state)}]

        # Write trajectory
        self._write_trajectory(state)

    def _write_trajectory(self, state: State) -> None:
        """Write the full trajectory to a readable file."""
        repo_dir = Path(__file__).parent.parent
        trajectories_dir = repo_dir / "trajectories"
        trajectories_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = state.get("seed", "unknown")
        filename = f"trajectory_{timestamp}_seed{seed}.txt"
        filepath = trajectories_dir / filename

        lines = []
        lines.append("=" * 80)
        lines.append("LEWIS SIGNALING GAME - TRAJECTORY")
        lines.append("=" * 80)
        lines.append("")

        # Config
        lines.append("CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"Real words vocabulary: {self.config.num_real_words}")
        lines.append(f"Alien words vocabulary: {self.config.num_alien_words}")
        lines.append(f"Distractors: {self.config.num_distractors}")
        lines.append("")

        # Game setup
        lines.append("GAME SETUP")
        lines.append("-" * 40)
        lines.append(f"Target: {state['target']}")
        lines.append(f"Distractors: {', '.join(state['distractors'])}")
        lines.append(f"Candidates: {', '.join(state['candidates'])}")
        lines.append("")

        # Result
        lines.append("RESULT")
        lines.append("-" * 40)
        lines.append(f"Game won: {state.get('game_won', False)}")
        lines.append(f"Sender's message: {state.get('sender_message')}")
        lines.append(f"Receiver's choice: {state.get('receiver_choice')}")
        lines.append(f"Correct answer: {state['target']}")
        lines.append("")

        # Conversations
        lines.append("=" * 80)
        lines.append("CONVERSATIONS")
        lines.append("=" * 80)

        for agent_id in ["sender", "receiver"]:
            lines.append("")
            lines.append(f"{'=' * 40}")
            lines.append(agent_id.upper())
            lines.append(f"{'=' * 40}")

            messages = state.get("agent_messages", {}).get(agent_id, [])
            for msg in messages:
                role = msg.get("role", "unknown").upper()
                msg_content = msg.get("content", "")

                if role == "SYSTEM":
                    lines.append("\n[SYSTEM PROMPT]")
                    lines.append("-" * 20)
                    lines.append(msg_content[:500] + "..." if len(msg_content) > 500 else msg_content)
                elif role == "USER":
                    lines.append("\n[OBSERVATION]")
                    lines.append("-" * 20)
                    lines.append(msg_content)
                elif role == "ASSISTANT":
                    lines.append("\n[RESPONSE]")
                    lines.append("-" * 20)
                    lines.append(msg_content)

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF TRAJECTORY")
        lines.append("=" * 80)

        filepath.write_text("\n".join(lines))
        print(f"\nTrajectory written to: {filepath}")


def success_reward_func(parser, completion: Messages, **_kwargs) -> float:
    """Reward: 1 if receiver selected target, 0 otherwise."""
    for msg in reversed(parser.get_user_messages(completion)):
        content = msg.get("content", "")
        if isinstance(content, str) and "game_complete" in content.lower():
            try:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    data = json.loads(content[json_start:json_end])
                    return 1.0 if data.get("game_won") else 0.0
            except (json.JSONDecodeError, ValueError):
                continue
    return 0.0


def load_environment(
    num_train_examples: int = 1000,
    num_eval_examples: int = 100,
    num_real_words: int = 32,
    num_alien_words: int = 16,
    num_distractors: int = 3,
) -> vf.Environment:
    """Load the Lewis Signaling environment.

    Args:
        num_train_examples: Number of training examples
        num_eval_examples: Number of evaluation examples
        num_real_words: Size of real word vocabulary (targets/distractors)
        num_alien_words: Size of alien word vocabulary (messages)
        num_distractors: Number of distractors per game

    Returns:
        Configured LewisSignalingEnv instance
    """
    parser = vf.XMLParser(fields=["answer"], answer_field="answer")
    rubric = vf.Rubric(parser=parser)

    rubric.add_reward_func(success_reward_func, weight=0.9)

    format_reward = parser.get_format_reward_func()
    format_reward.__name__ = "format_reward"
    rubric.add_reward_func(format_reward, weight=0.1)

    return LewisSignalingEnv(
        parser=parser,
        rubric=rubric,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        num_real_words=num_real_words,
        num_alien_words=num_alien_words,
        num_distractors=num_distractors,
    )

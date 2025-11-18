import json
import re
import logging
import textarena as ta
from prompts import BASE_PROMPT, BELIEF_PROMPT, DECISION_PROMPT, QUESTION_PROMPT, EIG_QUESTION_PROMPT, MOVE_PROMPT, CONSISTENCY_PROMPT
import numpy as np
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.resources
import random

logger = logging.getLogger(__name__)

# Configuration flags
ENABLED_PROMPT_TYPES = set()  # Set of prompt types to show (e.g., {'DECISION', 'QUESTION'})

EPSILON = 0.1  # Noise parameter for answers
TOP_N_BELIEFS = 10  # Number of top beliefs to show in prompts
BLANK_HISTORY_PLACEHOLDER = "(no history yet)"
ANSWER_REGEX = re.compile(r'<answer>(.*?)</answer>', re.IGNORECASE | re.DOTALL)

player_dict = {-1: "GAME", 0: "PLAYER"}

def log_prompt(label: str, prompt):
    """
    Log a prompt with consistent formatting if the label is in ENABLED_PROMPT_TYPES.

    Args:
        label: Description of the prompt (e.g., "DECISION", "QUESTION")
        prompt: Either a string or a list of message dicts
    """
    if label not in ENABLED_PROMPT_TYPES:
        return

    # Format the prompt based on its type
    if isinstance(prompt, str):
        prompt_text = prompt
    elif isinstance(prompt, list):
        # Handle list of messages (e.g., [{"role": "system", "content": "..."}, ...])
        prompt_text = "\n".join(f"[{msg.get('role', 'unknown')}] {msg.get('content', '')}" for msg in prompt)
    else:
        prompt_text = str(prompt)

    logger.info(f"{'=' * 60}\n{label} PROMPT\n{'=' * 60}\n{prompt_text}\n{'=' * 60}")

class DecisionType:
    QUESTION = "question"
    GUESS = "guess"

def binary_entropy(p: float) -> float:
    """
    Calculate the binary channel entropy given a probability p.
    Returns NaN if p is not in [0, 1].
    """
    if p < 0 or p > 1:
        return float("nan")
    elif p == 0 or p == 1:
        return 0.0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def shannon_entropy(probabilities: list) -> float:
    """
    Calculate Shannon entropy for a probability distribution.
    H = -sum(p_i * log2(p_i)) for all i where p_i > 0
    """
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

class LLMAgent(ta.agents.OpenRouterAgent):
    def __init__(self, openrouter_agent: ta.agents.OpenRouterAgent, ground_truth_theme: str):
        super().__init__(model_name=openrouter_agent.model_name)
        self.openrouter_agent = openrouter_agent
        self.ground_truth_theme = ground_truth_theme

        # Load words using importlib.resources for consistency with env.py
        with importlib.resources.files('textarena.envs.TwentyQuestions').joinpath('twenty_questions_words.json').open('r') as f:
            word_data = json.load(f)

        self.word_data = word_data["basic"][self.ground_truth_theme]

        # Initialize belief tracking (passive for all agents)
        self.sampling_agent = ta.agents.OpenRouterAgent(model_name=self.openrouter_agent.model_name)
        self._initialize_beliefs()
        self.last_processed_turn = 0
        self.simulated_answers_cache = {}

        # Initialize diagnostics tracking
        self.diagnostics = {
            'turns': [],  # List of turn data
            'llm_calls': [],  # All LLM prompt/completion pairs
            'current_turn': 0
        }

    def get_diagnostics(self):
        """Return collected diagnostics data"""
        return self.diagnostics

    def _record_llm_call(self, call_type: str, prompt: str, completion: str, metadata: dict = None):
        """Record an LLM call with its prompt and completion"""
        call_data = {
            'type': call_type,
            'prompt': prompt,
            'completion': completion,
            'turn': self.diagnostics['current_turn']
        }
        if metadata:
            call_data['metadata'] = metadata
        self.diagnostics['llm_calls'].append(call_data)

    def format_history(self, history: list) -> str:
        serialized_history = ""
        for entry in history:
            player = player_dict[entry["player"]]
            serialized_history += f"[{player}] {entry['message']}\n"
        return serialized_history

    def get_base_prompt(self):
        """Get the base prompt template. Override in subclasses for different prompts."""
        return BASE_PROMPT

    def _get_sorted_beliefs(self):
        """
        Get belief distribution sorted by probability (descending) with random tiebreaking.
        Uses deterministic seeding so the same belief state always produces the same order.
        Returns list of (word, probability) tuples.
        """
        # Create a deterministic seed from the belief distribution
        # Sort by word name first to ensure consistent ordering for hashing
        sorted_for_hash = sorted(self.belief_distribution.items())
        # Create a hash of the probabilities (rounded to avoid floating point issues)
        seed_str = ''.join(f"{word}:{prob:.10f}" for word, prob in sorted_for_hash)
        seed = hash(seed_str) % (2**32)

        # Create a local Random instance with this seed
        local_random = random.Random(seed)

        # Sort with deterministic random tiebreaking
        return sorted(self.belief_distribution.items(), key=lambda x: (-x[1], local_random.random()))

    def _format_belief_state(self, top_n: int = None):
        """
        Format belief state for prompts. Used by Bayesian agents.
        Args:
            top_n: Number of top beliefs to show (defaults to TOP_N_BELIEFS)
        """
        if top_n is None:
            top_n = TOP_N_BELIEFS

        sorted_beliefs = self._get_sorted_beliefs()[:top_n]
        return "\n".join(f"{i+1}. {word}: {prob:.3f}" for i, (word, prob) in enumerate(sorted_beliefs))

    def __call__(self, game_history: list) -> str:
        """Main method called by TextArena environment"""
        # Increment turn counter
        self.diagnostics['current_turn'] += 1

        # Initialize turn diagnostics
        turn_data = {
            'turn_number': self.diagnostics['current_turn']
        }

        # Add turn data early so question() method can access and modify it
        self.diagnostics['turns'].append(turn_data)

        # Process new question-answer pairs to update beliefs (passive tracking for all agents)
        self._process_history_for_beliefs(game_history)

        # Record belief state before asking this turn's question (after incorporating previous answer)
        turn_data['belief_state_before_question'] = dict(self._get_sorted_beliefs())
        turn_data['entropy'] = shannon_entropy(list(self.belief_distribution.values()))

        # Update game history
        formatted_history = self.format_history(game_history)

        # Calculate remaining questions (count player questions, max 20)
        player_questions = len([entry for entry in game_history if entry["player"] == 0])
        remaining_questions = max(0, 20 - player_questions)

        decision = self.decision(formatted_history, remaining_questions)
        turn_data['decision'] = decision

        if DecisionType.GUESS == decision or remaining_questions == 0:
            action = self.move(formatted_history, remaining_questions=remaining_questions)
            turn_data['action_type'] = 'guess'
        else:
            action = self.question(history=formatted_history, remaining_questions=remaining_questions)
            turn_data['action_type'] = 'question'

        turn_data['action'] = action

        return action

    def decision(self, history: str, remaining_questions: int, max_retries: int = 10) -> str:
        """Ask if the agent wants to ask more questions or try to guess"""
        context = self.get_base_prompt().format(
            word_list=self.word_data,
            history=history,
            remaining_questions=remaining_questions,
            belief_state=self._format_belief_state(),
        )
        prompt = DECISION_PROMPT.format(context=context)

        log_prompt("DECISION", prompt)
        for _ in range(max_retries):
            response = self.openrouter_agent(prompt)

            # Record LLM call
            self._record_llm_call('decision', prompt, response)

            match = ANSWER_REGEX.search(response)

            if match:
                decision = match.group(1).strip().lower()
                if decision == "question":
                    return DecisionType.QUESTION
                elif decision == "guess":
                    return DecisionType.GUESS
            logger.warning(f"Attempt {_} failed to parse decision from response")
        else:
            raise ValueError(f"Unexpected decision: {response}")

    def _process_question(self, question: str, history: str, remaining_questions: int):
        """
        Process a question by calculating its EIG and storing diagnostics.
        Returns tuple of (question, eig).
        """
        eig = self._calculate_eig(question, history, remaining_questions)
        # Store EIG value
        self.diagnostics['turns'][-1]['eig_values'][question] = eig
        # Store simulated answers (already cached)
        if question in self.simulated_answers_cache:
            self.diagnostics['turns'][-1]['simulated_answers'][question] = dict(self.simulated_answers_cache[question])
        return (question, eig)

    def question(self, history: str, remaining_questions: int = 20, max_retries: int = 10) -> str:
        """Ask the agent for a question"""
        context = self.get_base_prompt().format(
            word_list=self.word_data,
            history=history,
            remaining_questions=remaining_questions,
            belief_state=self._format_belief_state(),
        )
        prompt = QUESTION_PROMPT.format(context=context)
        log_prompt("QUESTION", prompt)

        # Initialize turn-specific EIG tracking
        self.diagnostics["turns"][-1]["candidate_questions"] = []
        self.diagnostics["turns"][-1]["eig_values"] = {}
        self.diagnostics["turns"][-1]["simulated_answers"] = {}

        for _ in range(max_retries):
            response = self.openrouter_agent(prompt)

            # Record LLM call
            self._record_llm_call('question', prompt, response)

            match = ANSWER_REGEX.search(response)
            if match:
                question = match.group(1).strip()
                self._process_question(question, history, remaining_questions)
                return question
            logger.warning(f"Attempt {_} failed to parse question from response")
        else:
            raise ValueError(f"Unexpected response: {response}")

    def move(self, history: str, remaining_questions: int = 20, max_retries: int = 10) -> str:
        """Ask the agent for a move (final guess)"""
        context = self.get_base_prompt().format(
            word_list=self.word_data,
            history=history,
            remaining_questions=remaining_questions,
            belief_state=self._format_belief_state(),
        )

        prompt = MOVE_PROMPT.format(context=context, word_list=self.word_data)

        log_prompt("MOVE", prompt)
        for _ in range(max_retries):
            response = self.openrouter_agent(prompt)

            # Record LLM call
            self._record_llm_call('move', prompt, response)

            # Extract move using regex for <answer></answer> tags
            match = ANSWER_REGEX.search(response)
            if match:
                answer = match.group(1).strip()
                # Ensure the answer is wrapped in brackets for the game format
                if not (answer.startswith('[') and answer.endswith(']')):
                    answer = f"[{answer}]"
                return answer
            logger.warning(f"Attempt {_} failed to parse move from response")
        else:
            raise ValueError(f"Unexpected move: {response}")

    # === Belief Tracking Methods (Passive for all agents) ===

    def _initialize_beliefs(self):
        """Initialize uniform belief distribution over word list"""
        n_words = len(self.word_data)
        uniform_prob = 1.0 / n_words
        self.belief_distribution = {word.lower().strip(): uniform_prob for word in self.word_data}
        logger.info(f"Initialized uniform belief distribution over {n_words} words")

    def _process_history_for_beliefs(self, game_history: list):
        """Process game history to update beliefs based on new Q&A pairs"""
        # Extract only new entries since last processed turn
        new_entries = game_history[self.last_processed_turn:]

        formatted_history = self.format_history(game_history[:self.last_processed_turn])

        # Look for player question -> gamemaster answer pairs
        i = 0
        while i < len(new_entries):
            entry = new_entries[i]

            # Check if this is a player's question (player 0)
            if entry["player"] == 0:
                # Look for the next gamemaster response (player -1)
                if i + 1 < len(new_entries) and new_entries[i + 1]["player"] == -1:
                    question = entry["message"]
                    answer = new_entries[i + 1]["message"]

                    # Update beliefs based on this Q&A pair
                    self._update_belief(question, answer, formatted_history)

                    # Update formatted history for next iteration
                    formatted_history += f"[PLAYER] {question}\n[GAME] {answer}\n"

                    i += 2  # Skip the answer we just processed
                    continue

            i += 1

        # Update the last processed turn marker
        self.last_processed_turn = len(game_history)

    def _update_belief(self, question: str, actual_answer: str, history):
        """
        Update belief distribution based on question and actual answer received.
        Uses Bayesian update: p(word|answer) ∝ p(word) * p(answer|word)
        where p(answer|word) accounts for noise (EPSILON):
        - If simulated_answer == actual_answer: p(answer|word) = (1 - EPSILON)
        - If simulated_answer != actual_answer: p(answer|word) = EPSILON / 2
        """
        logger.info(f"BELIEF UPDATE | Q: '{question}' → A: '{actual_answer}'")

        # Record the answer in the diagnostics of the turn where the question was asked
        # Current turn is T, question was asked in turn T-1, so index is current_turn - 2
        if self.diagnostics['current_turn'] >= 2:
            question_turn_idx = self.diagnostics['current_turn'] - 2
            if question_turn_idx >= 0 and question_turn_idx < len(self.diagnostics['turns']):
                self.diagnostics['turns'][question_turn_idx]['answer_received'] = actual_answer

        # Normalize actual answer
        actual_answer = actual_answer.lower().strip()

        # Check if we have cached simulated answers for this question
        if question not in self.simulated_answers_cache:
            raise ValueError(f"Question not in cache: {question}")

        simulated_answers = self.simulated_answers_cache[question]

        # Show simulated answers
        logger.info(f"Simulated answers: {simulated_answers}")

        # Calculate new unnormalized probabilities
        new_beliefs = {}
        for word, prob in self.belief_distribution.items():
            # Get simulated answer from cache
            if word in simulated_answers:
                simulated_answer = simulated_answers[word]

                # Update probability with noise model
                # p(answer|word) = (1-EPSILON) if match, EPSILON/2 if no match
                if simulated_answer == actual_answer:
                    # Simulated answer matches actual answer
                    likelihood = 1.0 - EPSILON
                else:
                    # Simulated answer doesn't match (noise case)
                    likelihood = EPSILON / 2

                new_beliefs[word] = prob * likelihood
            else:
                # If word not in simulated answers (shouldn't happen), set to 0
                raise ValueError(f"No simulated answer for word '{word}'")

        # Renormalize
        total_prob = sum(new_beliefs.values())

        if total_prob == 0:
            logger.warning("All probabilities became 0! Resetting to uniform.")
            self._initialize_beliefs()
            return

        self.belief_distribution = {word: p / total_prob for word, p in new_beliefs.items()}

        # Show top beliefs (with random tiebreaking)
        top_words = self._get_sorted_beliefs()
        beliefs_str = ", ".join(f"{w}:{p:.2f}" for w, p in top_words)
        entropy = shannon_entropy(list(self.belief_distribution.values()))
        logger.info(f"Top beliefs: [{beliefs_str}] | Entropy: {entropy:.3f}")

    def _simulate_answer(self, question: str, words: list, history, remaining_questions: int = 20, max_retries: int = 10):
        """
        Simulate what answers the gamemaster would give for all words in a single LLM call.
        Returns a dict mapping word -> answer
        """
        # Check cache first
        cache_key = question
        if cache_key in self.simulated_answers_cache:
            logger.debug(f"Using cached simulated answers for question: '{question}'")
            return self.simulated_answers_cache[cache_key]

        formatted_history = history
        # Use BASE_PROMPT directly for consistency checks (not belief-aware)
        context = BASE_PROMPT.format(word_list=self.word_data, history=formatted_history, remaining_questions=remaining_questions)
        prompt = CONSISTENCY_PROMPT.format(context=context, question=question)

        log_prompt("CONSISTENCY", prompt)
        for _ in range(max_retries):
            logger.debug(f"[Attempt {_}] Simulating answers for {len(words)} words and question: {question}")
            response = self.sampling_agent(prompt)

            # Record LLM call
            self._record_llm_call('consistency', prompt, response, {
                'question': question,
                'num_words': len(words)
            })

            # Extract answer using regex for <answer></answer> tags
            match = ANSWER_REGEX.search(response)
            if match:
                try:
                    dict_content = match.group(1).strip()
                    # Try JSON parsing first
                    try:
                        raw_dict = json.loads(dict_content)
                    except json.JSONDecodeError:
                        # Fallback to ast.literal_eval for Python dict format
                        raw_dict = ast.literal_eval(dict_content)

                    # Normalize all answers to lowercase
                    simulated_answers = {}
                    for key, value in raw_dict.items():
                        # Normalize key (word) to lowercase
                        normalized_key = str(key).lower().strip()
                        # Normalize value (answer) to lowercase
                        normalized_value = str(value).lower().strip()

                        # Validate that answer is yes, no, or i don't know
                        if normalized_value not in ["yes", "no", "i don't know"]:
                            logger.warning(f"Invalid answer '{value}' for word '{key}', skipping")
                            continue

                        simulated_answers[normalized_key] = normalized_value

                    # Validate that we have answers for all words
                    missing_words = [word for word in words if word.lower().strip() not in simulated_answers]
                    if missing_words:
                        raise ValueError(f"Missing simulated answers for words: {missing_words}")

                    # Validate that all words in simulated_answers are in the original words list
                    extra_words = [word for word in simulated_answers.keys() if word not in [word.lower().strip() for word in words]]
                    if extra_words:
                        raise ValueError(f"Simulated answers contain extra words not in original list: {extra_words}")

                    logger.debug(
                        f"Simulated {len(simulated_answers)} answers successfully: {simulated_answers}"
                    )

                    # Cache the results
                    self.simulated_answers_cache[cache_key] = simulated_answers

                    return simulated_answers

                except Exception as e:
                    logger.warning(f"Error parsing simulated answers from tags: {e}")
                    logger.warning(f"Tag content was: {dict_content}")
            logger.warning(f"Simulation attempt {_} failed to parse answers from response")
        else:
            raise ValueError(f"Failed to simulate answers after {max_retries} attempts")

    def _calculate_eig(self, question: str, history, remaining_questions):
        """
        Calculate Expected Information Gain for a question.
        EIG = H(current_beliefs) - E[H(beliefs | answer)]
        """
        # Calculate current entropy
        current_probs = list(self.belief_distribution.values())
        current_entropy = shannon_entropy(current_probs)

        logger.debug(f"Calculating EIG for: '{question}'")

        # Get all words
        words_to_simulate = list(self.belief_distribution.keys())

        # Simulate answers for all words in a single batch call
        simulated_answers = self._simulate_answer(question, words_to_simulate, history, remaining_questions=remaining_questions)

        # Calculate expected posterior entropy
        expected_posterior_entropy = 0.0

        for answer in ["yes", "no", "i don't know"]:
            # Calculate posterior distribution for this hypothetical answer
            # This is the same computation as _update_belief, but without actually updating
            posterior_distribution = {}

            for word, prob in self.belief_distribution.items():
                simulated_answer = simulated_answers[word]
                # p(word | answer) ∝ p(word) × p(answer | word)
                # where p(answer | word) = 1 if simulated_answer == answer, else 0
                if simulated_answer == answer:
                    posterior_distribution[word] = prob * (1.0 - EPSILON)
                else:
                    posterior_distribution[word] = prob * (EPSILON / 2)

            # Calculate p(answer) = sum of unnormalized posterior probabilities
            p_answer = sum(posterior_distribution.values())

            if p_answer == 0:
                # This answer is impossible given current beliefs
                continue

            # Normalize the posterior distribution
            posterior_distribution = {word: p / p_answer for word, p in posterior_distribution.items()}

            # Calculate posterior entropy over the full distribution
            posterior_probs = list(posterior_distribution.values())
            posterior_entropy = shannon_entropy(posterior_probs)

            # Add to expected posterior entropy
            expected_posterior_entropy += p_answer * posterior_entropy

            n_nonzero = sum(1 for p in posterior_probs if p > 0)
            logger.debug(f"  Answer '{answer}': p={p_answer:.3f}, n_words={n_nonzero}, H(posterior)={posterior_entropy:.3f}")

        # EIG is the reduction in entropy
        eig = current_entropy - expected_posterior_entropy

        logger.debug(f"  EIG={eig:.3f} for '{question}'")

        return eig

    def _move_belief(self) -> str:
        """
        Helper method to make a final guess based on the belief distribution.
        Returns the word with maximum probability (random tiebreaking if multiple words have max prob).
        """
        # Find maximum probability
        max_prob = max(self.belief_distribution.values())

        # Find all words with maximum probability
        max_words = [word for word, prob in self.belief_distribution.items() if prob == max_prob]

        # Randomly select among tied words
        word = random.choice(max_words)
        prob = max_prob

        # Show top 5 candidates (with random tiebreaking)
        top_5 = self._get_sorted_beliefs()[:5]
        candidates_str = ", ".join(f"{w}:{p:.2f}" for w, p in top_5)

        logger.info("=" * 60)
        logger.info(f"FINAL GUESS (belief-based)")
        logger.info(f"Selected: '{word}' (p={prob:.4f})")
        if len(max_words) > 1:
            logger.info(f"Tied words ({len(max_words)}): {max_words}")
        logger.info(f"Top 5: [{candidates_str}]")
        logger.info("=" * 60)

        # Return in the correct format with brackets
        return f"<answer>[{word}]</answer>"


class EIGQuestionMixin:
    """Mixin that provides EIG-based question selection"""

    def _sort_questions_by_eig(self, question_list):
        """
        Sort questions by EIG (descending) with random tiebreaking.
        Uses deterministic seeding so the same question set always produces the same order.
        Args:
            question_list: List of (question, eig) tuples
        Returns:
            Sorted list of (question, eig) tuples
        """
        # Create a deterministic seed from the question list
        # Sort by question text first to ensure consistent ordering for hashing
        sorted_for_hash = sorted(question_list, key=lambda x: x[0])
        # Create a hash of the questions and EIG values
        seed_str = ''.join(f"{q}:{eig:.10f}" for q, eig in sorted_for_hash)
        seed = hash(seed_str) % (2**32)

        # Create a local Random instance with this seed
        local_random = random.Random(seed)

        # Sort with deterministic random tiebreaking
        return sorted(question_list, key=lambda x: (-x[1], local_random.random()))

    def question(self, history, remaining_questions=20, max_retries: int = 5) -> str:
        """Generate question with highest Expected Information Gain"""
        # Calculate and log current entropy once
        current_probs = list(self.belief_distribution.values())
        current_entropy = shannon_entropy(current_probs)

        logger.info("=" * 60)
        logger.info(f"QUESTION GENERATION (remaining: {remaining_questions})")
        logger.info(f"Current entropy: {current_entropy:.3f}")
        logger.info("=" * 60)

        # Initialize turn-specific EIG tracking
        self.diagnostics['turns'][-1]['candidate_questions'] = []
        self.diagnostics['turns'][-1]['eig_values'] = {}
        self.diagnostics['turns'][-1]['simulated_answers'] = {}

        for _ in range(max_retries):
            # Generate k questions in a single batch
            formatted_history = history
            context = self.get_base_prompt().format(
                word_list=self.word_data,
                history=formatted_history,
                remaining_questions=remaining_questions,
                belief_state=self._format_belief_state(),
            )

            questions = self._generate_batch_questions(context, remaining_questions, self.k, max_retries)

            if not questions:
                continue

            # Record candidate questions
            self.diagnostics['turns'][-1]['candidate_questions'] = questions

            # Calculate EIG for all questions using the shared _process_question method
            logger.info(f"Evaluating {len(questions)} candidate questions...")
            with ThreadPoolExecutor(max_workers=max(len(questions), 4)) as executor:
                question_list = list(executor.map(
                    lambda q: self._process_question(q, history, remaining_questions),
                    questions
                ))

            sorted_questions = self._sort_questions_by_eig(question_list)

            logger.info(f"Top {min(10, len(question_list))} questions by EIG:")
            for i, (q, eig) in enumerate(sorted_questions[:10]):
                logger.info(f"  {i+1}. EIG={eig:.3f} | {q}")

            best_question, best_eig = sorted_questions[0]
            self.diagnostics['turns'][-1]['selected_question'] = best_question
            self.diagnostics['turns'][-1]['selected_eig'] = best_eig

            logger.info(f"✓ Selected (EIG={best_eig:.3f}): {best_question}")
            logger.info("=" * 60)
            return best_question
        else:
            raise ValueError(
                f"Failed to generate valid questions after {max_retries} attempts"
            )

    def _generate_batch_questions(self, context: str, remaining_questions: int, k: int, max_retries: int) -> list:
        """Generate k questions in a single batch using EIG_QUESTION_PROMPT"""

        prompt = EIG_QUESTION_PROMPT.format(
            context=context,
            k=k,
        )

        log_prompt("QUESTION", prompt)
        for _ in range(max_retries):
            response = self.openrouter_agent(prompt)

            # Record LLM call
            self._record_llm_call('batch_question', prompt, response, {'k': k})

            match = ANSWER_REGEX.search(response)

            if match:
                try:
                    dict_content = match.group(1).strip()
                    # Try JSON parsing first
                    questions_dict = json.loads(dict_content)
                    questions = [q.strip() for q in questions_dict.values()]
                    logger.debug(f"Generated {len(questions)} batch questions")
                    return questions
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for Python dict format
                    try:
                        questions_dict = ast.literal_eval(dict_content)
                        questions = [q.strip() for q in questions_dict.values()]
                        logger.debug(f"Generated {len(questions)} batch questions (via ast)")
                        return questions
                    except Exception as e:
                        logger.warning(f"Error parsing batch questions from tags: {e}")
                        logger.warning(f"Tag content was: {dict_content}")
                except Exception as e:
                    logger.warning(f"Error parsing batch questions from tags: {e}")
                    logger.warning(f"Tag content was: {dict_content}")
            logger.warning(f"Attempt {_} failed to make batch responses")

        logger.warning(f"Failed to generate batch questions after {max_retries} attempts")
        return []


class BayesMAgent(LLMAgent):
    """LLM questions + Belief-based moves"""

    def get_base_prompt(self):
        """Use BELIEF_PROMPT to show belief state in all prompts"""
        return BELIEF_PROMPT

    def move(self, history: str, remaining_questions: int) -> str:
        """Make final guess based on belief distribution"""
        return self._move_belief()


class BayesQAgent(EIGQuestionMixin, LLMAgent):
    """EIG questions + LLM moves"""

    def __init__(self, openrouter_agent: ta.agents.OpenRouterAgent, ground_truth_theme: str, k: int = 10):
        LLMAgent.__init__(self, openrouter_agent, ground_truth_theme)
        self.k = k

    def get_base_prompt(self):
        """Use BELIEF_PROMPT to show belief state in all prompts"""
        return BELIEF_PROMPT


class BayesQMAgent(EIGQuestionMixin, LLMAgent):
    """EIG questions + Belief-based moves"""

    def __init__(self, openrouter_agent: ta.agents.OpenRouterAgent, ground_truth_theme: str, k: int = 10):
        LLMAgent.__init__(self, openrouter_agent, ground_truth_theme)
        self.k = k

    def get_base_prompt(self):
        """Use BELIEF_PROMPT to show belief state in all prompts"""
        return BELIEF_PROMPT

    def move(self, history: str, remaining_questions: int) -> str:
        """Make final guess based on belief distribution"""
        return self._move_belief()

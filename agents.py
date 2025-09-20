import json
import re
import textarena as ta
from prompts import BASE_PROMPT, SAMPLES_PROMPT, DECISION_PROMPT, QUESTION_PROMPT, EIG_QUESTION_PROMPT, MOVE_PROMPT, CONSISTENCY_PROMPT
import numpy as np
import ast
from concurrent.futures import ThreadPoolExecutor

EPSILON = 0.1  # Noise parameter for answers
BLANK_HISTORY_PLACEHOLDER = "(no history yet)"
ANSWER_REGEX = re.compile(r'<answer>(.*?)</answer>', re.IGNORECASE | re.DOTALL)

player_dict = {-1: "GAME", 0: "PLAYER"}

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

class LLMAgent(ta.agents.OpenRouterAgent):
    def __init__(self, openrouter_agent: ta.agents.OpenRouterAgent, ground_truth_theme: str):
        super().__init__(model_name=openrouter_agent.model_name)
        self.openrouter_agent = openrouter_agent
        self.ground_truth_theme = ground_truth_theme

    def format_history(self, history: list) -> str:
        serialized_history = ""
        for entry in history:
            player = player_dict[entry["player"]]
            serialized_history += f"[{player}] {entry['message']}\n"
        return serialized_history

    def __call__(self, game_history: list) -> str:
        """Main method called by TextArena environment"""
        # Update game history
        formatted_history = self.format_history(game_history)

        # Calculate remaining questions (count player questions, max 20)
        player_questions = len([entry for entry in game_history if entry["player"] == 0])
        remaining_questions = max(0, 20 - player_questions)

        decision = self.decision(formatted_history, remaining_questions)

        if DecisionType.GUESS == decision or remaining_questions == 0:
            action = self.move(formatted_history)
        else:
            action = self.question(history=formatted_history, remaining_questions=remaining_questions)

        return action
    
    def decision(self, history: str, remaining_questions: int, max_retries: int = 10) -> str:
        """Ask if the agent wants to ask more questions or try to guess"""
        context = BASE_PROMPT.format(history=history)
        prompt = DECISION_PROMPT.format(context=context, remaining_questions=remaining_questions)

        for _ in range(max_retries):
            response = self.openrouter_agent(prompt)
            match = ANSWER_REGEX.search(response)

            if match:
                decision = match.group(1).strip().lower()
                if decision == "question":
                    return DecisionType.QUESTION
                elif decision == "guess":
                    return DecisionType.GUESS
            print(f"Attempt {_} failed to parse decision from response")
        else:
            raise ValueError(f"Unexpected decision: {response}")

    def question(self, history: str, remaining_questions: int = 20, max_retries: int = 10) -> str:
        """Ask the agent for a question"""
        context = BASE_PROMPT.format(history=history)
        prompt = QUESTION_PROMPT.format(context=context, remaining_questions=remaining_questions)

        for _ in range(max_retries):
            response = self.openrouter_agent(prompt)
            match = ANSWER_REGEX.search(response)
            if match:
                return match.group(1).strip()
            print(f"Attempt {_} failed to parse question from response")
        else:
            raise ValueError(f"Unexpected response: {response}")
    
    def move(self, history: str, max_retries: int = 10) -> str:
        """Ask the agent for a move (final guess)"""
        context = BASE_PROMPT.format(history=history)
        prompt = MOVE_PROMPT.format(context=context)

        for _ in range(max_retries):
            response = self.openrouter_agent(prompt)
            # Extract move using regex for <answer></answer> tags
            match = ANSWER_REGEX.search(response)
            if match:
                answer = match.group(1).strip()
                # Ensure the answer is wrapped in brackets for the game format
                if not (answer.startswith('[') and answer.endswith(']')):
                    answer = f"[{answer}]"
                return answer
            print(f"Attempt {_} failed to parse move from response")
        else:
            raise ValueError(f"Unexpected move: {response}")

class EIGAgent(LLMAgent):
    def __init__(self, openrouter_agent: ta.agents.OpenRouterAgent, ground_truth_theme: str):
        super().__init__(openrouter_agent=openrouter_agent, ground_truth_theme=ground_truth_theme)
        self.openrouter_agent = openrouter_agent
        self.sampling_agent = ta.agents.OpenRouterAgent(model_name="openai/gpt-5")  # Dedicated GPT-5 for sampling
        self.ground_truth_theme = ground_truth_theme

    def _generate_fresh_samples(self, history, max_retries: int = 10):
        """Generate fresh samples consistent with current game history"""
        formatted_history = history

        context = BASE_PROMPT.format(history=formatted_history)

        prompt = SAMPLES_PROMPT.format(context=context, theme=self.ground_truth_theme)
        for _ in range(max_retries):
            response = self.sampling_agent(prompt)

            # Extract samples using regex for <answer></answer> tags
            match = ANSWER_REGEX.search(response)
            
            if match:
                try:
                    dict_content = match.group(1).strip()
                    # Try JSON parsing first
                    samples_dict = json.loads(dict_content)
                    # Extract values from dictionary format {1: "coconut", 2: "tomato", ...}
                    samples = [obj.lower().strip() for obj in samples_dict.values()]
                    print(f"Generated {len(samples)} fresh samples for EIG calculation")
                    print(f"Sampled objects: {samples}")
                    return samples
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for Python dict format
                    try:
                        samples_dict = ast.literal_eval(dict_content)
                        samples = [obj.lower().strip() for obj in samples_dict.values()]
                        print(f"Generated {len(samples)} fresh samples for EIG calculation (via ast)")
                        return samples
                    except Exception as e:
                        print(f"Error parsing fresh samples from tags: {e}")
                        print(f"Tag content was: {dict_content}")
                except Exception as e:
                    print(f"Error parsing fresh samples from tags: {e}")
                    print(f"Tag content was: {dict_content}")
            print(f"Sampling attempt {_} failed to parse samples from response")
        else:
            raise ValueError(f"Failed to generate fresh samples after {max_retries} attempts")

    def _get_consistency_dict(self, question: str, samples: list, history, max_retries: int = 10):
        formatted_history = history #self.format_history(history)
        context = BASE_PROMPT.format(history=formatted_history)
        prompt = CONSISTENCY_PROMPT.format(context=context, question=question, objects=samples)

        for _ in range(max_retries):
            response = self.sampling_agent(prompt)

            # Extract consistency dict using regex for <answer></answer> tags
            match = ANSWER_REGEX.search(response)
            if match:
                try:
                    dict_content = match.group(1).strip()
                    # Try JSON parsing first
                    consistency_dict = json.loads(dict_content)
                    return consistency_dict
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for Python dict format
                    try:
                        consistency_dict = ast.literal_eval(dict_content)
                        return consistency_dict
                    except Exception as e:
                        print(f"Error parsing consistency dictionary from tags: {e}")
                        print(f"Tag content was: {dict_content}")
                except Exception as e:
                    print(f"Error parsing consistency dictionary from tags: {e}")
                    print(f"Tag content was: {dict_content}")
            print(f"Consistency attempt {_} failed to parse samples from response")
        else:
            raise ValueError(f"Failed to generate fresh samples after {max_retries} attempts")

    def _calculate_eig(self, consistency_dict, samples: list):
        results = {"yes": 0, "no": 0}

        for obj in samples:
            if obj in consistency_dict:
                answer = consistency_dict[obj]
                if answer == "yes":
                    results["yes"] += 1
                elif answer == "no":
                    results["no"] += 1
                else:
                    print(f"Unexpected answer '{answer}' for object '{obj}'")
                    return float("nan")

        if any(v == 0 for v in results.values()):
            return 0

        # Calculate EIG using equal probabilities for all samples
        total_count = sum(results.values())
        p_true = results["yes"] / total_count

        return binary_entropy(
            EPSILON + ((1 - 2 * EPSILON) * p_true)
        ) - binary_entropy(EPSILON)

    def question(self, history, remaining_questions=20, k=10, max_retries: int = 5) -> str:
        # Generate fresh samples consistent with current history
        for _ in range(max_retries):
            samples = self._generate_fresh_samples(history)

            # Generate k questions in a single batch
            formatted_history = history #self.format_history(history)
            context = BASE_PROMPT.format(history=formatted_history)

            questions = self._generate_batch_questions(context, remaining_questions, k, max_retries)

            # Calculate EIG for all questions
            question_list = []
            def process_question(question):
                print(f"Calculating EIG for question: {question}")
                consistency_dict = self._get_consistency_dict(question, samples, history)
                eig = self._calculate_eig(consistency_dict, samples)
                return (question, eig)

            with ThreadPoolExecutor(max_workers=max(len(questions), 4)) as executor:
                question_list = list(executor.map(process_question, questions))

            print(f"Question EIGs: {question_list}")

            best_question = sorted(question_list, key=lambda x: x[1], reverse=True)[0][0]
            return best_question
        else:
            raise ValueError(f"Failed to generate valid questions after {max_retries} attempts")

    def _generate_batch_questions(self, context: str, remaining_questions: int, k: int, max_retries: int) -> list:
        """Generate k questions in a single batch using EIG_QUESTION_PROMPT"""
        prompt = EIG_QUESTION_PROMPT.format(context=context, remaining_questions=remaining_questions, k=k)

        for _ in range(max_retries):
            response = self.openrouter_agent(prompt)
            match = ANSWER_REGEX.search(response)

            if match:
                try:
                    dict_content = match.group(1).strip()
                    # Try JSON parsing first
                    questions_dict = json.loads(dict_content)
                    questions = [q.strip() for q in questions_dict.values()]
                    print(f"Generated {len(questions)} batch questions")
                    return questions
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for Python dict format
                    try:
                        questions_dict = ast.literal_eval(dict_content)
                        questions = [q.strip() for q in questions_dict.values()]
                        print(f"Generated {len(questions)} batch questions (via ast)")
                        return questions
                    except Exception as e:
                        print(f"Error parsing batch questions from tags: {e}")
                        print(f"Tag content was: {dict_content}")
                except Exception as e:
                    print(f"Error parsing batch questions from tags: {e}")
                    print(f"Tag content was: {dict_content}")
            print(f"Attempt {_} failed to make batch responses")

        print(f"Failed to generate batch questions after {max_retries} attempts")
        return []
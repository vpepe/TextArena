import json
import re
import textarena as ta
from prompts import BASE_PROMPT, DECISION_PROMPT, QUESTION_PROMPT, EIG_QUESTION_PROMPT, MOVE_PROMPT, CONSISTENCY_PROMPT
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
    def __init__(self, openrouter_agent: ta.agents.OpenRouterAgent):
        super().__init__(model_name=openrouter_agent.model_name)
        self.openrouter_agent = openrouter_agent

        with open('/home/ubuntu/new_battleship/20_questions/guess_who/TextArena/textarena/envs/GuessWho/characters.json', 'r') as f:
            characters = json.load(f)

        self.characters = characters

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
        remaining_questions = max(0, 10 - player_questions)

        print(remaining_questions, player_questions)
        decision = self.decision(formatted_history, remaining_questions)

        if DecisionType.GUESS == decision or remaining_questions == 0:
            action = self.move(formatted_history)
        else:
            action = self.question(history=formatted_history, remaining_questions=remaining_questions)

        return action
    
    def decision(self, history: str, remaining_questions: int, max_retries: int = 10) -> str:
        """Ask if the agent wants to ask more questions or try to guess"""
        context = BASE_PROMPT.format(history=history, characters=self.characters)
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

    def question(self, history: str, remaining_questions: int = 10, max_retries: int = 10) -> str:
        """Ask the agent for a question"""
        context = BASE_PROMPT.format(history=history, characters=self.characters)
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
        context = BASE_PROMPT.format(history=history, characters=self.characters)

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
    def __init__(self, openrouter_agent: ta.agents.OpenRouterAgent, k: int = 10):
        super().__init__(openrouter_agent=openrouter_agent)
        self.openrouter_agent = openrouter_agent
        self.sampling_agent = ta.agents.OpenRouterAgent(model_name="openai/gpt-5")  # Dedicated GPT-5 for sampling
        self.k = k
        self.constraints = []
        self.eigs = []

    def decision(self, history: str, remaining_questions: int, max_retries: int = 10) -> str:
        """Ask if the agent wants to ask more questions or try to guess"""
        if remaining_questions == 0:
            return DecisionType.GUESS
        else:
            return DecisionType.QUESTION
        # if len(self.eigs) == 0 or self.eigs[-1] > 0.1:
        #     return DecisionType.QUESTION  # Always ask at least one question
        # else:
        #     return DecisionType.GUESS  # Guess if EIG is low

    def get_constraints_from_question(self, question: str, max_retries: int = 5) -> dict:
        for attempt in range(max_retries):
            context = CONSISTENCY_PROMPT.format(question=question, characters=self.characters)
            response = self.sampling_agent(context)
            match = ANSWER_REGEX.search(response)

            if match:
                try:
                    dict_content = match.group(1).strip()
                    constraints = json.loads(dict_content)
                    print(f"Extracted constraints: {constraints}")
                    if set(constraints.keys()) == set(char['name'] for char in self.characters):
                        return constraints
                    else:
                        print(f"Constraints keys do not match character names")
                        continue
                except Exception as e:
                    print(f"Error parsing constraints from tags: {e}")
                    print(f"Tag content was: {dict_content}")
            print(f"Attempt {attempt + 1} failed to extract constraints")

    def calculate_eig_for_question(self, candidate_constraint: dict) -> float:
        character_weights = {char['name']: 1.0 for char in self.characters}
        eig_weights = {"yes": 0.0, "no": 0.0}

        for constraint in self.constraints:
            for char_name, answer in constraint.items():
                if answer == "no":
                    character_weights[char_name] *= EPSILON
                elif answer == "yes":
                    character_weights[char_name] *= (1 - EPSILON)

        for char_name, answer in candidate_constraint.items():
            eig_weights[answer] += character_weights[char_name]

        p_yes = eig_weights["yes"] / (eig_weights["no"] + eig_weights["yes"])

        return binary_entropy(EPSILON + ((1 - (2*EPSILON))*p_yes)) - binary_entropy(EPSILON) 



    def question(self, history, remaining_questions=40, max_retries: int = 5) -> str:
        for attempt in range(max_retries):
            # Generate k questions in a single batch
            context = BASE_PROMPT.format(history=history, characters=self.characters)
            questions = self._generate_batch_questions(context, remaining_questions, self.k, max_retries)

            if not questions:
                print(f"Failed to generate questions on attempt {attempt + 1}")
                continue

            # Calculate EIG for all questions
            question_eigs = []

            # Helper function to calculate EIG for a single question
            def _process_question(question):
                constraints = self.get_constraints_from_question(question)
                if not constraints:
                    return None

                print(f"Calculating EIG for: {question}")
                eig = self.calculate_eig_for_question(constraints)
                print(f"EIG: {eig:.4f}")

                return {
                    'question': question,
                    'constraints': constraints,
                    'eig': eig
                }

            # Process questions in parallel
            with ThreadPoolExecutor() as executor:
                # Map each question to a future
                future_results = list(executor.map(_process_question, questions))

                question_eigs = [result for result in future_results if result]

            if question_eigs:
                # Sort by EIG and return the best question
                question_eigs.sort(key=lambda x: x['eig'], reverse=True)
                best_question_info = question_eigs[0]
                best_question = best_question_info['question']
                best_eig = best_question_info['eig']
                print(f"Selected question with EIG {best_eig:.4f}: {best_question}")
                self.constraints.append(best_question_info['constraints'])
                self.eigs.append(best_eig)
                return best_question
            
            print(f"Attempt {attempt + 1} failed to compute EIG for any question")

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
                except Exception as e:
                    print(f"Error parsing batch questions from tags: {e}")
                    print(f"Tag content was: {dict_content}")
            print(f"Attempt {_} failed to make batch responses")

        print(f"Failed to generate batch questions after {max_retries} attempts")
        return []
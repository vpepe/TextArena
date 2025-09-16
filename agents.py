import json
import textarena as ta
from prompts import SAMPLES_PROMPT, REGEN_PROMPT, DECISION_PROMPT, QUESTION_PROMPT, MOVE_PROMPT, CONSISTENCY_PROMPT
import numpy as np
import ast

EPSILON = 0.1  # Noise parameter for answers

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

        decision = self.decision(formatted_history)
        
        if DecisionType.GUESS == decision:
            action = self.move(game_history)
        else:
            action = self.question(game_history)
        
        return action
    
    def decision(self, history: str) -> str:
        """Ask if the agent wants to ask more questions or try to guess"""
        prompt = DECISION_PROMPT.format(history=history)

        response = self.openrouter_agent(prompt)
        # Extract just the decision word, removing any extra text
        decision = response.strip()
        
        if decision in [DecisionType.QUESTION, DecisionType.GUESS]:
            return decision

        assert False, f"Unexpected decision response: {response}"
    
    def question(self, history: str) -> str:
        """Ask the agent for a question"""
        prompt = QUESTION_PROMPT.format(history=history)

        return self.openrouter_agent(prompt)
    
    def move(self, history: str) -> str:
        """Ask the agent for a move (final guess)"""
        prompt = MOVE_PROMPT.format(history=history)
        
        return self.openrouter_agent(prompt)

class EIGAgent(LLMAgent):
    def __init__(self, openrouter_agent: ta.agents.OpenRouterAgent, ground_truth_theme: str):
        super().__init__(openrouter_agent=openrouter_agent, ground_truth_theme=ground_truth_theme)
        self.openrouter_agent = openrouter_agent
        self.ground_truth_theme = ground_truth_theme
        self.samples = {}
        self._initialize_samples()

    def _initialize_samples(self):
        """Query the OpenRouterAgent for objects consistent with the theme"""
        prompt = SAMPLES_PROMPT.format(theme=self.ground_truth_theme)

        response = self.openrouter_agent(prompt)
        try:
            # Try to extract JSON from the response
            objects = json.loads(response)
            
            # Initialize samples with weight 1
            for obj in objects:
                self.samples[obj.lower().strip()] = 1
                
            print(f"Initialized EIGAgent with {len(self.samples)} samples for theme '{self.ground_truth_theme}'")
                
        except Exception as e:
            print(f"Error parsing samples: {e}")
            print(f"Response was: {response}")

    def _perform_regeneration(self, history):
        print("Performing regeneration of samples...")
        # Remove samples with weight below the threshold
        formatted_history = self.format_history(history)

        objects_to_replace = [obj for obj, weight in self.samples.items() if weight < (EPSILON)]

        # Remove low-weight samples
        for obj in objects_to_replace:
            del self.samples[obj]

        # Query for new samples
        prompt = REGEN_PROMPT.format(theme=self.ground_truth_theme, history=formatted_history)
        response = self.openrouter_agent(prompt)
        try:
            new_objects = json.loads(response)

            # Add new objects with initial weight 1
            for obj in new_objects:
                obj = obj.lower().strip()
                if obj not in self.samples:
                    self.samples[obj] = 1

            for sample, weight in self.samples.items():
                self.samples[sample] = 1.0
            print(f"Regenerated {len(objects_to_replace)} samples, now have {len(self.samples)} samples.")
                
        except Exception as e:
            print(f"Error parsing samples during regeneration: {e}")
            print(f"Response was: {response}")

    def _get_consistency_dict(self, question: str):
        prompt = CONSISTENCY_PROMPT.format(question=question, objects=list(self.samples.keys()))
        response = self.openrouter_agent(prompt)
        try:
            consistency_dict = ast.literal_eval(response)
            return consistency_dict
        except json.JSONDecodeError as e:
            print(f"Error parsing consistency dictionary: {e}")
            print(f"Response was: {response}")
            return {}

    def _calculate_eig(self, consistency_dict):
        weighted_results = {"yes": 0, "no": 0}

        for object, weight in self.samples.items():
            if object in consistency_dict:
                answer = consistency_dict[object]
                if answer == "yes":
                    weighted_results["yes"] += weight
                elif answer == "no":
                    weighted_results["no"] += weight
                else:
                    print(f"Unexpected answer '{answer}' for object '{object}'")
                    return float("nan")

        if any(v == 0 for v in weighted_results.values()):
            return 0

        # Calculate EIG using weighted probabilities
        total_weight = sum(weighted_results.values())
        p_true = weighted_results["yes"] / total_weight

        return binary_entropy(
            EPSILON + ((1 - 2 * EPSILON) * p_true)
        ) - binary_entropy(EPSILON)

    def update_weights(self, question: str, true_answer: str):
        print("Updating weights on:", question, true_answer)
        consistency_dict = self._get_consistency_dict(question)
        for object, weight in self.samples.items():
            if object in consistency_dict:
                answer = consistency_dict[object]
                if answer.lower() == true_answer.lower():
                    self.samples[object] *= (1 - EPSILON)
                else:
                    self.samples[object] *= EPSILON

    def question(self, history, k=5):
        if any([weight < (EPSILON) for weight in self.samples.values()]):
            self._perform_regeneration(history)

        question_list = []
        for _ in range(k):
            prompt = QUESTION_PROMPT.format(history=history)
            question = self.openrouter_agent(prompt)
            consistency_dict = self._get_consistency_dict(question)
            eig = self._calculate_eig(consistency_dict)
            question_list.append((question, eig))
        
        print(f"Question EIGs: {question_list}")

        best_question = sorted(question_list, key=lambda x: x[1], reverse=True)[0][0]

        return best_question
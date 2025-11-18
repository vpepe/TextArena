import re, random, json, os, importlib.resources
from typing import Any, Dict, Optional, Tuple
import textarena as ta
from textarena.envs.TwentyQuestions.renderer import create_board_str

class TwentyQuestionsEnv(ta.Env):
    def __init__(self, hardcore: Optional[bool]=False, max_turns: int=21):
        """
        Args:
            hardcore: Whether to use more challenging words
            max_turns: Maximum number of turns allowed in the game
        """
        self.hardcore = hardcore
        self.max_turns = max_turns

        # Initialize the gamemaster
        self.gamemaster = ta.agents.OpenRouterAgent(model_name="openai/gpt-4o")
        self.gamemaster_options = ["Yes", "No", "I don't know"]
        self.gamemaster_context = None
        self.gamemaster_history = []

        # Load the word list
        self.word_list = self._load_words()

    def _load_words(self, words_path: Optional[str] = None):
        try:
            if words_path is not None:
                # Use provided path
                if not os.path.exists(words_path): raise FileNotFoundError(f"Words data file not found at: {words_path}")
                with open(words_path, "r", encoding="utf-8") as file: word_data = json.load(file)
            else:
                # Use package resource
                with importlib.resources.files('textarena.envs.TwentyQuestions').joinpath('twenty_questions_words.json').open('r') as file:
                    word_data = json.load(file)
            category = "hardcore" if self.hardcore else "basic"
            words = word_data.get(category, [])
            if not words: raise ValueError(f"No words found for difficulty level '{category}'.")
            return words
        except Exception as e: raise FileNotFoundError(f"Failed to load words data: {str(e)}")

    def get_board_str(self): return create_board_str(game_state=self.state.game_state)
    def get_gamemaster_response(self, action: str) -> str:
        # Validate gamemaster state
        ANSWER_REGEX = re.compile(r'<answer>(.*?)</answer>', re.IGNORECASE | re.DOTALL)
        if self.gamemaster_context is None: raise ValueError("Gamemaster context is not set.")
        if self.gamemaster_history is None: raise ValueError("History is not set.")
        if self.gamemaster_options is None: raise ValueError("Gamemaster options are not set.")
        options = ", ".join(f"'{opt}'" for opt in self.gamemaster_options) # Format available response options
        history = "\n".join(f"Q: {q}\nA: {a}" for q, a in self.gamemaster_history) # Construct conversation history
        prompt = (f"{self.gamemaster_context}\n{history}\n\nQ: {action}\nOptions: {options}\n\nPlease respond with the most appropriate option.") # Create prompt
        for _ in range(10):
            #response = self.gamemaster(prompt).strip() # Get response from the gamemaster agent
            completion = self.gamemaster(prompt)
            response = ANSWER_REGEX.search(completion)

            if response:
                response = response.group(1).strip()
                break
        else:
            raise RuntimeError("Failed to get a valid response from the gamemaster after multiple attempts.")


        # Validate response
        if any(option.lower() in response.lower() for option in self.gamemaster_options): self.gamemaster_history.append((action, response))  # Store valid responses
        else: self.gamemaster_history.append((action, "I'm sorry, I don't understand. Please try asking again."))  # Log fallback response
        return response

    def reset(self, num_players: int, game_theme: Optional[str] = None, seed: Optional[int] = None, target_word: Optional[str] = None):
        self.state = ta.SinglePlayerState(num_players=num_players, seed=seed, max_turns=self.max_turns)
        ## load the game word
        if game_theme is None:
            self.game_theme = random.choice(list(self.word_list.keys()))
        else:
            self.game_theme = game_theme

        # Use provided target_word if specified, otherwise randomly select
        if target_word is not None:
            self.game_word = target_word
        else:
            self.game_word = random.choice(self.word_list[self.game_theme])

        self.gamemaster_context = (
            f"You are playing 20 Questions. You are thinking of a word, '{self.game_word}'. \n Your task is to provide honest answers to the other player's yes-or-no questions to guide them towards finding the word you are thinking of (you both win if the other player guesses the word, and you both lose if they do not). \n Be sure to provide answers that are truthful and helpful, which may include some nuance and inference about what the other player may be thinking: for example, if the word is 'pharmacist', and the question is 'Does the profession deal with numbers or financial data?', while it is technically true that pharmacists deal with numbers, the best answer is 'No', since the question is likely trying to determine if the word is related to finance or accounting. You should only respond with one of the following options: {', '.join(self.gamemaster_options)}. \n Please think about your answer step-by-step before responding, and wrap your final answer in <answer></answer> tags, e.g. <answer>Yes</answer>."
        )
        self.state.reset(game_state={"target_word": self.game_word, "rendered_text": f"Game word: {self.game_word}"}, player_prompt_function=self._prompt)

    def _prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        return f"The game has begun."
        #     f"You are Player {player_id}. You are playing 20 Questions ).\n"
        #     f"The gamemaster has chosen a word that can be one or two words. You have to guess this word by asking yes-or-no questions. This word is related to {self.game_theme}.\n"
        #     "The game will last for a maximum of 20 questions. After 20 questions, the gamemaster will prompt you to make a guess.\n"
        #     "You may ask your question in any manner, so long they are not wrapped in square brackets.\n"
        #     "Then, to make your final word guess, ensure that you wrap it with square brackets, e.g. [plane], [diving bell].\n"
        #     "As you play, the history of your questions and gamemaster's responses will be displayed."
        # )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        self.state.add_observation(from_id=player_id, to_id=-1, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        action_match = re.compile(r"\[([a-zA-Z\s]+)\]").search(action) # e.g. [diving bell]
        if not action_match or (action_match and '?' in action): ## if the action is not a guess, or if it is a action but contains a question mark, then it is a question
            gamemaster_response = self.get_gamemaster_response(action)
            if "history" not in self.state.game_state: self.state.game_state["history"] = []
            self.state.game_state["history"].append((action, gamemaster_response))
            if self.state.turn == self.state.max_turns-2: gamemaster_response += "\nYou have run out of questions. What is your final guess?"
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=gamemaster_response, observation_type=ta.ObservationType.GAME_MESSAGE)
        else: ## if the action is a guess
            action_text = action_match.group(1).lower()
            if self.game_word in action_text: self.state.set_outcome(reward=1, reason=f"Congratulations! You guessed the word.")
            else: self.state.set_outcome(reward=0, reason=f"Invalid guess. You guessed incorrectly.")
            self.state.game_state["rendered_text"] = f"Game word: {self.game_word}"
        if self.state.check_turn_limit() and not self.state.done: self.state.set_outcome(reward=0, reason=f"The turn limit has been reached")
        return self.state.step()



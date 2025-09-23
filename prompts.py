# Define ground_truth_theme here for this cell to work independently

BASE_PROMPT = """
You are playing Guess Who. The other player has selected a character from the list below, and your goal is to guess which character they have selected by asking yes-or-no questions about the character's traits. Your goal is to identify the character in as few questions as possible.

You have 40 questions to ask, and you can make your guess at any time. If you guess correctly, you win the game. If you run out of questions or guess incorrectly, you lose the game.

Given this game history:

<history>
{history}
</history>

This is the list of characters you can ask questions about:

<characters>
{characters}
</characters>
"""


DECISION_PROMPT = """
{context}

Your task is to choose whether you'd like to ask another question about the word to gather more information about it, or, if you feel you already have enough information, to make your guess for the final answer.

You have {remaining_questions} questions left, including this one. Be very careful about guessing early, since there is no cost to asking questions, and if you guess wrong, you lose the game.

Please think about your decision step by step. When you have come up with a final answer, respond with your decision wrapped in <answer></answer> tags: <answer>question</answer> if you would like to ask a question or <answer>guess</answer> if you would like to make your guess. Do not give the actual question or guess yet.
"""

QUESTION_PROMPT = """
{context}

Your task is to ask a single question that will help you gain the most information possible about the secret word. You can ask any question, but is must be answerable with a Boolean answer (yes/no). Make sure your questions are clear, specific and different from ones you have asked previously, to avoid pigeonholing a potentially wrong answer too quickly.

You have {remaining_questions} questions left, including this one.

Please think about your answer step by step. When you have come up with your question, please wrap it in <answer></answer> tags: e.g. <answer>Is the character male?</answer>
"""

EIG_QUESTION_PROMPT = """
{context}

Your task is to generate {k} question(s) that will help you gain the most information possible about the secret word. Each question must be answerable with a Boolean answer (yes/no).

You have {remaining_questions} batches of {k} question(s) left.

Please think about your answer step by step. When you have come up with your question, please return your question(s) as a JSON dictionary with numbered keys, wrapped in <answer></answer> tags like this: <answer>{{"1": "Is the character male?", "2": "Do they have a mustache?", "3": "Do they have a hat?"}}</answer>

IMPORTANT: Use proper JSON format with double quotes around both keys and values.
"""

MOVE_PROMPT = """
{context}

Your task is to make your one and only guess for the character. Make sure you consider the context of the theme and all previous questions and answers. 

Guess from the list above.

Please think about your answer step by step. When you have come up with a final answer, respond with your guess wrapped in <answer></answer> tags, and optionally square brackets, e.g. <answer>Mike</answer> or <answer>[Mike]</answer>"""

#---------------------------------------------

CONSISTENCY_PROMPT = """
You are playing Guess Who. Your task is to determine which characters from the list below are consistent with a "yes" answer to the most recent question asked, and which are consistent with a "no" answer.

The most recent question is "{question}".

Given the following characters:

<characters>
{characters}
</characters>

Respond with a JSON dictionary wrapped in <answer></answer> tags where the keys are the characters and the values are either "yes", if the character satisfies the constraint given in the question, or "no" if they do not.

<answer>{{"character1": "yes", "character2": "no", "character3": "yes"}}</answer>

IMPORTANT: Use proper JSON format with double quotes around both keys and values.

"""
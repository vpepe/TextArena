# Define ground_truth_theme here for this cell to work independently

BASE_PROMPT = """
You are playing 20 Questions. Your goal is to gather enough information to be able to guess a secret word that goes with a given theme by asking at most 20 yes/no questions.

The secret word can be one or two words long, and can be any word of concept that fits the theme. You can guess at any time, but if you guess wrong, you lose the game. After 20 questions, you will be forced to make a final guess.

Given this game history:

{history}
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

Please think about your answer step by step. When you have come up with your question, please wrap it in <answer></answer> tags: e.g. <answer>Is it a living thing?</answer>
"""

EIG_QUESTION_PROMPT = """
{context}

Your task is to generate {k} questions that will help you gain the most information possible about the secret word. Each question must be answerable with a Boolean answer (yes/no).

You have {remaining_questions} batches of {k} questions left.

Please think about your answer step by step. When you have come up with your question, please return your questions as a JSON dictionary with numbered keys, wrapped in <answer></answer> tags like this: <answer>{{"1": "Is it a living thing?", "2": "Is it larger than a car?", "3": "Is it made of metal?"}}</answer>

IMPORTANT: Use proper JSON format with double quotes around both keys and values.
"""

MOVE_PROMPT = """
{context}

Your task is to make your one and only guess for the secret word. Make sure you consider the context of the theme and all previous questions and answers. 

Please think about your answer step by step. When you have come up with a final answer, respond with your guess wrapped in <answer></answer> tags, and optionally square brackets, e.g. <answer>elephant</answer> or <answer>[elephant]</answer>"""

#---------------------------------------------

SAMPLES_PROMPT = """
{context}

List 100 different objects, items, or concepts that fit the question-answer history given thus far.

For example, if the history includes a question "Is it a living thing?" with answer "no", then none of the objects can be living things. 

Return your answer as a JSON dictionary with numbered keys, wrapped in <answer></answer> tags like this:
    <answer>{{"1": "object1", "2": "object2", "3": "object3", "4": "object4", "5": "object5"}}</answer>

IMPORTANT: Use proper JSON format with double quotes around both keys and values.
"""


# SAMPLES_PROMPT = """
# {context}

# List 100 different objects, items, or concepts that would be appropriate for the theme/category "{theme}". You should list as many as possible, but take care to make sure they all fit the question-answer history given thus far.

# For example, if the history includes a question "Is it a living thing?" with answer "no", then none of the objects can be living things. 

# Return your answer as a JSON dictionary with numbered keys, wrapped in <answer></answer> tags like this:
#     <answer>{{"1": "object1", "2": "object2", "3": "object3", "4": "object4", "5": "object5"}}</answer>

# IMPORTANT: Use proper JSON format with double quotes around both keys and values.
# """

CONSISTENCY_PROMPT = """
{context}

Here is the most recent question and a list of objects that might be the secret word:

Question: "{question}"
Possible objects: {objects}

Your task is to determine which objects are consistent with a "yes" answer to the question, and which are consistent with a "no" answer.

Respond with a JSON dictionary wrapped in <answer></answer> tags where the keys are the objects and the values are either "yes" or "no".

<answer>{{"object1": "yes", "object2": "no", "object3": "yes"}}</answer>

IMPORTANT: Use proper JSON format with double quotes around both keys and values.

"""
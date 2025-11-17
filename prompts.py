# Define ground_truth_theme here for this cell to work independently

GAME_PROMPT = """
You are playing 20 Questions. Your goal is to gather enough information to be able to guess a secret word by asking at most 20 questions.

RULES:
- You can ask any question, but the Gamemaster can only answer "Yes," "No," or "I don't know."
- You can guess at any time, but if you guess wrong, you lose the game.
- After 20 questions, you will be forced to make a final guess.
"""

BASE_PROMPT = GAME_PROMPT + """
Here is the list of possible words:
{word_list}

Here is the history of questions asked and answers given so far in the game:
{history}
"""

BELIEF_PROMPT = BASE_PROMPT + """
Based on the history so far, here are your current beliefs about what the secret word might be, ranked by probability:
{belief_state}
"""

DECISION_PROMPT = """
{context}

Choose whether to ask another question or make your guess for the final answer. Be careful about guessing early, since if you guess wrong, you lose the game.

You have {remaining_questions} questions left, including this one.

Please think about your decision step by step and answer with one of the following options:

- If you would like to ask a question, respond `<answer>question</answer>`
- If you would like to make your final guess, respond `<answer>guess</answer>`

Be sure your response is wrapped in `<answer></answer>` tags. Do not give the actual question or guess.
"""

QUESTION_PROMPT = """
{context}

Your task is to ask a single question that will help you gain the most information possible about the secret word. You can ask any question, but is must be answerable with "Yes," "No," or "I don't know." Make sure your questions are clear and distinct from ones you have asked previously.

You have {remaining_questions} turns left.

Please think about your answer step by step. When you have come up with your question, please wrap it in <answer></answer> tags. Here is an example:

<answer>Is it a living thing?</answer>
"""

EIG_QUESTION_PROMPT = """
{context}

Your task is to generate a set of {k} candidate question(s) that will help you gain the most information possible about the secret word. You can ask any question, but is must be answerable with "Yes," "No," or "I don't know." Make sure your questions are clear and distinct from ones you have asked previously. If providing multiple candidates, please ensure that the questions are diverse and cover different aspects of the secret word.

You have {remaining_questions} turns left.

Use this belief state to guide your question generation. Focus on questions that will help distinguish between the most likely candidates.

Please think about your answer step by step. When you have come up with your question, please return your question(s) as a JSON dictionary with numbered keys, wrapped in <answer></answer> tags like this: <answer>{{"1": "Is it a living thing?", "2": "Is it larger than a car?", "3": "Is it made of metal?"}}</answer>

IMPORTANT: Use proper JSON format with double quotes around both keys and values.
"""

MOVE_PROMPT = """
{context}

Your task is to make your one and only guess for the secret word. Make sure you consider the context of the theme and all previous questions and answers.

Guess from the list below:

<objects>
{objects}
</objects>

Please think about your answer step by step. When you have come up with a final answer, respond with your guess wrapped in <answer></answer> tags, and optionally square brackets, e.g. <answer>elephant</answer> or <answer>[elephant]</answer>"""

# ---------------------------------------------

SAMPLES_PROMPT = """
{context}

From the list below, list all the different objects, items, or concepts that fit the question-answer history given thus far.

<objects>
{objects}
</objects>

For example, if the history includes a question "Is it a living thing?" with answer "no", then none of the objects can be living things.

Return your answer as a JSON dictionary with numbered keys, wrapped in <answer></answer> tags like this:
    <answer>{{"1": "object1", "2": "object2", "3": "object3", "4": "object4", "5": "object5"}}</answer>

IMPORTANT: Use proper JSON format with double quotes around both keys and values.
"""

CONSISTENCY_PROMPT = """
{context}

Here is the most recent question and a list of objects that might be the secret word:

Question: "{question}"
Possible objects: {objects}

Your task is to determine which objects are consistent with a "Yes" answer to the question, and which are consistent with a "No" answer. If the answer is ambiguous, you should respond with "I don't know" for that object.

Respond with a JSON dictionary wrapped in <answer></answer> tags where the keys are the objects and the values are one of ["Yes", "No", "I don't know"], like this:

<answer>{{"object1": "Yes", "object2": "No", "object3": "I don't know"}}</answer>

IMPORTANT: Use proper JSON format with double quotes around both keys and values.

"""

# Define ground_truth_theme here for this cell to work independently
SAMPLES_PROMPT = f"""
List 20 different objects, items, or concepts that would be appropriate for the theme/category "{{theme}}" in a 20 Questions game. 

Return only a list of strings, like: ["object1", "object2", "object3", ...]. Do not use markdown fences or other formatting.

Objects should be:
- Concrete nouns that can be guessed in 20 questions
- Appropriate for the theme "{{theme}}"
- Diverse within the theme
- 1-2 words each"""

REGEN_PROMPT = f"""
List 20 different objects, items, or concepts that would be appropriate for the theme/category "{{theme}}" in a 20 Questions game. 

Return only a list of strings, like: ["object1", "object2", "object3", ...]. Do not use markdown fences or other formatting.

Objects should be:
- Concrete nouns that can be guessed in 20 questions
- Appropriate for the theme "{{theme}}"
- Diverse within the theme
- 1-2 words each

They must also be consistent with the following previous questions and answers:
{{history}}"""

DECISION_PROMPT = f"""
Given this 20 Questions game history:

{{history}}

Should you:
A) Ask another question to gather more information
B) Make a guess for the final answer

Consider:
- How much information you have
- How confident you are about the answer
- How many questions remain

Respond with ONLY one word: either "question" or "guess" (no quotes, no explanation)."""

QUESTION_PROMPT = f"""
You are playing 20 Questions. Based on this game history:

{{history}}

What is your next yes/no question? Make it strategic and informative.
Ask only the question, without brackets or extra formatting."""

MOVE_PROMPT = f"""
Based on this 20 Questions game history:

{{history}}

What is your final guess? Provide your answer in square brackets, like [your guess].
Make your best guess based on all the information gathered."""

CONSISTENCY_PROMPT = f"""
You will be given a question and a list of possible objects, coming from a 20 questions game. Your task is to determine which objects are consistent with a "yes" answer to the question, and which are consistent with a "no" answer. Respond with a Python dictionary where the keys are the objects and the values are either "yes" or "no".

Just output the dictionary, no markdown fences or extra text.

Question: "{{question}}"
Possible objects: {{objects}}
"""
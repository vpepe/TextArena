"""Test script for the asymmetric noise matrix implementation."""

import numpy as np

# Define the noise matrix (same as in agents.py)
NOISE_MATRIX = {
    "yes": {
        "yes": 0.90,
        "no": 0.01,
        "i don't know": 0.25,
    },
    "no": {
        "yes": 0.01,
        "no": 0.90,
        "i don't know": 0.25,
    },
    "i don't know": {
        "yes": 0.09,
        "no": 0.09,
        "i don't know": 0.50,
    }
}

def get_likelihood(observed_answer: str, true_answer: str) -> float:
    """Get the likelihood p(observed_answer | true_answer) from the noise matrix."""
    observed_answer = observed_answer.lower().strip()
    true_answer = true_answer.lower().strip()

    valid_answers = ["yes", "no", "i don't know"]
    if observed_answer not in valid_answers:
        raise ValueError(f"Invalid observed answer: '{observed_answer}'. Must be one of {valid_answers}")
    if true_answer not in valid_answers:
        raise ValueError(f"Invalid true answer: '{true_answer}'. Must be one of {valid_answers}")

    return NOISE_MATRIX[observed_answer][true_answer]

def test_noise_matrix_validation():
    """Test that noise matrix columns sum to 1.0."""
    print("Testing noise matrix validation...")
    for true_answer in ["yes", "no", "i don't know"]:
        column_sum = sum(NOISE_MATRIX[obs][true_answer] for obs in ["yes", "no", "i don't know"])
        print(f"  Column '{true_answer}' sum: {column_sum:.6f}")
        assert np.isclose(column_sum, 1.0, atol=1e-6), f"Column '{true_answer}' doesn't sum to 1.0"
    print("✓ All columns sum to 1.0\n")

def test_get_likelihood():
    """Test the get_likelihood function."""
    print("Testing get_likelihood function...")

    # Test exact matches
    assert get_likelihood("yes", "yes") == 0.90
    assert get_likelihood("no", "no") == 0.90
    assert get_likelihood("i don't know", "i don't know") == 0.50
    print("✓ Exact matches work correctly")

    # Test asymmetry: IDK is more likely confused with yes/no than vice versa
    assert get_likelihood("yes", "i don't know") == 0.25  # p(observe yes | true IDK)
    assert get_likelihood("i don't know", "yes") == 0.09  # p(observe IDK | true yes)
    print("✓ Asymmetry works correctly: p(yes|IDK) > p(IDK|yes)")

    # Test confusion between yes and no is low
    assert get_likelihood("yes", "no") == 0.01
    assert get_likelihood("no", "yes") == 0.01
    print("✓ Yes/No confusion is minimal")

    # Test case insensitivity and whitespace handling
    assert get_likelihood("  YES  ", "yes") == 0.90
    assert get_likelihood("NO", "no") == 0.90
    print("✓ Case insensitivity and whitespace handling work\n")

def test_bayesian_update_example():
    """Test a simple Bayesian update example."""
    print("Testing Bayesian update example...")

    # Initial beliefs (uniform over 3 words)
    beliefs = {
        "word1": 1/3,  # true answer: yes
        "word2": 1/3,  # true answer: no
        "word3": 1/3,  # true answer: i don't know
    }

    # Simulated answers for some question
    simulated_answers = {
        "word1": "yes",
        "word2": "no",
        "word3": "i don't know",
    }

    # Observed answer is "yes"
    observed_answer = "yes"

    # Calculate posteriors
    new_beliefs = {}
    for word, prob in beliefs.items():
        simulated = simulated_answers[word]
        likelihood = get_likelihood(observed_answer, simulated)
        new_beliefs[word] = prob * likelihood

    # Normalize
    total = sum(new_beliefs.values())
    new_beliefs = {word: p / total for word, p in new_beliefs.items()}

    print(f"  Prior beliefs: {beliefs}")
    print(f"  Simulated answers: {simulated_answers}")
    print(f"  Observed answer: '{observed_answer}'")
    print(f"  Posterior beliefs: {new_beliefs}")

    # word1 should have highest belief (true answer matches observation)
    assert new_beliefs["word1"] > new_beliefs["word2"]
    assert new_beliefs["word1"] > new_beliefs["word3"]
    print(f"✓ Belief in word1 increased most (true=yes, observed=yes)")

    # word3 should have higher belief than word2 (IDK more likely to be confused with yes)
    assert new_beliefs["word3"] > new_beliefs["word2"]
    print(f"✓ word3 (IDK) has higher belief than word2 (no) due to asymmetry\n")

if __name__ == "__main__":
    print("="*60)
    print("ASYMMETRIC NOISE MATRIX TESTS")
    print("="*60 + "\n")

    test_noise_matrix_validation()
    test_get_likelihood()
    test_bayesian_update_example()

    print("="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)

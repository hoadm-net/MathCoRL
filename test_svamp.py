import json
from fpp import solve_single
from mint.utils import load_svamp_test_data


def test_svamp_dataset():
    # Load test data
    svamp_test_data = load_svamp_test_data('datasets/SVAMP/test.json')

    # Initialize results list
    results = []

    # Test with FPP
    for sample in svamp_test_data:
        question = sample['question']
        context = sample['context']
        ground_truth = sample['ground_truth']

        # Solve using FPP
        result = solve_single(question, context)

        # Append result
        results.append({
            'question': question,
            'context': context,
            'ground_truth': ground_truth,
            'result': result,
            'correct': result == ground_truth
        })

    # Save results to file
    with open('svamp_test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Calculate statistics
    correct_count = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct_count / total * 100

    # Print statistics
    print(f"Tested {total} samples.")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    test_svamp_dataset() 
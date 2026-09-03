"""A simple Python fundamentals example for Week 1."""


def calculate_average(values: list[int]) -> float:
    """Return the average of a non-empty list of numbers."""
    return sum(values) / len(values)


def main() -> None:
    student_name = "Hong Gil Dong"
    scores = [90, 85, 95]
    average = calculate_average(scores)

    print(f"Student: {student_name}")
    print("Scores:")

    for index, score in enumerate(scores, start=1):
        print(f"  Subject {index}: {score}")

    result = {
        "average": average,
        "passed": average >= 60,
    }

    print(f"Average: {result['average']:.1f}")
    print(f"Passed: {result['passed']}")


if __name__ == "__main__":
    main()

# Python sample code
# Run: python sample.py


def greet(name: str) -> str:
    return f"Hello, {name}! Welcome to Python!"


def average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)


def main():
    print(greet("student"))

    scores = [85, 90, 78, 96, 88]
    avg = average(scores)

    print(f"Scores: {scores}")
    print(f"Average: {avg:.2f}")

    # Simple input example
    user_name = input("이름을 입력하세요: ")
    print(f"안녕하세요, {user_name}님!")

    print("Hello, World!")


if __name__ == "__main__":
    main()

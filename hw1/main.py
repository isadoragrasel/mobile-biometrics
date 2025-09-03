# Homework 1 - Mobile Biometrics

# 1. Variables and Basic Operations
def basic_operations(a, b):
    try:
        print("addition: ", a + b)
        print("subtraction: ", a - b)
        print("multiplication: ", a * b)
        if b == 0:
            raise ZeroDivisionError('No number can be divided by zero.')
        print("division: ", a / b)
    except ZeroDivisionError as (e):
        print(e)


# 2. Loops and Lists
def loops_and_lists():
    numbers = list(range(1, 11))
    print("Full list: ")
    for num in numbers:
        print(num)
    print("Even numbers only: ")
    for num in numbers:
        if num % 2 == 0:
            print(num)


# 3. Conditionals
def check_numbers():
    try:
        x = int(input("Enter a number x here: "))

        if x == 0:
            print("x is even (and zero)")
        elif x > 0:
            if x % 2 == 0:
                print("x is even (and positive)")
            else:
                print("x is odd (and positive)")
        else:
            if x % 2 == 0:
                print("x is even (and negative)")
            else:
                print("x is odd (and negative)")
    except ValueError:
        print("Invalid input. Please enter an integer.")


# 4. Factorial
def factorial(n):
    try:
        if n < 0:
            raise ValueError('We need a nonnegative integer to calculate factorial')
        elif n == 0:
            return 1
        else:
            return n * factorial(n - 1)
    except ValueError as e:
        print(e)


def main():
    try:
        a = int(input("Enter a number a here: "))
        b = int(input("Enter a number b here: "))
        basic_operations(a, b)
        loops_and_lists()
        check_numbers()
        n = int(input("Enter a number n here: "))
        print(factorial(n))
    except ValueError:
        print('Invalid input. Please enter integers for a, b, and n.')


main()


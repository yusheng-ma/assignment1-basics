# hello_scalene.py

import time

def slow_function():
    total = 0
    for i in range(10**6):
        total += i
    return total

def fast_function():
    return sum(range(100000))

def main():
    slow_function()
    fast_function()
    time.sleep(1)

if __name__ == "__main__":
    main()

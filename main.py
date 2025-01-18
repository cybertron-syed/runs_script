import os

count_file = 'run_count.txt'

def read_count():
    """Read the current count from the file."""
    if os.path.exists(count_file):
        with open(count_file, 'r') as file:
            return int(file.read().strip())
    return 0

def write_count(count):
    """Write the updated count to the file."""
    with open(count_file, 'w') as file:
        file.write(str(count))

def main():
    count = read_count()
    count += 1
    write_count(count)
    print(f"The script has run {count} times.")

if __name__ == "__main__":
    main()

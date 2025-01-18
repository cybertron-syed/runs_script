import os

# Define the file where the count will be stored
count_file = 'run_count.txt'

def read_count():
    """Read the current count from the file."""
    if os.path.exists(count_file):
        with open(count_file, 'r') as file:
            return int(file.read().strip())
    return 0  # Return 0 if the file doesn't exist

def write_count(count):
    """Write the updated count to the file."""
    with open(count_file, 'w') as file:
        file.write(str(count))

def main():
    # Read the current count
    count = read_count()
    
    # Increment the count
    count += 1
    
    # Write the new count back to the file
    write_count(count)
    
    # Print or log the current run count
    print(f"The script has run {count} times.")

if __name__ == "__main__":
    main()

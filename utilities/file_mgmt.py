import os
from contextlib import contextmanager
import sys
from datetime import datetime
import random
from pathlib import Path

def create_timestamped_dir(base_path: Path):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Generate a random number (between 1 and 100, for example)
    random_number = random.randint(1, 100)

    # Combine timestamp and random number
    directory_name = f"{timestamp}_{random_number}"

    # Create the timestamped directory
    timestamped_directory = base_path / directory_name

    # Create the directory if it doesn't exist
    timestamped_directory.mkdir(parents=True, exist_ok=True)

    print(f"Timestamped directory created: {timestamped_directory}")

    return timestamped_directory

def img_timestamped_fname():
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Generate a random number (between 1 and 100, for example)
    random_number = random.randint(1, 100)

    # Combine timestamp and random number for filename
    file_name = f"{timestamp}_{random_number}.png"

    print(f"Timestamped PNG filename created: {file_name}")
    return file_name

@contextmanager
def suppress_stdout():   # Suppress prints from imported modules
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def empty_directory_and_subdirectories(directory_path):
    directory_path = str(directory_path)
    # Empty files in the specified directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Empty files in immediate subdirectories
    for subdirectory_name in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdirectory_name)
        if os.path.isdir(subdirectory_path):
            for filename in os.listdir(subdirectory_path):
                file_path = os.path.join(subdirectory_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)




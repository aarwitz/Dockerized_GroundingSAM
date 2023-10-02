import os
from contextlib import contextmanager
import sys


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




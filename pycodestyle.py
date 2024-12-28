import sys
import subprocess

def run_linter(command):
    """Run a linter command and return the result."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        # Пытаемся декодировать stderr с игнорированием ошибок
        print(f"Error: {stderr.decode(errors='ignore')}")
    return stdout.decode(errors='ignore')

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 pycodestyle.py <script.py>")
        sys.exit(1)

    script = sys.argv[1]

    # Running flake8 using python -m flake8
    print(f"Running flake8 on {script}...")
    flake8_command = f"flake8 {script} --max-line-length=120"
    flake8_output = run_linter(flake8_command)
    print(flake8_output)

    # Running pylint
    print(f"Running pylint on {script}...")
    pylint_command = f"pylint {script} --max-line-length=120 --disable='C0103,C0114,C0115'"
    pylint_output = run_linter(pylint_command)
    print(pylint_output)

if __name__ == "__main__":
    main()

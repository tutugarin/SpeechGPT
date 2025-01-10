import subprocess
from setuptools import setup


pip_v = subprocess.check_output(["pip", "--version"]).decode("utf-8")
if not "pip 24.0" in pip_v:
    install_pip = subprocess.run(["python", "-m", "pip", "install", "pip==24.0"])
    if install_pip.returncode != 0:
        raise RuntimeError("pip==24.0 cannot be installed automatically. Please install it manually.")


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="speechgpt",
    version="1.0",
    packages=["speechgpt"],
    install_requires=required,
)

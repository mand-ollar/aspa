from setuptools import find_packages, setup


def parse_requirements(filename: str) -> list[str]:
    with open(file=filename, mode="r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="aspa",
    version="0.1.0",
    author="Daniel Oh",
    description="Audio Signal Processing & Analysis (ASPA) is a Python package for audio signal processing and analysis.",
    packages=find_packages(include=["aspa", "aspa.*"]),
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.10",
    package_data={"aspa": ["py.typed"]},
    include_package_data=True,
)

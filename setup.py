from setuptools import setup, find_packages

setup(
    name="your_project_name",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch==2.0.1",
        "transformers==4.28.1",
        "pillow==9.5.0",
        "openai",
        "salesforce-lavis==1.0.2"
    ],
)

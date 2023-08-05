from setuptools import setup, find_packages
from typing import List

hyphen_e = "-e ."

def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as f:
        requirements =  f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if hyphen_e in requirements:
            requirements.remove(hyphen_e)
        
    return requirements
    


setup(
    name='mlops',
    author='hjskhan',
    author_email="hjskhan47@gmail.com",
    version='0.0.1', 
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
    
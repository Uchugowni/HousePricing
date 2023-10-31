from setuptools import find_packages, setup
from typing import List

__version__="0.0.0"

REPO_NAME = "HousePricing"    #github repository name
AUTHOR_USER_NAME = "Uchugowni"  #github name
AUTHOR_EMAIL = "uchugownigiribabu@gmail.com" #github email id

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)
from setuptools import find_packages, setup
#import setuptools 

#with open("README.md", "r", encoding="utf-8") as f:
#    long_description = f.read()


__version__="0.0.0"

REPO_NAME = "HousePricing"    #github repository name
AUTHOR_USER_NAME = "Uchugowni"  #github name
SRC_REPO = "housePricing"   #src\housePricing under src repo
AUTHOR_EMAIL = "uchugownigiribabu@gmail.com" #github email id

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->list[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements


setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for NLP app",
    long_description_content = "text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    install_requires = get_requirements('requirements.txt')
)
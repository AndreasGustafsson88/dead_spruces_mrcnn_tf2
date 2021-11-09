from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dead_pines_mrcnn_tf2",
    version="1.0.0",
    author="Andreas Gustafsson",
    author_email="andreas.gustafsson88@gmail.com",
    description="Automatic detection of standing dead pines from aerial imagery using mask_rcnn and tensorflow 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AndreasGustafsson88/dead_pines_mrcnn_tf2",
    project_urls={
        "Source Code": "https://github.com/AndreasGustafsson88/dead_pines_mrcnn_tf2",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3',
)
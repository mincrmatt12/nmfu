from setuptools import setup
import sys
import os

# allow importing nmfu
sys.path.append(os.path.dirname(__file__))

from nmfu import __version__ as nmfu_version

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    readme = f.read()

setup(
        name="nmfu",
        version=nmfu_version,
        py_modules=["nmfu"],
        description="A parser generator that turns procedural programs into C state machines",
        entry_points={
            "console_scripts": ["nmfu=nmfu:main"]
        },
        install_requires=["lark-parser==0.11.*"],
        author="Matthew Mirvish",
        author_email="matthew@mm12.xyz",

        license="GPLv3",
        long_description=readme,
        long_description_content_type="text/markdown",

        keywords="c parser parser-generator cli tool",
        url="https://github.com/mincrmatt12/nmfu",
        project_urls={
            "Bug Tracker": "https://github.com/mincrmatt12/nmfu/issues",
            "Source Code": "https://github.com/mincrmatt12/nmfu"
        },

        extras_require={
            "debug": ["graphviz==0.14"],
            "tests": ["pytest", "hypothesis"],
            "coverage": ["pytest-cov"]
        },
        python_requires="~=3.6",
        
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Intended Audience :: Developers",
            "Programming Language :: C",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Topic :: Software Development :: Code Generators"
        ]
)

# nmfu

<img src="https://user-images.githubusercontent.com/5255209/117226360-7e69a900-ade2-11eb-9127-4a146a443199.png" alt="nmfu logo banner" width="100%"/>

---
_the "no memory for you" "parser" generator_

---

![PyPI - License](https://img.shields.io/pypi/l/nmfu) [![PyPI](https://img.shields.io/pypi/v/nmfu)](https://pypi.org/project/nmfu) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nmfu) [![Jenkins](https://img.shields.io/jenkins/build?jobUrl=https%3A%2F%2Fjenkins.mm12.xyz%2Fjenkins%2Fjob%2Fnmfu%2Fjob%2Fmaster)](https://jenkins.mm12.xyz/job/nmfu) [![Jenkins tests](https://img.shields.io/jenkins/tests?compact_message&jobUrl=https%3A%2F%2Fjenkins.mm12.xyz%2Fjenkins%2Fjob%2Fnmfu%2Fjob%2Fmaster)](https://jenkins.mm12.xyz/jenkins/job/nmfu/job/master/lastCompletedBuild/testReport/) [![Jenkins Coverage](https://img.shields.io/jenkins/coverage/apiv4?jobUrl=https%3A%2F%2Fjenkins.mm12.xyz%2Fjenkins%2Fjob%2Fnmfu%2Fjob%2Fmaster)](https://jenkins.mm12.xyz/jenkins/job/nmfu/job/master/lastCompletedBuild/coverage/) [![Read the Docs](https://img.shields.io/readthedocs/nmfu)](https://nmfu.rtfd.io)

`nmfu` is a "protocol parser generator" -- it converts a DSL representing some protocol handler into an efficient interruptable C routine (based on a state machine) that parses it.

- Confused? Try reading the [tutorial](./tutorial/http1.md).
- Want to find specific information about nmfu? Check out the [reference](./user-ref/parser.md).

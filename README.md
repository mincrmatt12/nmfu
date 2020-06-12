# nmfu

---
_the "no memory for you" "parser" generator_

---

`nmfu` attempts to turn a parser specified as a procedural matching thing into a state machine. It's much more obvious what it
does if you read some of the examples.

It takes in a "program" containing various match expressions, control structures and actions and converts it into a DFA with actions
on the transitions. This allows simple protocols (for example HTTP) to be parsed using an extremely small memory footprint and without
requiring a separate task since it can be done character by character.

You can also define various output variables which can be manipulated inside the parser program, which can then be examined after parsing.
See `example/http.nmfu` for a good example of using this functionality.

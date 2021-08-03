<img src="https://user-images.githubusercontent.com/5255209/117226360-7e69a900-ade2-11eb-9127-4a146a443199.png" alt="nmfu logo banner" width="100%"/>

# nmfu

---
_the "no memory for you" "parser" generator_

---

![PyPI - License](https://img.shields.io/pypi/l/nmfu) [![PyPI](https://img.shields.io/pypi/v/nmfu)](https://pypi.org/project/nmfu) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nmfu) [![Jenkins](https://img.shields.io/jenkins/build?jobUrl=https%3A%2F%2Fjenkins.mm12.xyz%2Fjenkins%2Fjob%2Fnmfu%2Fjob%2Fmaster)](https://jenkins.mm12.xyz/job/nmfu) [![Jenkins tests](https://img.shields.io/jenkins/tests?compact_message&jobUrl=https%3A%2F%2Fjenkins.mm12.xyz%2Fjenkins%2Fjob%2Fnmfu%2Fjob%2Fmaster)](https://jenkins.mm12.xyz/jenkins/job/nmfu/job/master/lastCompletedBuild/testReport/) [![Jenkins Coverage](https://img.shields.io/jenkins/coverage/api?jobUrl=https%3A%2F%2Fjenkins.mm12.xyz%2Fjenkins%2Fjob%2Fnmfu%2Fjob%2Fmaster%2F)](https://jenkins.mm12.xyz/jenkins/job/nmfu/job/master/lastCompletedBuild/coverage/cobertura__coverage_xml/project/_/nmfu_py/) [![nmfu](https://snapcraft.io//nmfu/badge.svg)](https://snapcraft.io/nmfu)

`nmfu` attempts to turn a parser specified as a procedural matching thing into a state machine. It's much more obvious what it
does if you read some of the examples.

It takes in a "program" containing various match expressions, control structures and actions and converts it into a DFA with actions
on the transitions. This allows simple protocols (for example HTTP) to be parsed using an extremely small memory footprint and without
requiring a separate task since it can be done character by character.

You can also define various output variables which can be manipulated inside the parser program, which can then be examined after parsing.
See `example/http.nmfu` for a good example of using this functionality.

The rest of this README is a guide to using NMFU.

## Parser Specification

NMFU source files support C++-style line comments (text after `//` on a line is ignored until the end of the line)

### Top-Level Constructs

At the top-level, all NMFU parsers consist of a set of output-variables, macro-definitions, hook-definitions and the parser code itself.

The output-variables are specified with the _output-declaration_:

```lark
out_decl: "out" out_type IDENTIFIER ";"
        | "out" out_type IDENTIFIER "=" atom ";"

out_type: "bool" -> bool_type
        | "int" -> int_type
        | "enum" "{" IDENTIFIER ("," IDENTIFIER)+ "}" -> enum_type
        | "str" "[" NUMBER "]" -> str_type
```

For example: 

```
out int content_length = 32;
out bool supports_gzip = false;

// note you can't set default values for strings and enums

out str[32] url;
out enum{GET,POST} method;
```

All strings have a defined maximum size, which includes the null-terminator.

Macros in NMFU are simple parse-tree level replacements. They look like:

```lark
macro_decl: "macro" IDENTIFIER macro_args "{" statement* "}"

macro_args: "(" macro_arg ("," macro_arg)* ")"
          | "(" ")" -> macro_arg_empty

macro_arg: "macro" IDENTIFIER -> macro_macro_arg
         | "out"   IDENTIFIER -> macro_out_arg
         | "match" IDENTIFIER -> macro_match_expr_arg
         | "expr"  IDENTIFIER -> macro_int_expr_arg
         | "hook"  IDENTIFIER -> macro_hook_arg
         | "loop"  IDENTIFIER -> macro_breaktgt_arg
```

For example:

```
macro ows() { // optional white space
   optional {
      " ";
   }
}
```

When macros are "called", or instantiated, all NMFU does is copy the contents of the parse tree from the macro
declaration to the call-site. Note that although macros can call other macros, they cannot recurse.

Macros can also take arguments, which are similarly treated as parse-tree level replacements, with the added restriction
that their types _are_ checked. For example:

```
macro read_number(out target, match delimit) {
    target = 0;
    foreach {
        /\d+/;
    } do {
        target = [target * 10 + ($last - '0')];
    }

    delimit;
}
```

There are 6 types of arguments:

- `macro`: a reference to another macro
- `hook`: a reference to a hook
- `out`: a reference to an _output-variable_
- `match`: an arbitrary _match-expression_
- `expr`: an arbitrary _integer-expression_
- `loop`: an arbitrary named _loop-statement_, for use in _break-statements_.

Hooks (which are callbacks to user code which the parser can call at certain points) are defined with a _hook-declaration_:

```lark
hook_decl: "hook" IDENTIFIER ";"
```

For example:

```
hook got_header;
```

### Parser Declaration

The parser proper is declared with the _parser-declaration_,

```lark
parser_decl: "parser" "{" statement+ "}"
```

and contains a set of statements which are "executed" in order to parse the input.

### Basic Statements

Basic statements are statements which do not have an associated code block, and which end with a semicolon.

```lark
simple_stmt: expr -> match_stmt
           | IDENTIFIER "=" expr -> assign_stmt
           | IDENTIFIER "+=" expr -> append_stmt
           | IDENTIFIER "(" (expr ("," expr)*)? ")" -> call_stmt
           | "break" IDENTIFIER? -> break_stmt
           | "finish" -> finish_stmt
           | "wait" expr -> wait_stmt
```

The most basic form of statement in NMFU is a _match-statement_, which matches any _match-expression_ (explained in the next section).

The next two statements are the _assign-statement_ and _append_statement_. The _assign-statement_ parses an _integer-expression_ (which are not limited to just integers, again explained in the next section).
and assigns its result into the named _output-variable_. The _append-statement_ instead appends whatever is matched by the _match-expression_ into the named _output-variable_ which must by a string type. Additionally,
if the argument to an _append-statement_ is a _math-expression_, then the result of evaluating the expression will be treated as a character code and appended to the string.

The _call-stmt_ instantiates a macro or calls a hook. Note that there is currently no valid way to pass parameters to a hook, and as such the expressions provided
in that case will be ignored. Macro arguments are always parsed as generic expressions and then interpreted according to the type given to them at declaration.

If a hook and macro have the same name, the macro will take priority. Priority is undefined if a macro argument and global hook or macro share a name.

The _break-statement_ is explained along with loops in a later section.

The _finish-statement_ causes the parser to immediately stop and return a `DONE` status code, which should be interpreted by the calling application as a termination condition.

The _wait-statement_ spins and consumes input until the _match-expression_ provided matches successfully. Importantly, no event (including end of input!) can stop the
wait statement, which makes it useful primarily in error handlers. 

It is also important to note that this is _not_ the same as using a regex like `/.*someterminator/`, as
the wait statement does _not_ "try" different starting positions for a string when its match fails. More concretely, something like `wait abcdabce` would _not_ match `abcdabcdabce`, as
the statement would bail and restart matching from the beginning at the second `d`.

### Expressions

There are three types of expressions in NMFU, _match-expressions_, _integer-expressions_ and _math-expressions_.

A _match-expression_ is anything that can consume input to the parser and check it:

```lark
?expr: atom // string match
     | regex // not an atom to simplify things
     | "end" -> end_expr
     | "(" expr+ ")" -> concat_expr

atom: STRING "i" -> string_case_const
    | STRING -> string_const
```

The simplest form of _match-expression_ is the _direct-match_, which matches a literal string. It can optionally match with case insensitivity by suffixing the literal string with an "i".

The _end-match-expression_ is a match expression which only matches the end of input.

The _concat-expression_ matches any number of _match-expressions_ in order.

The _regex-expression_ matches a subset of regexes. The quirks of the regex dialect NMFU uses can be found in a later section.

Additionally, the name of a macro argument of type `match` can replace any _match-expression_, including the sub-expressions inside _concat-expressions_.

An _integer-expression_ is anything that can be directly assigned to an output variable, **including strings**:

```lark
?expr: atom 
     | "[" sum_expr "]"

atom: BOOL_CONST -> bool_const
    | NUMBER -> number_const
    | CHAR_CONSTANT -> char_const
    | STRING -> string_const
    | IDENTIFIER -> enum_const
```

The only two kinds of _integer-expressions_ are _literal-expressions_, which are just literal strings, integers, character constants
(which behave as they do in C, just becoming integers), enumeration constants (which are resolved based on the context and which _output-variable_ is being assigned) and booleans ("true"/"false"); and _math-expressions_, which are surrounded in square brackets:

```lark
?sum_expr: mul_expr (SUM_OP mul_expr)*
?mul_expr: math_atom (MUL_OP math_atom)*
?math_atom: NUMBER -> math_num
          | IDENTIFIER -> math_var
          | CHAR_CONSTANT -> math_char_const
          | "$" IDENTIFIER -> builtin_math_var
          | "(" sum_expr ")"
```

(named `sum_expr` in the grammar)

_math-expressions_ are effectively any arithmetic with either normal numbers, or references to _output-variables_ (referencing their current value) or _builtin-math-variables_.

The current list of _builtin-math-variables_ is:

| Name | Meaning |
| ---- | ------- |
| `last` | The codepoint of the last matched character. Useful for interpreting numbers in input streams. |

For example:

```
content_length = [content_length * 10 + ($last - '0')];
```

Additionally, the name of a macro argument of type `expr` can replace any _integer-expression_. Priority versus _output-variable_ names is undefined.

### Block Statements

Block statements are statements which contain a block of other statements:

```lark
block_stmt: "loop" IDENTIFIER? "{" statement+ "}" -> loop_stmt
          | "case" "{" case_clause+ "}" -> case_stmt
          | "optional" "{" statement+ "}" -> optional_stmt
          | "try" "{" statement+ "}" catch_block -> try_stmt
          | "foreach" "{" statement+ "}" "do" "{" foreach_actions "}" -> foreach_stmt

catch_block: "catch" catch_options? "{" statement* "}"

case_clause: case_predicate ("," case_predicate)* "->" "{" statement* "}"
case_predicate: "else" -> else_predicate
              | expr -> expr_predicate

catch_options: "(" CATCH_OPTION ("," CATCH_OPTION)* ")" 

CATCH_OPTION: /nomatch|outofspace/
```

_loop-statements_ repeat their block forever until broken out of using a _break-statement_. _loop-statements_ can optionally have names, which
can be referred to in the _break-statement_ to break out of nested loops. If a bare _break-statement_ is encountered, the innermost loop is broken from.

_optional-statements_ either execute their contents if the first match within them matches, otherwise do nothing. It is an error to have anything
that does not match as the first statement in an _optional-statement_.

_case-statements_ match one or more _match-expressions_ and upon successful matching of one of those expressions execute a given block.

For example:

```
case {
   "GET" -> {method = GET;}
   "POST" -> {method = POST;}
   "PUT","PATCH" -> {wait "\r\n"; result=INVALID_METHOD; finish;}
   else, "OPTIONS" -> {}
}
```

The special constant "else" means anything not matched by any of the other clauses.

_try-except-statements_ can redirect certain error conditions to the beginning of a different block. They follow the general structure of

```
try {
   // ... some set of statements ...
}
catch {
   // ... an error handler ...
}
```

The specific error conditions which they match can be limited by placing a parenthesized comma-separated list of _catch-options_ after the "catch" token, like

```
try {
   url = /\w+/;
}
catch (outofspace) {
   // ... something to deal with this ...
}
```

_foreach-statements_ allow you to run various statements for every character read by some other set of statements. For example:

```
number = 0;
foreach {
   /\d+/;
} do {
   number = [number * 10 + ($last - '0')];
}
```

which will read a number in base 10 and store it into a variable. It accomplishes this by executing `number = [number * 10 + ($last - '0')];` for each digit. Only
statements which do _not_ consume any input themselves are allowed in the `do` section of a _foreach-statement_. 

### NMFU Regexes

The regex grammar in NMFU is

```lark
regex: "/" regex_alternation "/"

regex_char_class: "\\" REGEX_CHARCLASS

?regex_alternation: regex_group ("|" regex_group)*

?regex_group: regex_alternation_element+

?regex_alternation_element: regex_literal
                          | regex_literal REGEX_OP -> regex_operation
                          | regex_literal "{" NUMBER "}" -> regex_exact_repeat
                          | regex_literal "{" NUMBER "," NUMBER "}" -> regex_range_repeat
                          | regex_literal "{" NUMBER "," "}" -> regex_at_least_repeat

?regex_literal: REGEX_UNIMPORTANT -> regex_raw_match // creates multiple char classes
              | regex_char_class // creates a char class
              | "(" regex_alternation ")" -> regex_group
              | "[^" regex_set_element+ "]" -> regex_inverted_set // creates an inverted char class
              | "[" regex_set_element+ "]" -> regex_set // creates a char class
              | "." -> regex_any // creates an inverted char class

?regex_set_element: REGEX_CHARGROUP_ELEMENT_RAW
                  | REGEX_CHARGROUP_ELEMENT_RAW "-" REGEX_CHARGROUP_ELEMENT_RAW -> regex_set_range
                  | regex_char_class

REGEX_UNIMPORTANT: /[^.?*()\[\]\\+{}|\/]|\\\.|\\\*|\\\(|\\\)|\\\[|\\\]|\\\+|\\\\|\\\{|\\\}|\\\||\\\//
REGEX_OP: /[+*?]/
REGEX_CHARGROUP_ELEMENT_RAW: /[^\-\]\\\/]|\\-|\\\]|\\\\|\\\//
REGEX_CHARCLASS: /[wWdDsSntr ]/
```

NMFU regexes support the following common features:

- matching characters
- matching character classes (`\w`, `\S`, `\d`)
- matching character sets (`[ab]`, `[a-z\w]`, `[^b]`)
- the plus, star and question operators
- the wildcard dot
- groups
- alternation (also called OR) (`abc|def`)
- repetition

There are some key differences though:

- There is no direct way to match the end of input, even with the normal regex syntax `$` (it just matches the literal `$`)
- The wildcard dot matches _every single input except end_
- Groups are non-capturing, in the sense that they serve no function other than to logically group components together
- The space character must be escaped, due to limitations in the lexer.

## Generated C API

The output of NMFU is a C source file and header pair. All declarations within the header are prefixed with the output name provided
to NMFU.

The generated API contains two or three functions, depending on whether or not dynamic memory was used in the parser. These are:

- `{parser_name}_result_t {parser_name}_start({parser_name}_state_t * state);`
- `{parser_name}_result_t {parser_name}_feed (const uint8_t *start, const uint8_t *end, {parser_name}_state_t * state);`
- `{parser_name}_result_t {parser_name}_end  ({parser_name}_state_t * state);` (only generated if EOF support is on)
- `void                   {parser_name}_free ({parser_name}_state_t * state);` (only generated if dynamic memory is used)

These are used to initialize the parser state, send input characters to the parser, send an end condition (if enabled) and free any allocated resources
for the parser (if present).

The `_feed` function can be optionally configured to instead take a `const uint8_t **` (an "indirect" start pointer), in which case the pointed-to pointer
will be incremented as characters are read. The final value depends on the return value:

- `_DONE`: the `start` pointer now points to the last character that was read (_not_ like how end points to one past the last character)
- `_FAIL`: the `start` pointer now points to the first invalid character
- `_OK`:   the `start` pointer is now at `end` (the buffer was exhausted before either `_DONE` or `_FAIL`)

The generated `{parser_name}_state_t` struct contains the member `c` within which all of the declared _output-variables_ are accessible. Additionally,
the length of all string variables can be found under members with the name `{out_var_name}_counter`.

For example, the generated state structure for 

```
out str[32] url;
out bool accept_gzip = false;
out bool has_etag = false;
out str[32] etag;
out enum{GET,POST} method;
out enum{OK,BADR,URLL,ETAGL,BADM} result;
out int content_size = 0;
```

looks like:

```c
// state object
struct http_state {
    struct {
        char url[32];
        bool accept_gzip;
        bool has_etag;
        char etag[32];
        http_out_method_t method;
        http_out_result_t result;
        int32_t content_size;
    } c;
    uint8_t url_counter;
    uint8_t etag_counter;
    uint8_t state;
};
typedef struct http_state http_state_t;
```

Additional enumeration types for every declared enum _output-variable_ are also generated, using the name `{parser_name}_out_{out_var_name}_t`. The names
of the constants use, in all-capitals, `{parser_name}_{out_var_name}_{enum_constant}`; for example `HTTP_RESULT_URLL`.

If hooks were used, they are either declared (but not implemented) as functions with the signature

```
void {parser_name}_{hook_name}_hook({parser_name}_state_t * state, uint8_t last_inval);
```

or added as members to the state object as function pointers (with the same signature, typedef'd as `{parser_name}_hook_t`) with the name `{hook_name}_hook` outside
of the `c` sub-struct, depending on command line options.

One additional enumeration is always defined, called `{parser_name}_result_t` which contains the various result codes from NMFU-generated functions.
These contain the values `{parser_name}_DONE`, `{parser_name}_OK` and `{parser_name}_FAIL`, for example `HTTP_OK`.

- `OK` means that more input should be sent.
- `FAIL` means that the parser has entered a failing state. This is the default behaviour if no _try-except-statement_ is present.
- `DONE` means that either a _finish-statement_ executed or the parser has reached the end of it's statements.

### Example Usage

A basic usage of a generated parser looks something like

```c

http_state_t state;
http_start(&state); // could potentially return something other than OK if, for example, if the first statement in the parser was "finish" for some reason.

const uint8_t *in = "GET";
http_feed(in, in + 3, &state);

// or, in indirect mode:
uint8_t *place = in;
http_feed(&place, in + 3, &state);

// ...

http_result_t code = http_end(&state);

// ... do something with the code and state ...

http_free(&state);
```

A more complete example is present in `example/http_test.c`.

## Plugins

There is a vim plugin available which adds syntax highlighting for `.nmfu` files at [mincrmatt12/nmfu-vim](https://github.com/mincrmatt12/nmfu-vim).

## License

NMFU is licensed under the GPLv3.
Copyright (C) 2020-2021 Matthew Mirvish.

# Tutorial: A Basic Lexer

!!! note
	The features in this tutorial are new in NMFU 0.5.0

This tutorial introduces the `yield` and `greedy case` constructs in the context of building a lexer.

!!! note
    While NMFU does a reasonable job at being a lexer, for more serious lexing needs you may wish to investigate [re2c](https://re2c.org).

## The basic structure

Our lexer will be tokenizing a very simple LISP-ish language. Specifically, we'll be generating the following tokens:

- LP/RP: left right parentheses 
- SYMBOL: some alphanumeric word
- NUMBER: some integer

and will be ignoring whitespace.

For example, we want to convert `(1 -2 (test asdf32 ) 4)` into the sequence `LP NUMBER NUMBER LP SYMBOL SYMBOL RP NUMBER RP`.

To start, we'll use a loop and a case statement to match the various tokens:

```nmfu
parser {
	loop {
		case {
			/\s+/ -> {} // ignore white space
			"(" -> {} // left paren
			")" -> {} // right paren
			/[a-zA-Z_]\w*/ -> {} // symbol (not starting with a number)
			/-?\d+/ -> {} // signed integer
		}
	}
}
```

However, this leaves us with the question of how to report the tokens to our application. While we could use the `hook` mechanism, there's
another, cleaner way.

## The yield statement

In the [HTTP tutorial](./http2.md), we introduced custom return codes. Here, we'll introduce another similar mechanism. The `yield` statement allows
the parser to pause execution, return back to the caller with a custom status code and then be resumed. 

We can use this to send the stream of tokens out of our parser:

```nmfu
yieldcode LP, RP, SYMBOL, INTEGER;

parser {
	loop {
		case {
			/\s+/ -> {} // ignore white space
			"(" -> {yield LP;} // left paren
			")" -> {yield RP;} // right paren
			/[a-zA-Z_]\w*/ -> {yield SYMBOL;} // symbol (not starting with a number)
			/-?\d+/ -> {yield INTEGER;} // signed integer
		}
	}
}
```

However, in order for this to compile, we must explicitly enable yield support with the `-fyield-support` flag:

```
$ nmfu -fyield-support lexer.nmfu
```

This also enables the [indirect start pointer](/user-ref/cli#indirect-start-pointer) mode, which changes the signature of the `_feed` function to

```c
lexer_result_t lexer_feed(lexer_state_t * state, const uint8_t **start, const uint8_t * end);
```

or in other words, makes the start pointer a double pointer, so that the parser can modify it. The reason why should be fairly obvious: if the parser
had no way of communicating how far into the input it got before yielding, there'd be no way to properly resume it at the right point. Luckily, accommodating this difference is usually fairly trivial.

Now, we can write our C code. Let's start with some basic boilerplate to read lines of standard in, as well as initialize the parser state:

```c
#include <stdio.h>
#include <stdlib.h>
#include "lexer.h"

int main(void) {
	lexer_state_t state = {};
	lexer_start(&state);

	uint8_t *line = NULL;
	size_t len = 0;
	ssize_t nread;

	while (true) {
		// prompt user
		printf("> ");
		// grab more input
		if ((nread = getline((char **)&line, &len, stdin)) == -1) break;

		// ... parse line ...
	}

	free(line);
}
```

We need to keep track of the start pointer separately to the line (since we need the line pointer separately to free it later), so let's keep track of it in
a variable called `pos`. 

```c
const uint8_t *pos = line;
```

Now, we can start doing some actual parsing. The structure here is very simple: keep running `lexer_feed` until either it returns `_LEXER_FAIL`, indicating a syntax error,
`_LEXER_OK`, which indicates we've parsed the entire line, or `_LEXER_DONE`, which theoretically shouldn't happen with our current grammar but would indicate reaching
the end of the parser.

```c
lexer_result_t code;

do {
	switch (code = lexer_feed(&pos, line + nread, &state)) {
		case LEXER_FAIL:
			fprintf(stderr, "\nSYNTAX ERROR\n");
			return 1;
		case LEXER_DONE: return 0;
		default: break;
	}
} while (code != LEXER_OK);
```

Notice here that we've given a pointer to `pos` so that the parser can update it. At this point, our mini-lexer will now run without problems, but we still haven't
got it actually reporting tokens!

Luckily, all we have to do is add some more cases to our switch:

```c
printf(":"); // show start of "parsed" line

do {
	switch (code = lexer_feed(&pos, line + nread, &state)) {
		case LEXER_FAIL:
			fprintf(stderr, "\nSYNTAX ERROR\n");
			return 1;
		case LEXER_YIELD_INTEGER:
			printf(" INTEGER");
			break;
		case LEXER_YIELD_LP:
			printf(" (");
			break;
		case LEXER_YIELD_RP:
			printf(" )");
			break;
		case LEXER_YIELD_SYMBOL:
			printf(" SYMBOL");
			break;
		case LEXER_DONE: return 0;
		default: break;
	}
} while (code != LEXER_OK);
puts(""); // end line of output
```

Notice that the `yieldcode`s we declared earlier have been translated into additional result codes, with names `LEXER_YIELD_{code}`. 

We should now have a functional (if a little primitive lexer):

```
$ nmfu -fyield-support lexer.nmfu
$ cc main.c lexer.c
$ ./a.out
> (1 -2 (test asdf32 ) 4) 
: ( INTEGER INTEGER ( SYMBOL SYMBOL ) INTEGER )
^D
$ 
```

## Retrieving token contents: interpreting the start pointer

One of the great things about the `yield` mechanism is that you can directly inspect where in the input stream the parser is at. We can use this to extract the start and
end of tokens.

The value of `pos` after a `yield` is defined to be the character immediately following the last character consumed by a match. In our example, the last match 
is always the predicate of the case statement, so the value of `pos` after a `yield` will always be the character _one after_ the end of a token -- and critically,
also the start of the _next_ token.

We can use this to our advantage by keeping track of an additional pointer to the beginning of the current token:

```c
const uint8_t *pos = line, *begin = line;

do {
	switch (code = lexer_feed(&pos, line + nread, &state)) {
		case LEXER_FAIL:
			fprintf(stderr, "\nSYNTAX ERROR\n");
			return 1;
		case LEXER_YIELD_INTEGER:
			printf(" INTEGER(%.*s)", (pos - begin), begin);
			break;
		case LEXER_YIELD_LP:
			printf(" (");
			break;
		case LEXER_YIELD_RP:
			printf(" )");
			break;
		case LEXER_YIELD_SYMBOL:
			printf(" SYMBOL(%.*s)", (pos - begin), begin);
			break;
		case LEXER_DONE: return 0;
		default: break;
	}

	begin = pos;
} while (code != LEXER_OK);
puts("");
```

Here, we keep track of the beginning of the current token in the `begin` pointer, and update it after each token is read. We can then access the contents of a token
by reading between the `begin` and `pos` pointers, which we do here with some `printf` magic.

If we try this, we can see that while it does work, it has one major flaw:

```
$ ./a.out
> (1 -2 (test asdf32 ) 4) 
: ( INTEGER(1) INTEGER( -2) ( SYMBOL(test) SYMBOL( asdf32) ) INTEGER( 4) )
^D
$ 
```

We're leaking whitespace into our tokens! This is happening since while the NMFU side of our parser is ignoring the whitespace, it's never telling the C side to reset the `begin`
pointer once it consumes some. Luckily, this is trivially easy to solve. All we need to do is yield some bogus non-token after reading any characters we wish to ignore, for example:

```nmfu
yieldcode LP, RP, SYMBOL, INTEGER;
yieldcode _RESET; // ignored, only used to mark ignored chars

parser {
	loop {
		case {
			/\s+/ -> {yield _RESET;} // ignore white space
			// ... rest of parser ...
```

Changing the NMFU side is all we need to do since our C code is setup to ignore all unknown yields, but still reset the begin pointer in those cases.

Using this version, we now have a more desirable result:

```
$ ./a.out
> (1 -2 (test asdf32 ) 4) 
: ( INTEGER(1) INTEGER(-2) ( SYMBOL(test) SYMBOL(asdf32) ) INTEGER(4) )
^D
$ 
```

## Token priorities and ambiguities: Greedy case statements

What if we wanted to allow symbols to start with digits? Specifically, let's say that something comprised entirely of digits is a number, but any other alphanumeric word is a symbol.

If we try to naively adjust the case statement, say to something like

```nmfu
parser {
	loop {
		case {
			/\s+/ -> {} // ignore white space
			"(" -> {yield LP;} // left paren
			")" -> {yield RP;} // right paren
			/\w*[a-zA-Z_]\w*/ -> {yield SYMBOL;} // symbol (containing at least one non-numeric character)
			/-?\d+/ -> {yield INTEGER;} // signed integer
		}
	}
}
```

NMFU will correctly point out that this is ambiguous and refuse to compile it:

```
$ nmfu -fyield-support lexer.nmfu
Compile error: Ambigious case label: should finish or check next. If you mean to finish, use a greedy case.
Due to:
- line 13:
			/-?\d+/ -> {yield INTEGER;}
			^
- line 12:
			/\w*[a-zA-Z_]\w*/ -> {yield SYMBOL;}
```

The error here is telling us that we have an _ambiguous case label_ -- that is, for some input NMFU is unsure as to which case branch it should execute -- and that it's confused
since it can't tell whether to _finish or check next_. By "finish", it means "finish matching predicate and begin executing case body", and by "check next" it means "continue
matching characters against predicates". 

Luckily, the error message also tells us how to solve our problem. We want to resolve the conflict _greedily_, i.e. that whichever match finishes first should be taken.
We indicate this by making the case greedy:

```nmfu
greedy case {
	/\s+/ -> {} // ignore white space
	"(" -> {yield LP;} // left paren
	")" -> {yield RP;} // right paren
	/\w*[a-zA-Z_]\w*/ -> {yield SYMBOL;} // symbol (containing at least one non-numeric character)
	/-?\d+/ -> {yield INTEGER;} // signed integer
}
```

And this now does what we want:

```
$ ./a.out
> (1 -2 (54test asdf32 ) 4) 
: ( INTEGER(1) INTEGER(-2) ( SYMBOL(54test) SYMBOL(asdf32) ) INTEGER(4) )
^D
$ 
```

What if we wanted to add some special keywords to the lexer? Let's try making the symbol `define` have some special meaning:

```nmfu
yieldcode LP, RP, SYMBOL, INTEGER, DEFINE;

// ...

greedy case {
	/\s+/ -> {} // ignore white space
	"(" -> {yield LP;} // left paren
	")" -> {yield RP;} // right paren
	/\w*[a-zA-Z_]\w*/ -> {yield SYMBOL;} // symbol (containing at least one non-numeric character)
	/-?\d+/ -> {yield INTEGER;} // signed integer
	"define" -> {yield DEFINE;} // keyword
}
```

This, unfortunately, gives us a different error:

```
$ nmfu -fyield-support lexer.nmfu
Compile error: Ambigious case label: multiple possible finishes with same priority 0
Due to:
- line 12:
			/\w*[a-zA-Z_]\w*/ -> {yield SYMBOL;}
			^
- direct match 'define':
  at line 17:
				"define" -> {yield DEFINE;}
				^
```

The confusion NMFU has here is more obvious: the input "define" could be both a `SYMBOL` _or_ a `DEFINE`, and we haven't told NMFU explicitly which to prefer, so it throws an error alerting
us to the ambiguity. Luckily, greedy case statements let us solve this too, with the `prio` construct:

```nmfu
greedy case {
	/\s+/ -> {} // ignore white space
	"(" -> {yield LP;} // left paren
	")" -> {yield RP;} // right paren
	/\w*[a-zA-Z_]\w*/ -> {yield SYMBOL;} // symbol (containing at least one non-numeric character)
	/-?\d+/ -> {yield INTEGER;} // signed integer

	prio 1 {
		"define" -> {yield DEFINE;} // keyword
	}
}
```

The `prio` block assigns an integral priority to one or more case branches. Whichever one has the unique maximum in some ambiguity scenario is chosen, and unannotated branches have a default
priority of 0.

If we inform our C code of the new token type (left as an exercise to the reader), we can see that this works:

```
$ ./a.out
> (define a (b 4 5))
: ( DEFINE SYMBOL(a) ( SYMBOL(b) INTEGER(4) INTEGER(5) ) )
^D
$
```

It is here that we conclude this tutorial, although there are some potential other exercises:

- Try and parse the integers directly in NMFU and report them back to the C code with an `out`. You'll probably need to either make symbols not start with numbers or
use some clever foreach setups to make this work. 
- Experiment with what happens if you add extra `yield` statements at other places in the parser body.

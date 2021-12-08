# CLI Reference

This page goes into slightly more detail about some of the options exposed
by NMFU's CLI; for a full list use `nmfu --help`.

## Optimization

NMFU by default only performs minimal simplification on the parser you give it
before generating C code; the `-O` option (used like a C compiler, i.e. `-O3` for full
optimization) can be used to enable more optimizations than the default of level `1`.

Specific optimizations (listed in the help under `Optimization Flags`) can be
enabled/disabled with the normal `-fthing/-fno-thing` mechanism.

## User Pointers

The `-finclude-user-ptr` option will add an extra `void * userptr` to the state object that
can be used by the calling code to store whatever.

## Dynamic Memory

By default, NMFU tries to be completely static in its memory allocation (since this
usually gives the most flexibility to the downstream program, especially in the
original use-case on embedded systems), however this can be customized if desired. Currently,
only strings support this mechanism. With the `-fallocate-str-space-dynamic` option,
strings are stored outside of the main state struct and are instead `malloc`'d separately.
With the `-fallocate-str-space-dynamic-on-demand` option, the memory is only allocated
on first write, and the `-fdelete-string-free-memory` will cause the `delete` statement
to `free` the string when it is empty if desired.

If you enable any of the dynamic memory options, NMFU creates an additional `parser_free` function
which will clean up all of the dynamically allocated variables in the state object.

## Indirect Start Pointer

By enabling `-findirect-start-ptr`, the `_feed` function instead takes a `const uint8_t **` for the `start` pointer in which case the pointed-to pointer
will be incremented as characters are read. The final value depends on the return value:

- `_DONE`: the `start` pointer now points to the last character that was read (_not_ like how end points to one past the last character)
- `_FAIL`: the `start` pointer now points to the first invalid character
- `_OK`:   the `start` pointer is now at `end` (the buffer was exhausted before either `_DONE` or `_FAIL`)

This can be useful for sending data after a parser error to an external error handler, or for
correctly dealing with sequential messages.

## Per-instance Hooks

By default, hooks are declared as global functions, however the `-fhook-per-state` instead adds them
as function-pointer members in the state object.

## EOF Support

NMFU has support for explicit "end of file/input" handling. If `-feof-support` is turned on,
the `end` match can be used to match the end of file condition, which can be useful
for properly handing truncated messages in certain contexts.

A future version will likely generalize this mechanism to allow for "events", e.g.
"serial parity error" or "connection idle" injected in a similar way (or perhaps
as exceptions).

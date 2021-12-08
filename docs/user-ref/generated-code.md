# Generated API

The output of NMFU is a C source file and header pair. All declarations within the header are prefixed with the output name provided
to NMFU.

The generated API contains two or three functions, depending on whether or not dynamic memory was used in the parser. These are:

- `{parser_name}_result_t {parser_name}_start({parser_name}_state_t * state);`
- `{parser_name}_result_t {parser_name}_feed (const uint8_t *start, const uint8_t *end, {parser_name}_state_t * state);`
- `{parser_name}_result_t {parser_name}_end  ({parser_name}_state_t * state);` (only generated if EOF support is on)
- `void                   {parser_name}_free ({parser_name}_state_t * state);` (only generated if dynamic memory is used)

These are used to initialize the parser state, send input characters to the parser, send an end condition (if enabled) and free any allocated resources
for the parser (if present).

The generated `{parser_name}_state_t` struct contains the member `c` within which all of the declared _output-variables_ are accessible. Additionally,
the length of all string variables can be found under members with the name `{out_var_name}_counter`.

For example, the generated state structure for 

```nmfu
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

## Example Usage

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


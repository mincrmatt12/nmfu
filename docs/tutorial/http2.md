# HTTP Continued: Headers with Loops

The parser created in the [last tutorial](./http1.md), while functional, is lacking a fairly important component: parsing request headers.

Currently, our parser looks something like 

```nmfu
out enum{GET, POST, UNSUPPORTED} method;
out str[32] request_path;
out bool uri_too_long = false;

parser {
   case {
      "GET " -> {method = GET;}
      "POST " -> {method = POST;}
      else -> {method = UNSUPPORTED; wait " ";}
   }
   "/";

   try {
      request_path += /[\/a-zA-Z0-9.\-_?=&+]+/;
   }
   catch (outofspace) {
      uri_too_long = true;
      wait "\r\n\r\n";
      finish;
   }

   " HTTP/1."; /\d/;

   wait "\r\n\r\n";
}
```

Now, we'll try to make it handle a few headers. Before we do that though, let's clean up
our error handling a bit.

## Basic Macros

You may already see a repeated pattern starting to emerge in our parser -- waiting for `\r\n\r\n` on an error condition.
Repeating patterns like this can make the parser source ugly, and so NMFU offers a basic form of code reuse with _macros_:

For example, we could add 

```nmfu
macro waitend() {
   wait "\r\n\r\n";
   finish;
}
```

and replace the body of our error handler with just 

```nmfu
uri_too_long = true;
waitend();
```

Since our header parsing is going to add more error conditions, it'd be nice to consolidate all the error conditions.
One approach we could use would be to replace the ad-hoc boolean flag with an `error_code` enum:

```nmfu
out enum{OK, URI_TOO_LONG} result_code;

macro waitend(expr result) {
   result_code = result;
   wait "\r\n\r\n";
   finish;
}

parser {
   result_code = OK;
   // ...
```

The `expr` here means an _integer-expression_, (as opposed to a _match-expression_), effectively anything that can be
stored into a scalar variable (for more information, see the [section on expressions in the reference](../user-ref/parser.md#expressions)).

Now we can succinctly set an error code and ignore the rest of a request with just one line:

```nmfu
catch (outofspace) {
   waitend(URI_TOO_LONG);
}
```

!!! note
    We won't cover the C implementation of reading this enum here; if you can't figure it out go back to the previous
    tutorial and read the section on handling HTTP request methods again -- the technique here is very similar.

NMFU, however, provides a more intelligent system for this. Consider this slightly different version of the above example:

```nmfu
finishcode URI_TOO_LONG;

macro waitend(finishcode result) {
	wait "\r\n\r\n";
	finish result;
}

parser {
	// .. parser code ..
	catch (outofspace) {
		waitend(URI_TOO_LONG);
	}
}
```

The `finishcode` directive defines an additional result that the `http_feed` function can return. When a statement like `finish URI_TOO_LONG;` is executed, instead
of simply returning `HTTP_DONE`, it instead returns `HTTP_FINISH_URI_TOO_LONG`. This lets us clean up the C side of our HTTP parser:

```c
switch (http_server_feed(&parser, buf, buf + count)) {
	case HTTP_SERVER_OK:
		continue;
	case HTTP_SERVER_FAIL:
		printf("HTTP/1.1 400 Bad Request\r\n\r\n");
		return 0;
	case HTTP_SERVER_FINISH_URI_TOO_LONG:
		printf("HTTP/1.1 414 URI Too Long\r\n\r\n");
		return 0;
	case HTTP_SERVER_OK:
		goto finished;
}
```

## Basic Authorization

Alright, now let's try to parse headers. The first block construct we'll introduce is the _loop-statement_. It's very simple:
it runs its body forever until a _break-statement_ is executed. We'll use this to implement a loop where each
iteration consumes exactly one header (including the newline). 

Consider this example:

```nmfu
wait "\r\n"; // read to end of request URI

loop {
   case {
      "\r\n" -> {finish;} // end of request (two newlines in a row)
      "Authorization: "i -> {
         // todo
         wait "\r\n";
      }
      else -> {
         // ignore header
         wait "\r\n";
      }
   }
}
```

We start by reading to the end of the request line. Then, we loop. If we immediately encounter another newline, we've
clearly got two in a row, which is the end of the request. If we see `Authorization: ` (the `i` suffix indicating case-insensitive
matching, since header names are case-insensitive), we could parse the authorization. Otherwise, if we see a header
we don't recognize, we read up to the newline ending it.

This simple loop structure will read all headers and stop exactly at the end of the request, just like we want.

Next, we'll fill in the body of the `Authorization: ` block. Let's add another output to store the received
authorization data (which for basic authorization is just a base64'd string) and another error code for improperly
formatted authorization (we'll leave verifying interpretable credentials to the C code):

```nmfu
out str[60] auth_payload;
finishcode URI_TOO_LONG, MALFORMED_AUTH;
```

Then, we'll fill in the case handler:

```nmfu
"Authorization: "i -> {
   delete auth_payload;
   try {
      "Basic "i;
      auth_payload += /[a-zA-Z0-9+\/=]+/;
      "\r\n";
   }
   catch {
      waitend(MALFORMED_AUTH);
   }
}
```

This introduces a few new things: the `delete` statement, which just clears a string output, and the bare form
of the `catch` clause, which just means "catch all errors".

## String length: Checking for authorization presence

Unfortunately, we have a bit of a problem if we try to use this parser in C: how do we know if an authorization
header was included? While we could just use a `strcmp`, since the string is null-terminated and initialized to
zeroes, what if we wanted to use the `unterminated str` mode in NMFU? 

Well, NMFU exposes the length of any string-like output (which includes `raw` outputs covered in a later section) as
a separate member in the state object. Instead of being inside the `c` subobject, it's at the root level and
called `auth_payload_counter` (the name of the output + `_counter`) 

So, we can check for authorization by just seeing if `parser.auth_payload_counter > 0`:

```c
// assume VALID_AUTH is some correct base64 authorization string

if (parser.auth_payload_counter == 0) {
   // send a 401 with a WWW-Authenticate header
}
else if (strcmp(parser.c.auth_payload, VALID_AUTH)) {
   // send a 401 invalid auth
}
else {
   // auth ok
}
```

## More advanced loops: Checking for `gzip` support

What if we want our HTTP server to be more bandwidth-efficient? One of the ways we could accomplish this is by
supporting response compression, perhaps by storing a pre-compressed version of all our pages and sending
them instead of the plain version if the client asks.

The HTTP protocol supports this with the `Accept-Encoding` header. Its value is a comma-separated list of supported
encodings, where we're looking for `gzip`.

There are a number of ways we could handle this. We could use a regex that matches all comma-separated lists
containing gzip and then throw that into a case statement, but let's use a loop to match each item instead. You
might imagine expanding this approach later to support matching more than one kind of encoding, and being able
to go through all of them (perhaps to prioritize the smallest one) would be useful.

Let's ignore the fact it's comma-separated for now, and try a basic approach:

```nmfu
out bool supports_gzip = false;

// ... [snip] ...

case {
   "Accept-Encoding: "i -> {
      loop {
         case {
            "\r\n" -> {break;}  // end of list
            "gzip"i -> { supports_gzip = true; }
            else -> {} // ignore everything else
         }
      }
   }
   // ...
```

This might look sensible, but if we try to compile it, we get an error:

```
Compile error: Infinite loop due to self-referential fallthrough
Due to:
- line 36:
                        case {
                        ^
```

!!! note
    NMFU's errors can be a little off in terms of position; the error should really be pointing at the `else` of the case statement. 

This raises the question: what's a "self-referential fallthrough"? Well, when an `else` matches, it's supposed to "send" the first nonmatching
character to the body of the `else` condition (so that the character isn't silently eaten by the `else`), however the next statement in this case
is the case statement itself. So, the nonmatching character _falls through_ back to the case statement, goes to the `else`, which falls through again,
etc.; forming an infinite loop.

The solution is very simple. We just need to explicitly discard the nonmatching character, which can be accomplished by just placing a "match any character"
regex in the body of the `else` condition:

```nmfu
case {                                  
   "\r\n" -> {break;}  // end of list   
   "gzip"i -> { supports_gzip = true; } 
   else -> {/./;} // ignore everything else 
}                                       
```

This, unlike the previous example, will actually compile and work. Making this actually verify the list is comma-separated is left as an exercise to the reader.

Notice also that we've used a nested loop with a `break`. By default `break` exits the innermost loop, however it can break out of an outer loop by using the loop
name mechanism:

```nmfu
loop outer {
   loop inner {
      break outer;
   }
}
```

would break out of the outer loop instead.

## Foreach: Reading base-10 numbers

What if we want to support POST request bodies? We'd need to read the `Content-Length` header to figure out how long the payload is, which is encoded as an
ASCII integer. How can we read an integer in this format with NMFU?

First, let's define an output to store the length:

```nmfu
out int content_length = -1; // -1 representing not received
```

Then, we'll introduce a new statement, the _foreach-statement_. Unlike the _loop-statement_ which continually runs some statements, the _foreach-statement_ instead
executes a series of non-matching statements after each character is read by some other potentially matching statements.

For example,

```nmfu
"Content-Length: "i -> {
   content_length = 0;
   foreach {
      /\d+/;
   } do {
      content_length = [content_length * 10 + ($last - '0')];
   }
   wait "\r\n";
}
```

This will run the statement `content_length = [content_length * 10 + ($last - '0')]` _for each_ each character in the number is read by the `/\d+/` regex. The math
syntax here is also new. The square brackets are to indicate a math expression; the only part inside them that should be new is the `$last` constant. The value of
`$last` is whatever the byte value of the last read character is -- so in this case, the digit of the number. Some basic ASCII math later and we're parsing
the entire number as desired.

Reading this from C is, as you probably have guessed, similar to before:

```c
printf("got %d bytes of payload", parser.c.content_length);
```

This concludes the second and final part of the http tutorial.

#include <stdio.h>
#include <string.h>
#include "http.h"

int main(int argc, char ** argv) {
	http_state_t state;
	http_start(&state);
	http_result_t code;

	while (true) {
		char c = getchar();
		if (c == '\n') {
			http_feed('\r', false, &state);
		}
		if (c == EOF) break;
		printf("input %c\n", c);
		printf("return code %d, ", code = http_feed((uint8_t)c, false, &state));
		printf("state %d\n", state.state);
		if (code != HTTP_OK) goto end;
	}

	puts("final");
	printf("return code %d, ", http_feed(0, true, &state));
	printf("state %d\n", state.state);

end:
	printf("method %d; has_gzip %d; has_etag %d; url %s; etag %s; content_size %d; result %d\n", state.c.method, state.c.accept_gzip, state.c.has_etag, state.c.url, state.c.etag, state.c.content_size, state.c.result);

	printf("and it only uses %lu bytes!\n", sizeof(state));
	puts("response:");

	if (state.c.result == HTTP_RESULT_URLL) {
		puts("HTTP/1.1 414 URI Too Long");
		puts("Content-Length: 9");
		puts("Content-Type: text/plain");
		puts("");
		puts("too long");
	}
	else if (state.c.result == HTTP_RESULT_ETAGL) {
		puts("HTTP/1.1 431 Request Header Fields Too Large");
		puts("Content-Length: 15");
		puts("Content-Type: text/plain");
		puts("");
		puts("too long (etag)");
	}
	else if (state.c.result == HTTP_RESULT_BADR || state.c.result == HTTP_RESULT_BADM) {
		puts("HTTP/1.1 400 Bad Request");
		puts("Content-Length: 11");
		puts("Content-Type: text/plain");
		puts("");
		puts("bad request");
	}
	else {
		if (strncmp(state.c.url, "/root", 32) == 0) {
			puts("HTTP/1.1 200 OK");
			puts("Content-Length: 11");
			puts("Content-Type: text/plain");
			if (state.c.accept_gzip) puts("Content-Encoding: gzip");
			puts("");
			puts("root content");
		}
		else {
			puts("HTTP/1.1 404 Not Found");
			puts("Content-Length: 9");
			puts("Content-Type: text/plain");
			puts("");
			puts("not found");
		}
	}

	http_free(&state);
}

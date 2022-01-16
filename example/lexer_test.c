#include "lexer.h"
#include <stdio.h>
#include <stdlib.h>

void show_between(const uint8_t *begin, const uint8_t *end) {
	while (begin != end) {
		putchar(*begin++);
	}
}

int main(void) {
	lexer_state_t state = {};
	lexer_start(&state);

	uint8_t *line = NULL;
	size_t len = 0;
	ssize_t nread;

	while (true) {
		// grab more input
		if ((nread = getline((char **)&line, &len, stdin)) == -1) break;
		printf(":");
		
		lexer_result_t code;
		const uint8_t *pos = line, *end = line + nread, *begin = pos;
		do {
			switch (code = lexer_feed(&pos, end, &state)) {
				case LEXER_FAIL:
					fprintf(stderr, "\nSYNTAX ERROR\n");
					return 1;
				case LEXER_YIELD_INTEGER:
					printf(" ");
					show_between(begin, pos);
					printf("=INTEGER");
					break;
				case LEXER_YIELD_LP:
					printf(" (");
					break;
				case LEXER_YIELD_RP:
					printf(" )");
					break;
				case LEXER_YIELD_DEFINE:
					printf(" define");
					break;
				case LEXER_YIELD_DISPLAY:
					printf(" display");
					break;
				case LEXER_YIELD_SYMBOL:
					printf(" ");
					show_between(begin, pos);
					printf("=SYM");
					break;
				case LEXER_DONE: return 0;
				default: break;
			}
			begin = pos;
		} while (code != LEXER_OK);
		puts("");
	}

	free(line);
}

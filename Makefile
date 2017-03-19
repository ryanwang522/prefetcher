CFLAGS = -msse2 --std gnu99 -O0 -Wall -Wextra

GIT_HOOKS := .git/hooks/applied

EXEC = naive sse_transpose sse_prefetch_transpose

all: $(GIT_HOOKS) $(EXEC) main.c
	$(CC) $(CFLAGS) -o main main.c

$(GIT_HOOKS):
	@scripts/install-git-hooks
	@echo

naive: main.c
	$(CC) $(CFLAGS) -DNAIVE -o naive main.c

sse_transpose: main.c
	$(CC) $(CFLAGS) -DSSE -o sse_transpose main.c

sse_prefetch_transpose: main.c
	$(CC) $(CFLAGS) -DSSE_PREFETCH -o sse_prefetch_transpose main.c

clean:
	$(RM) main naive sse_transpose sse_prefetch_transpose

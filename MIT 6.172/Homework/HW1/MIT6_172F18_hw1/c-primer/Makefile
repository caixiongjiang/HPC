CC := clang

ifeq ($(DEBUG),1)
	CFLAGS := -Wall -O0 -g
else
	CFLAGS := -Wall -O1 -DNDEBUG 
endif

LDFLAGS := -lrt -flto -fuse-ld=gold

all: sizes pointer swap

sizes.o: sizes.c
	$(CC) $(CFLAGS) -c sizes.c

sizes: sizes.o
	$(CC) -o sizes sizes.o $(LDFLAGS)

pointer.o: pointer.c
	$(CC) $(CFLAGS) -c pointer.c

pointer: pointer.o
	$(CC) -o pointer pointer.o $(LDFLAGS)

swap.o: swap.c
	$(CC) $(CFLAGS) -c swap.c

swap: swap.o
	$(CC) -o swap swap.o $(LDFLAGS)
clean:
	rm -f sizes pointer swap *.o *.gcda *.gcno *.gcov perf.data */perf.data cachegrind.out.*

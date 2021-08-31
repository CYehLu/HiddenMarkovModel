OBJS = hmm.o helper.o

all: $(OBJS)

%.o: %.c %.h types.h
	gcc -c $< -std=c99 -o $@ -lm

.PHONY: clean
clean:
	rm -f *.o


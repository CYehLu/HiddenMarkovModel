OBJS = hmm.o
EXES = likelihood

all : $(EXES)

likelihood: main_likelihood.c hmm.o
	gcc $^ -std=c99 -lm -o $@

%.o: %.c %.h
	gcc -c $< -std=c99 -o $@ -lm

.PHONY: clean
clean:
	rm hmm.o $(EXES)

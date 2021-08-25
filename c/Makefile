EXES = simulate likelihood decoding learning
OBJS = hmm.o helper.o

all : $(EXES)

likelihood: main_likelihood.c $(OBJS)
	gcc $^ -std=c99 -lm -o $@

decoding: main_decoding.c $(OBJS)
	gcc $^ -std=c99 -lm -o $@
    
learning: main_learning.c $(OBJS)
	gcc $^ -std=c99 -lm -o $@

simulate: main_simulate.c $(OBJS)
	gcc $^ -std=c99 -lm -o $@

%.o: %.c %.h
	gcc -c $< -std=c99 -o $@ -lm

.PHONY: clean
clean:
	rm $(OBJS) $(EXES)

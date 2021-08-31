EXES = simulate likelihood decoding learning
OBJS = hmm.o helper.o
OBJPATH = ../src/
OBJLINKER = $(addprefix $(OBJPATH),$(OBJS))

all: $(EXES)

%: %.c $(OBJLINKER)
	gcc $^ -I$(OBJPATH) -std=c99 -lm -o $@

$(OBJLINKER):
	$(MAKE) -C $(OBJPATH)

.PHONY: clean
clean: 
	$(MAKE) -C $(OBJPATH) clean
	rm -f $(EXES)

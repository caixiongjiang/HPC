# Copyright (c) 2012 MIT License by 6.172 Staff

# List all of your source files here (but not your headers), separated by
# spaces.  You'll have to add to this list every time you create a new
# source file.
SRC := testbed.c matrix_multiply.c

# Set the name of your binary.  Change it if you like.
PRODUCT := matrix_multiply

################################################################################
# These configuration options change how your code (listed above) is compiled
# every time you type "make".  You may have to change these values to complete
# some assignments; you should also feel free to experiment with them.
################################################################################

# This option sets which compiler your code will be compiled with.  Likely
# choices are icc, icpc, gcc, g++, clang
CC := clang

# These flags will be applied to your code any time it is built.
CFLAGS := -Wall -std=c99 -D_POSIX_C_SOURCE=200809L

# These flags are applied only if you build your code with "make DEBUG=1".  -g
# generates debugging symbols, -DDEBUG defines the preprocessor symbol "DEBUG"
# (so that you can use "#ifdef DEBUG" in your code), and -O0 disables compiler
# optimizations, so that the binary generated more directly corresponds to your
# source code.
CFLAGS_DEBUG := -g -DDEBUG -O0

# In the release version, we ask for many optimizations; -O3 sets the
# optimization level to three.  -DNDEBUG defines the NDEBUG macro,
# which disables assertion checks.
CFLAGS_RELEASE := -O1 -DNDEBUG

# These flags are used to invoke Clang's address sanitizer.
CFLAGS_ASAN := -O1 -g -fsanitize=address

# These flags are applied when linking object files together into your binary.
# If you need to link against libraries, add the appropriate flags here.  By
# default, your code is linked against the "rt" library with the flag -lrt;
# this library is used by the timing code in the testbed.
LDFLAGS := -lrt -flto -fuse-ld=gold

################################################################################
# You probably won't need to change anything below this line, but if you're
# curious about how makefiles work, or if you'd like to customize the behavior
# of your makefile, go ahead and take a peek.
################################################################################

# You shouldn't need to touch this.  This keeps track of whether you are
# building in a release or debug configuration, and sets CFLAGS appropriately.
# (This mechanism is based on one from the original materials for 6.197 by
# Ceryen Tan and Marek Olszewski.)
OLDMODE=$(shell cat .buildmode 2> /dev/null)
ifeq ($(DEBUG),1)
  CFLAGS := $(CFLAGS_DEBUG) $(CFLAGS)
  ifneq ($(OLDMODE),debug)
    $(shell echo debug > .buildmode)
  endif
else ifeq ($(ASAN),1)
  CFLAGS := $(CFLAGS_ASAN) $(CFLAGS)
  LDFLAGS := $(LDFLAGS) -fsanitize=address
  ifneq ($(OLDMODE),asan)
    $(shell echo asan > .buildmode)
  endif
else
  CFLAGS := $(CFLAGS_RELEASE) $(CFLAGS)
  ifneq ($(OLDMODE),nodebug)
    $(shell echo nodebug > .buildmode)
  endif
endif

# When you invoke make without an argument, make behaves as though you had
# typed "make all", and builds whatever you have listed here.  (It knows to
# pick "make all" because "all" is the first rule listed.)
all: $(PRODUCT)

# This special "target" will remove the binary and all intermediate files.
clean::
	rm -f $(OBJ) $(PRODUCT) .buildmode \
        $(addsuffix .gcda, $(basename $(SRC))) \
        $(addsuffix .gcno, $(basename $(SRC))) \
        $(addsuffix .gcov, $(SRC) fasttime.h)

# This rule generates a list of object names.  Each of your source files (but
# not your header files) produces a single object file when it's compiled.  In
# a later step, all of those object files are linked together to produce the
# binary that you run.
OBJ = $(addsuffix .o, $(basename $(SRC)))

# These rules tell make how to automatically generate rules that build the
# appropriate object-file from each of the source files listed in SRC (above).
%.o : %.c .buildmode
	$(CC) $(CFLAGS) -c $< -o $@
%.o : %.cc .buildmode
	$(CC) $(CFLAGS) -c $< -o $@
%.o : %.cpp .buildmode
	$(CC) $(CFLAGS) -c $< -o $@

# This rule tells make that it can produce your binary by linking together all
# of the object files produced from your source files and any necessary
# libraries.
$(PRODUCT): $(OBJ) .buildmode
	$(CC) -o $@ $(OBJ) $(LDFLAGS)

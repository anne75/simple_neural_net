#
# Makefile for the MNIST neural network
#

# compiler to tuse
CC = g++

# name of the executable
EXE = mnist

# list of source files
SRCS = neural_network.cpp mnist_nn.cpp read_mnist.cpp

# list of libraries
LIBS = -lnlopt -larmadillo

# headers
HDRS = neural_network.h

# automatically generated list of object files
OBJS = $(SRCS:.c=.o)

# Do not mistake rules for files
.PHONY: clean all re

# default target
all: $(OBJS) $(HDRS) Makefile
	$(CC) -O3 -o $(EXE) $(OBJS) $(LIBS)

# dependencies
$(OBJS): $(HDRS) Makefile

# housekeeping
clean:
	-$(RM) *.o $(EXE) *~

re:
	clean all

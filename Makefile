#
# Makefile for the MNIST neural network
#

# compiler to tuse
CC = g++

# name of the executable
EXE = mnist

# list of source files
g++ neural_network.cpp neural_wrap.cpp mnist_nn.cpp read_mnist.cpp -lnlopt -larmadillo


# Simple Neural Network
Single hidden layer neural net in C++ adapted from Coursera Machine Learning.  

[AI](https://en.wikipedia.org/wiki/Artificial_intelligence) is a buzzword of our times. It's everywhere, and it sounds both really 
cool and super smart. It is also regularly accompanied with *something* neural network. I wanted to know more, or more accurately remember about it.   
As I was trying to see if I liked programming 3 years ago I took the first class that was opened at [UCSC Extension](https://www.ucsc-extension.edu/), and it was about machine learning.
I thoroughly enjoyed that class, and went on to take more classes both at the UCSC extension and online before attending Holberton School.
One of those courses was the famous [Machine Learning](https://www.coursera.org/learn/machine-learning) on Coursera.   
I decided this spring after much coding which had nothing to do with data, but a lot with me learning, to loop the loop and try to re-code
the course neural network but this time in C++.  
Why C++ ? Mostly to learn a new language, and I had heard it had great linear algebra libraries.
Here is my attempt.  

In the folder, the class assignment as I did it 3 years ago.  
Some files:  
- ```make_training_sample.cpp```: from Dave Miller as well, how to make an `XOR` gate to test the neural network.  
- ```mnist_nn.cpp```: the entry point for the neural network.  
- ```neural_network.cpp```: all the neural network related functions.   
- ```neural_netowrk_training.cpp```: another neural network from [Dave Miller](http://millermattson.com/dave).  
- ```read_mnist.cpp```: adapted from this [post](http://eric-yuan.me/cpp-read-mnist/)   
_____
### To use
**Libraries**  
This neural network relies on the [Armadillo](http://arma.sourceforge.net/) linear algebra library and [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt_C-plus-plus_Reference) optimization library.  
**Data**  
Download on Yann Lecun [website](http://yann.lecun.com/exdb/mnist/).  

After download/installation you can use `Make all` to compile and then `./mnist` to run. 
If you want to play with the various parameters, they are in ```mnist_nn.cpp```.   

### Acknowledgement  
To help me get started with this project I used this [repo](https://github.com/joedivita/Machine-Learning-Series/blob/master/Part6-Neural-Net-Backpropagation/main6.cpp).

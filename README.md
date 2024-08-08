This repo is a very simple implementation of deep neural networks written in the Java programming language.

It does not utilize tensors or any automatic differentiation library as it is only composed of fully connected feed forward layers.

To solve MNIST, create a new directory "data" in the root of the project, download the MNIST files from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/), and unzip and store the files in the "data" directory. Then run:

```
javac src/*.java
java -cp src Main
```

If the above url is not working for you, checkout: [https://github.com/cvdfoundation/mnist](https://github.com/cvdfoundation/mnist/).

## Maven Build Instructions

To build and run the project using Maven, follow these steps:

1. Ensure you have Maven installed on your system. You can download it from [https://maven.apache.org/download.cgi](https://maven.apache.org/download.cgi).

2. Open a terminal and navigate to the root directory of the project.

3. Run the following command to compile the project:

```
mvn compile
```

4. To run the project, use the following command:

```
mvn exec:java -Dexec.mainClass="Main"
```

here's some benchmark results from fiddling with hyper parameters etc 
[spreadsheet](https://docs.google.com/spreadsheets/d/1825onhH1uPmXJjqjOOZLDkHyJ_BaYP1LKGJzq5vDf4s/edit?usp=sharing)

they're messy and look something like this: ![image](https://github.com/user-attachments/assets/3f8ad5c4-2677-4d49-ae10-74fa578447ee)

feel free to give suggestions on namings etc in there, but it's mostly for personal tests and fiddling, perhaps eventually this whole project will be an actual library :)

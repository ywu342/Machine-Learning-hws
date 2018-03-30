All the code are embedded within the ABAGAIL source code itself along with some modified source code, so if you try to use the online master version of ABAGAIL, it will not work with my code. You need to open an IDE and import the provided ABAGAIL-master as a project folder such that you can run below described code directly with an IDE tool. Running the code output data I used to draw the graphs.

Corresponding files for two parts:
part 1: 
All the experiment code is in ABAGAIL-master/src/opt/test/NN_RandOPT.java. The dataset car.data is also under the same directory such that java will be able to find it. 

To run part one: Optimization for neural network
- make sure you have a directory named "outputs/" under the root directory of ABAGAIL-master
- make sure car.data is under the directory as is because the path to find it is hardcoded in the code.

part 2: 
relevant code: 
ABAGAIL-master/src/opt/test/FourPeaksTest.java
ABAGAIL-master/src/opt/test/FlipFlopTest.java 
ABAGAIL-master/src/opt/test/TravelingSalesmanTest.java

To run part two: Three Optimization problems
- make sure you have a directory named "outputs/" under the root directory of ABAGAIL-master

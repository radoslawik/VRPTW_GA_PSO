## GA and PSO for Vehicle Routing Problem with Time Windows

### Overview
Application is divided into four modules with different areas to cover:
-  Data preprocessing - transformation of a selected problem instance to a structure that can be processed in the further modules;
-  Core functions - methods used within both algorithms, such as calculating fitness, generating initial population of individuals or particles, updating particles velocity and position, performing crossover and mutation;
-  Algorithms creator - functions allowing to start both algorithms with desired number of iterations and other corresponding parameters such as population size, social and cognitive acceleration coefficients, particle speed limit, crossover and mutation probability. Algorithms are coded with two DEAP containers support: Toolbox and Creator;
-  Application execution - script to initialize parameters and run specified problem instance and chosen algorithm. Results are printed using Statistics and Logbook tools from DEAP
framework;

### Parameters
There are various parameters that can be modified in order to optimize and compare the
performance of both algorithms. They can be divided into four categories:
-  input data properties (for both algorithms),
-  iterations, population size and fitness function coefficients population size (for both algorithms),
-  genetic operators probability (for GA),
-  properties of particles (for PSO).

### Quick start
All the parameters can be changed in the `run.py` file. To start the algorithm simply simply run this this file with the problem name (R101, ...) and chosen algorithm (GA/PSO) as arguments. For example:
```
python run.py R101 GA
```

### Things to consider
The algorithm parameters are not optimized and results are often quite poor.  Also the way the individual is coded could be improved.

### References

This project was inspired by: https://github.com/iRB-Lab/py-ga-VRPTW

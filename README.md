# Differential Privacy Solver

A simple algorithm to break a differential privacy implementation that uses a binomial distribution to add noise to its aggregated data members.

## Pseudocode & Results

All of the necessary documentation is within the docs/ folder.

## Structure of Code

The Problem class defines the problem that is trying to be solved, whilst the Solver class will contain the solve method and take in parameters that define what the Solver can use and what type of algorithm the solver should use to figure out the guesses. The Solver never gets to look at anything other than the a list and the w list, if it is using guesses with the algorithm. 

The program will first build the allowable inputs into lists and then loop over every combination of them. The accuracies from the number of iterations will have their mean and stdev computed and have it placed into a .csv file, whilst the accuracies list will be pushed alongside necessary metadata into a matplotlib scatter plot. The scatter plot will be output into a .png file for viewing, but when the script is ran the graph will display through the matplotlib library.

## Installing

Make sure you are running Python >=3.9, then run the following commands to setup the environment.

```
python -m venv env

env\Scripts\activate.bat (Windows)

pip install -r requirements.txt
```

This should setup the project environment.

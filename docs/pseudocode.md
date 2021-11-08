# Algorithm Pseudocode

I won't discuss the implementation of the release mechanism or the problem, because its implementation shouldn't matter to the solver at all. The only thing necessary for solver should be the answers (a) list that compiles the aggregated results that are output by the DP algorithm.

Alongside the a list, part b) requires that there exist a guesses (w) list that contains guesses that have a 2/3 chance of being correct.

## Pseudocode
```
for i from (0 to num_entries) {
    // calculate the percentage chance that the user clicked the ad by computing a_i / (2 * i)
    chance = a_i / (2 * i) 
    
    guesses = new list

    if using guesses  {
        // calculate whether the chance clicked is greater than the chance of a guess being right
        if guess_chance > chance and chance > (1 - guess_chance) // we must check both bounds, as guess_chance
                                                                 // implies it is correct for its value, but chance implies it is a 1 if it is > 0.5
            add guess value
        else {
            add 1 to guesses if chance >= .5 otherwise add a 0
        }
    }

    else {
        add 1 to guesses if chance >= 0.5 otherwise add a 0
    }

    return the guesses
}
```

after the guesses are computed, compare the x array with the guesses to get the accuracy of the algorithm

## Code

Code is present in main.py. The main solver is a class called Solver, which takes in the parameters for the problem, such as whether it uses the guesses or not. There is associated documentation in the form of docstrings that convey how the code structure is set up.

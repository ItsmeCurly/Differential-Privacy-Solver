import functools
import itertools
import statistics

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt


class Problem:
    """A class that defines the Problem (the list of randomized clicks for the problem
    itself. This will handle the release mechanism alongside generating all of the
    necessary lists (guesses, binomial distribution, etc.))
    """

    def __init__(self, num_entries: int) -> None:
        self.num_entries = num_entries

        self._x = self._build_x(num_entries)
        self._z = self._build_z(num_entries)

        self._a = self._build_answers(num_entries)

        logger.debug(f"{self.x=}\n{self.z=}\n{self.a=}")

    def __len__(self):
        return self.num_entries

    @property
    def x(self):
        return self._x

    actual = x

    @property
    def z(self):
        return self._z

    @property
    def a(self):
        return self._a

    # Alias answers to a, allow self.answers call
    answers = a

    @property
    def guess_percentage(self):
        return 2 / 3

    @functools.cached_property
    def w(self):
        correct_guesses = np.random.binomial(1, self.guess_percentage, self.num_entries)

        res = np.zeros(self.num_entries)

        for i, n in enumerate(res):
            if correct_guesses[i] == 1:
                res[i] = self._x[i]
            else:
                res[i] = self._x[i] + np.random.randint(low=0, high=self.z[i] + 1)

        return res

    def _build_x(self, length: int) -> np.ndarray:
        return np.random.randint(2, size=length)

    def _compute_binom(self, num_flips: int) -> int:
        return np.random.binomial(num_flips, p=0.5)

    def _build_z(self, length: int) -> np.ndarray:
        res = np.zeros(shape=length, dtype=int)

        for i, _ in enumerate(res):
            res[i] = self._compute_binom(i)

        return res

    def _build_answers(self, length):
        res = np.zeros(length, dtype=int)

        sigma_vals = list(itertools.accumulate(self.x))

        for i, _ in enumerate(res):
            res[i] = sigma_vals[i] + self.z[i]
            logger.debug(f"{res[i]} = {sigma_vals[i]} + {self.z[i]}")

        return res

    def get_guess(self, i: int) -> int:
        """Returns a guess given the specified index to access it from

        Args:
            i (int): The index

        Returns:
            int: The specified guess (w_i) at that index
        """
        return self.w[i]


class Solver:
    """A class that facilitates "solving" the specified problem. Takes in parameters
    that define what functionality it has access to when solving the problem.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        algorithm_type: str = "logic",
        use_guesses: bool = False,
    ):
        self.problem = problem
        self.algorithm_type = algorithm_type
        self.use_guesses = use_guesses

    def solve(self):
        """Will attempt to solve the problem. The solver will iterate over the answers
        list and generate a 1:1 list in size that contains the solver's guesses to the
        answers.

        Raises:
            Exception: Unknown Algorithm type specified

        Returns:
            list: A list of the guesses (binary between 0 or 1)
        """
        guesses = []

        for i, a_i in enumerate(self.problem.answers):
            if i > 0:
                if self.algorithm_type == "logic":
                    # Thanks to Mac Creamer for helping with determing this algorithm
                    likelihood_clicked = a_i / (2 * i) 

                    if self.use_guesses:
                        if (
                            (1 - self.problem.guess_percentage)
                            <= likelihood_clicked
                            <= self.problem.guess_percentage
                        ):
                            guesses.append(self.problem.get_guess(i))
                        else:
                            guesses.append(1 if likelihood_clicked >= 0.5 else 0)
                    else:
                        guesses.append(1 if likelihood_clicked >= 0.5 else 0)
                elif self.algorithm_type == "one":
                    if self.use_guesses:
                        guesses.append(self.problem.get_guess(i))
                    else:
                        guesses.append(1)
                elif self.algorithm_type == "zero":
                    if self.use_guesses:
                        guesses.append(self.problem.get_guess(i))
                    else:
                        guesses.append(0)
                else:
                    raise Exception("Unknown algorithm type")
            else:
                guesses.append(a_i)

        return guesses


def _compute_accuracy(actual, guesses):
    correct = 0

    for answer, guess in zip(actual, guesses):
        if answer == guess:
            correct += 1

    return correct / len(actual)


def benchmark(entries: int, use_guesses: bool, iters: int = 20):
    """Iterates over the problem, generating a new problem and solver for each
    iteration to then solve. Will return the accuracy that the solver got for
    the problem.

    Args:
        entries (int): The number of entries
        use_guesses (bool): Whether the solver can use guesses
        iters (int, optional): The iterations that the process will go through.
        Defaults to 20.
    """

    def _benchmark():
        for _ in range(iters):
            problem = Problem(entries)

            solver = Solver(problem, use_guesses=use_guesses)

            guesses = solver.solve()

            accuracy = _compute_accuracy(problem.actual, guesses)

            yield accuracy

    return list(_benchmark())


def write_mean_stdev(
    accuracies: list[int], entries: int, use_guesses: bool, out_filename: str
):
    mean, stdev = statistics.mean(accuracies), statistics.stdev(accuracies)

    with open(out_filename, mode="a") as results_file:
        results_file.write(f"{entries},{int(use_guesses)},{mean:.4f},{stdev:.4f}\n")


def run_tests(
    *,
    display_graph: bool = False,
    iters: int = 20,
    graph_filename: str = "results.png",
    out_filename: str = "results.csv",
):
    entries = [50, 100, 500, 1000, 2500, 5000,]
    algorithm_type = ["logic",]
    use_guesses = [False, True,]

    res = {"x": [], "y": [], "c": []}

    with open(out_filename, mode="w") as results_file:
        results_file.write(f"entries,use_guesses,mean,stdev\n")

    fig, ax = plt.subplots()
    for entries, algo_type, use_guess in itertools.product(
        entries, algorithm_type, use_guesses
    ):
        accuracies = benchmark(entries, use_guess)

        write_mean_stdev(accuracies, entries, use_guess, out_filename)

        res["x"].extend([entries] * iters)
        res["y"].extend(accuracies)

        res["c"].extend([int(use_guess)] * iters)

    scatter = ax.scatter(**res)

    legend1 = ax.legend(
        *scatter.legend_elements(), loc="upper right", title="Using Guess"
    )
    
    ax.add_artist(legend1)
    ax.grid()
    plt.savefig(graph_filename)

    if display_graph:
        plt.show()


def main():
    logger.remove()
    logger.add("debug.log")

    run_tests()


if __name__ == "__main__":
    main()

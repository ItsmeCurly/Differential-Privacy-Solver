import numpy as np
import itertools
from loguru import logger
import functools
from matplotlib import pyplot as plt


class Problem:
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
        return self.w[i]


class Solver(object):
    def __init__(
        self,
        problem: Problem,
        algorithm_type: str = "logical",
        use_guesses: bool = False,
    ):
        self.problem = problem
        self.algorithm_type = algorithm_type
        self.use_guesses = use_guesses

    def solve(self):
        guesses = []

        for i, a_i in enumerate(self.problem.answers):
            if i > 0:
                if self.algorithm_type == "logic":
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


def benchmark(problem, solver=None, iters: int = 20):
    solver = solver or Solver(problem)

    def _benchmark():
        for _ in range(iters):
            guesses = solver.solve()

            accuracy = _compute_accuracy(problem.actual, guesses)

            yield accuracy

    return sum(_benchmark()) / iters


def run_tests(display=False):
    entries = [100, 500, 1000, 5000]
    algorithm_type = ["logic"]
    use_guesses = [False, True]

    res = {"x": [], "y": [], "c": []}

    fig, ax = plt.subplots()
    for entries, algo_type, use_guess in itertools.product(
        entries, algorithm_type, use_guesses
    ):
        problem = Problem(entries)

        solver = Solver(problem, algorithm_type=algo_type, use_guesses=use_guess)

        accuracy = benchmark(problem, solver)
        res["x"].append(entries)
        res["y"].append(accuracy)

        res["c"].append(int(use_guess))

        logger.info(f"{benchmark(problem, solver):.2f} {entries=} {algo_type=} {use_guess=}")

    plt.ylim([0, 1])

    scatter = ax.scatter(**res)
    print(scatter.legend_elements())
    legend1 = ax.legend(
        *scatter.legend_elements(), loc="upper right", title="Using Guess"
    )
    ax.add_artist(legend1)

    ax.grid()

    plt.savefig("out.png")
    if display:
        plt.show()


def main():
    logger.remove()
    logger.add("debug.log")

    run_tests()


if __name__ == "__main__":
    main()

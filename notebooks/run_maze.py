#!/usr/bin/env python

import argparse
from enum import Enum
import logging
from math import sqrt
import numpy as np
from time import sleep, time
from typing import Any, NamedTuple, Optional, Sequence, Union

# import Maze class from utility file for maze generation and display
from maze_utils import Maze

from skdecide import DeterministicPlanningDomain, Solver, Space, Value
from skdecide.builders.domain import Renderable, UnrestrictedActions
from skdecide.hub.solver.astar import Astar
from skdecide.hub.solver.lazy_astar import LazyAstar
from skdecide.hub.space.gym import EnumSpace, ListSpace, MultiDiscreteSpace


class State(NamedTuple):
    x: int
    y: int


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


# ### Domain type
# Then we define the domain type from a base template (`DeterministicPlanningDomain`) with optional refinements (`UnrestrictedActions` and `Renderable`). This corresponds to the following characteristics:
# - `DeterministicPlanningDomain`:
#     - only one agent
#     - deterministic starting state
#     - handle only actions
#     - actions are sequential
#     - deterministic transitions
#     - white box transition model
#     - goal states are defined
#     - positive costs (i.e. negative rewards)
#     - fully observable
#     - renderable (can be displayed)
# - `UnrestrictedActions`: all actions are available at each step
# - `Renderable`: can be displayed
#
# We also specify the type of states, observations, events, transition values, ...
#
# This is needed so that solvers know how to work properly with this domain, and this will also help IDE or Jupyter to propose you intelligent code completion.

# In[4]:


class D(DeterministicPlanningDomain, UnrestrictedActions, Renderable):
    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = None  # Type of additional information in environment outcome
    T_agent = Union  # Inherited from SingleAgent


# ### Actual domain class
# We can now implement the maze domain by
# - deriving from the above domain type
# - filling all non-implemented methods
# - adding a constructor to define the maze & start/end positions.
#
# We also define (to help solvers that can make use of it)
# - an heuristic for search algorithms
#
#
# *NB: To know the methods not yet implemented, one can either use an IDE which can find them automatically or the [code generators](https://airbus.github.io/scikit-decide/guide/codegen.html) page in the online documentation, which generates the corresponding boilerplate code.*

# In[5]:


class MazeDomain(D):
    """Maze scikit-decide domain

    Attributes:
        start: the starting position
        end: the goal to reach
        maze: underlying Maze object

    """

    def __init__(self, start: State, end: State, maze: Maze):
        self.start = start
        self.end = end
        self.maze = maze
        # display
        self._image = None  # image to update when rendering the maze
        self._ax = None  # subplot in which the maze is rendered

    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        """Get the next state given a memory and action.

        Move agent according to action (except if bumping into a wall).

        """

        next_x, next_y = memory.x, memory.y
        if action == Action.up:
            next_x -= 1
        if action == Action.down:
            next_x += 1
        if action == Action.left:
            next_y -= 1
        if action == Action.right:
            next_y += 1
        return (
            State(next_x, next_y)
            if self.maze.is_an_empty_cell(next_x, next_y)
            else memory
        )

    def _get_transition_value(
        self,
        memory: D.T_state,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> Value[D.T_value]:
        """Get the value (reward or cost) of a transition.

        Set cost to 1 when moving (energy cost)
        and to 2 when bumping into a wall (damage cost).

        """
        #
        return Value(cost=1 if next_state != memory else 2)

    def _get_initial_state_(self) -> D.T_state:
        """Get the initial state.

        Set the start position as initial state.

        """
        return self.start

    def _get_goals_(self) -> Space[D.T_observation]:
        """Get the domain goals space (finite or infinite set).

        Set the end position as goal.

        """
        return ListSpace([self.end])

    def _is_terminal(self, state: State) -> D.T_predicate:
        """Indicate whether a state is terminal.

        Stop an episode only when goal reached.

        """
        return self._is_goal(state)

    def _get_action_space_(self) -> Space[D.T_event]:
        """Define action space."""
        return EnumSpace(Action)

    def _get_observation_space_(self) -> Space[D.T_observation]:
        """Define observation space."""
        return MultiDiscreteSpace([self.maze.height, self.maze.width])

    def _render_from(self, memory: State, **kwargs: Any) -> Any:
        """Render visually the maze.

        Returns:
            matplotlib figure

        """
        # store used matplotlib subplot and image to only update them afterwards
        self._ax, self._image = self.maze.render(
            current_position=memory,
            goal=self.end,
            ax=self._ax,
            image=self._image,
        )
        return self._image.figure

    def heuristic(self, s: D.T_state) -> Value[D.T_value]:
        """Heuristic to be used by search algorithms.

        Here Euclidean distance to goal.

        """
        return Value(cost=sqrt((self.end.x - s.x) ** 2 + (self.end.y - s.y) ** 2))


# ### Domain factory

# To use scikit-decide solvers on the maze problem, we will need a domain factory recreating the domain at will.
#
# Indeed the method `solve_with()` used [later](#Training-solver-on-the-domain) needs such a domain factory so that parallel solvers can create identical domains on separate processes.
# (Even though we do not use parallel solvers in this particular notebook.)
#
# Here is such a domain factory reusing the maze created in [first section](#About-maze-problem). We render again the maze using the `render` method of the wrapping domain.

# In[6]:


def rollout(
    domain: MazeDomain,
    solver: Solver,
    max_steps: int,
    pause_between_steps: Optional[float] = 0.01,
):
    """Roll out one episode in a domain according to the policy of a trained solver.

    Args:
        domain: the maze domain to solve
        solver: a trained solver
        max_steps: maximum number of steps allowed to reach the goal
        pause_between_steps: time (s) paused between agent movements.
          No pause if None.

    """
    # Initialize episode
    solver.reset()
    observation = domain.reset()

    # loop until max_steps or goal is reached
    for i_step in range(1, max_steps + 1):
        if pause_between_steps is not None:
            sleep(pause_between_steps)

        # choose action according to solver
        action = solver.sample_action(observation)
        # get corresponding action
        outcome = domain.step(action)
        observation = outcome.observation

        # final state reached?
        if domain.is_terminal(observation):
            break

    # goal reached?
    is_goal_reached = domain.is_goal(observation)
    return is_goal_reached, i_step


def run(size: int, algos: Sequence[str], repeat: int) -> np.array:

    # generate the maze
    maze = Maze.generate_empty_maze(width=size, height=size)
    # starting position
    start = State(1, 1)
    # goal position
    end = State(size - 2, size - 2)
    # domain factory
    domain_factory = lambda: MazeDomain(maze=maze, start=start, end=end)
    # instanciate the domain
    domain = domain_factory()
    # init the start position
    domain.reset()

    # We set a maximum number of steps to reach the goal according to maze size in order to decide if the proposed solution is working or not.
    max_steps = maze.width * maze.height

    timings = np.zeros((repeat, 5), dtype=float)
    for k in range(repeat):
        if algos is None or "astar" in algos:
            name = "astar"
            with Astar(
                domain_factory=domain_factory, heuristic=lambda d, s: d.heuristic(s)
            ) as solver:
                t = time()
                MazeDomain.solve_with(solver, domain_factory)
                timings[k, 1] = time() - t
                logging.info(f"solve: {timings[k, 1]} {name}")
                t = time()
                found, nsteps = rollout(
                    domain=domain,
                    solver=solver,
                    max_steps=max_steps,
                    pause_between_steps=None,
                )
                timings[k, 2] = time() - t
                timings[k, 0] = nsteps
                logging.info(
                    f"rollout: {timings[k, 2]}, {name} - length : {nsteps} found? {found}"
                )

        if algos is None or "lazy_astar" in algos:
            name = "lazy_astar"
            with LazyAstar(
                from_state=start, heuristic=lambda d, s: d.heuristic(s)
            ) as solver:
                t = time()
                solver.solve(domain_factory=domain_factory)
                timings[k, 3] = time() - t
                logging.info(f"solve: {timings[k, 3]} {name}")
                t = time()
                found, nsteps = rollout(
                    domain=domain,
                    solver=solver,
                    max_steps=max_steps,
                    pause_between_steps=None,
                )
                timings[k, 4] = time() - t
                timings[k, 0] = nsteps
                logging.info(
                    f"rollout: {timings[k, 4]}, {name} - length : {nsteps} found? {found}"
                )

    return timings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Solve maze problem")
    parser.add_argument(
        "--size", type=int, required=False, default=101, help="maze size"
    )
    parser.add_argument(
        "--algo",
        type=str,
        action="append",
        required=False,
        choices=["astar", "lazy_astar"],
        help="algorithm to run (can be repeated)",
    )
    parser.add_argument(
        "--output", type=str, required=False, help="CSV file containing timing results"
    )
    parser.add_argument(
        "--repeat", type=int, required=False, default=10, help="number of runs"
    )
    args = parser.parse_args()
    results = run(size=args.size, algos=args.algo, repeat=args.repeat)
    if args.output:
        np.savetxt(args.output, results, fmt="%8.3e")

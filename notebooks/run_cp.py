from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain
from skdecide.hub.domain.rcpsp.rcpsp_sk import RCPSP
from skdecide.hub.solver.do_solver.do_solver_scheduling import DOSolver, PolicyMethodParams, SolvingMethod, BasePolicyMethod
from skdecide.discrete_optimization.generic_tools.cp_tools import ParametersCP
from skdecide.utils import rollout_episode

import argparse
import time

def solve_lns(sm_file:str):
    domain: RCPSP = load_domain(sm_file)
    domain.set_inplace_environment(False)
    state = domain.get_initial_state()

    params_cp = ParametersCP.default()
    params_cp.TimeLimit = 300
    dict_params = {"parameters_cp": params_cp, "max_time_seconds": 300, "verbose": False}

    policy_method_params = PolicyMethodParams(base_policy_method=BasePolicyMethod.FOLLOW_GANTT, delta_index_freedom=0, delta_time_freedom=0)

    solver = DOSolver(policy_method_params=policy_method_params,
                      method=SolvingMethod.LNS_CP,
                      dict_params=dict_params)

    possible_solvers = solver.get_available_methods(domain)
    print(f"Possible solvers: {possible_solvers}")

    tic = time.perf_counter()
    solver.solve(domain_factory=lambda: domain)
    toc = time.perf_counter()
    print(f"LNS_CP Time: {toc - tic:0.4f} seconds")


    states, actions, values = rollout_episode(domain=domain,
                                              solver=solver,
                                              from_memory=state,
                                              max_steps=1000,
                                              action_formatter=lambda a: f'{a}',
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print("rollout done")
    print('end times: ')
    for task_id in states[-1].tasks_details.keys():
        print('end task', task_id, ': ', states[-1].tasks_details[task_id].end)


def solve_cp(sm_file:str):
    domain: RCPSP = load_domain(sm_file)

    domain.set_inplace_environment(False)

    state = domain.get_initial_state()
    params_cp = ParametersCP.default()
    params_cp.TimeLimit = 200
    dict_params = {"parameters_cp": params_cp, "max_time_seconds": 200, "verbose": False}

    policy_method_params = PolicyMethodParams(base_policy_method=BasePolicyMethod.FOLLOW_GANTT, delta_index_freedom=0, delta_time_freedom=0)

    solver = DOSolver(policy_method_params=policy_method_params,
                      method=SolvingMethod.CP,
                      dict_params=dict_params)

    possible_solvers = solver.get_available_methods(domain)
    print(f"Possible solvers: {possible_solvers}")

    tic = time.perf_counter()
    solver.solve(domain_factory=lambda: domain)
    toc = time.perf_counter()
    print(f"CP Time: {toc - tic:0.4f} seconds")

    states, actions, values = rollout_episode(domain=domain,
                                              solver=solver,
                                              from_memory=state,
                                              max_steps=1000,
                                              action_formatter=lambda a: f'{a}',
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print("rollout done")
    print('end times: ')
    for task_id in states[-1].tasks_details.keys():
        print('end task', task_id, ': ', states[-1].tasks_details[task_id].end)


def main():
    parser = argparse.ArgumentParser(description='Solve RCPSP with CP.')
    parser.add_argument('--sm_file', type=str, required=True, help='SM file')
    args = parser.parse_args()

    solve_cp(args.sm_file)

    solve_lns(args.sm_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Local environment test — runs without the HTTP server.
Validates all 3 tasks end-to-end: reset, step loop, grader scores.

Run from project root:
    python scripts/test_local.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from envs.traffic_signal_env.environment import TrafficSignalEnvironment
from envs.traffic_signal_env.graders import grade_episode


def heuristic(obs):
    """Simple queue-length heuristic for baseline."""
    phases = {}
    for i in obs['intersections']:
        iid = i['intersection_id']
        if i.get('emergency_vehicle_present'):
            d = i.get('emergency_vehicle_direction', 'N')
            phases[iid] = 0 if d in ['N', 'S'] else 2
        elif i.get('pedestrian_demand'):
            phases[iid] = 4
        else:
            ns = i['queue_north'] + i['queue_south']
            ew = i['queue_east'] + i['queue_west']
            phases[iid] = 0 if ns >= ew else 2
    return phases


def run_task(task_id, seed=42, verbose=False):
    env = TrafficSignalEnvironment(task_id=task_id, seed=seed)
    obs = env.reset()

    total_reward = 0.0
    final_score = 0.0

    for step in range(1, env.network.max_steps + 1):
        phases = heuristic(obs)
        result = env.step({'phase_assignments': phases})
        total_reward += result['reward']
        obs = result

        if verbose and step % 20 == 0:
            print(f"  step={step} waiting={obs['total_vehicles_waiting']} "
                  f"throughput={obs['network_throughput']} reward={round(obs['reward'],3)}")

        if result['done']:
            final_score = result['metadata'].get('final_score', 0.0)
            break

    state = env.get_state()
    return {
        'task_id': task_id,
        'steps': state['step_count'],
        'throughput': state['total_throughput'],
        'cumulative_wait': round(state['cumulative_wait_time'], 1),
        'total_reward': round(total_reward, 3),
        'final_score': final_score,
    }


def main():
    print("=" * 60)
    print("Traffic Signal Control — Local Environment Test")
    print("=" * 60)

    tasks = [
        'single_intersection_easy',
        'arterial_corridor_medium',
        'urban_grid_hard',
    ]

    results = []
    all_passed = True

    for task in tasks:
        print(f"\nTask: {task}")
        r = run_task(task, seed=42, verbose=True)
        results.append(r)

        score = r['final_score']
        passed = 0.0 <= score <= 1.0
        all_passed = all_passed and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  score={score:.4f}  throughput={r['throughput']}  "
              f"wait={r['cumulative_wait']}  reward={r['total_reward']}")

    print("\n" + "=" * 60)
    print("GRADER VALIDATION — boundary inputs")
    print("=" * 60)

    # Verify graders stay in [0,1] for extreme inputs
    for task in tasks:
        for wait, tp in [(0, 9999), (99999, 0), (500, 500)]:
            stats = {
                'total_steps': 100, 'cumulative_wait_time': wait,
                'total_throughput': tp, 'emergency_events': 0,
                'emergency_events_cleared': 0, 'green_wave_steps': 0,
                'pedestrian_demand_steps': 0, 'pedestrian_phases_given': 0,
                'phase_oscillations': 0,
            }
            s = grade_episode(task, stats)
            assert 0.0 <= s <= 1.0, f"GRADER BUG: {task} returned {s}"
    print("  ✓ All grader boundary checks passed")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        bar = "█" * int(r['final_score'] * 30)
        print(f"  {r['task_id']:35s} [{bar:<30}] {r['final_score']:.4f}")

    print()
    if all_passed:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

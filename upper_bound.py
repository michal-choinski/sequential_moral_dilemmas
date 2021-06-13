import argparse
import math

import numpy as np


EXAMPLE_USAGE = """
Example Usage:
    python upper_bound --n_agents 4 --episode_len 1000 --n_apples 20
"""

def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Determine maximum number of gathered apples",
        epilog=EXAMPLE_USAGE)

    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--n_agents",
        type=int,
        required=True,
        help="Number of agents")
    required_named.add_argument(
        "--episode_len",
        type=int,
        required=True,
        help="Length of the episode")
    required_named.add_argument(
        "--n_apples",
        type=int,
        required=True,
        help="Initial number of spawned apples")
    required_named.add_argument(
        "--n_apple_frames",
        type=int,
        required=True,
        help="Number of frames after which apples are respawned")

    return parser


class UpperBoundSimulator:

    def __init__(self, n_apples, n_apple_frames, episode_length):
        self.apple_spawn_queue = np.zeros(n_apple_frames, dtype=np.uint8)
        self.max_n_apples_in_this_turn = n_apples
        self.n_apples = n_apples
        self.n_apples_init = n_apples
        self.episode_length = episode_length

    def n_apples_at_step(self, step_number):
        return round(
            ((math.cos(step_number * 2 * math.pi / self.episode_length) + 1) * 0.75 + 0.5) * self.n_apples_init / 2)

    def gather_and_enqueue_apples_to_respawn(self, step_number, apples_to_be_spawned):
        new_max_n_apples_in_this_turn = self.n_apples_at_step(step_number)
        if apples_to_be_spawned > 0 and self.max_n_apples_in_this_turn > new_max_n_apples_in_this_turn:
            # using min function to discount the remaining apples in the next steps
            difference = min(apples_to_be_spawned, self.max_n_apples_in_this_turn - new_max_n_apples_in_this_turn)
            self.max_n_apples_in_this_turn -= difference
            apples_to_be_spawned -= difference
        elif self.max_n_apples_in_this_turn < new_max_n_apples_in_this_turn:
            difference = new_max_n_apples_in_this_turn - self.max_n_apples_in_this_turn
            self.max_n_apples_in_this_turn += difference
            apples_to_be_spawned += difference

        self.queue_apples_to_be_spawned(apples_to_be_spawned)

    def spawn_apples(self):
        self.n_apples += self.apple_spawn_queue[0]
        self.apple_spawn_queue[0] = 0
        self.apple_spawn_queue = np.roll(self.apple_spawn_queue, -1)

    def queue_apples_to_be_spawned(self, apples: int):
        self.apple_spawn_queue[-1] += apples


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    u_simulator = UpperBoundSimulator(args.n_apples, args.n_apple_frames, args.episode_len)

    consumed_total = 0

    for t in range(1, args.episode_len + 1):
        consumed = min(args.n_agents, u_simulator.n_apples)
        u_simulator.n_apples -= consumed
        consumed_total += consumed
        u_simulator.gather_and_enqueue_apples_to_respawn(t, consumed)
        u_simulator.spawn_apples()

    print(consumed_total, "apples consumed")



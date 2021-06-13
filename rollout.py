#!/usr/bin/env python

import argparse
import collections
import copy
import gym
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import shelve
import shutil

import ray
import ray.cloudpickle as cloudpickle
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.tune.utils import merge_dicts
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR, register_env

from envs.smd_basic.gathering_env import GatheringEnv
import utility_funcs
from envs.smd_crp.gathering_crp_env import GatheringCRPEnv
from model.two_input_model import CRPCustomQModel
from envs.ssd.map_env import DEFAULT_COLOURS

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""

# Note: if you use any custom models or envs, register them here first, e.g.:
#
# from ray.rllib.examples.env.parametric_actions_cartpole import \
#     ParametricActionsCartPole
# from ray.rllib.examples.model.parametric_actions_model import \
#     ParametricActionsModel
# ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# register_env("pa_cartpole", lambda _: ParametricActionsCartPole(10))


class RolloutSaver:
    """Utility class for storing rollouts.

    Currently supports two behaviours: the original, which
    simply dumps everything to a pickle file once complete,
    and a mode which stores each rollout as an entry in a Python
    shelf db file. The latter mode is more robust to memory problems
    or crashes part-way through the rollout generation. Each rollout
    is stored with a key based on the episode number (0-indexed),
    and the number of episodes is stored with the key "num_episodes",
    so to load the shelf file, use something like:

    with shelve.open('rollouts.pkl') as rollouts:
       for episode_index in range(rollouts["num_episodes"]):
          rollout = rollouts[str(episode_index)]

    If outfile is None, this class does nothing.
    """

    def __init__(self,
                 outfile=None,
                 use_shelve=False,
                 write_update_file=False,
                 target_steps=None,
                 target_episodes=None,
                 save_info=False):
        self._outfile = outfile
        self._update_file = None
        self._use_shelve = use_shelve
        self._write_update_file = write_update_file
        self._shelf = None
        self._num_episodes = 0
        self._rollouts = []
        self._current_rollout = []
        self._total_steps = 0
        self._target_episodes = target_episodes
        self._target_steps = target_steps
        self._save_info = save_info

    def _get_tmp_progress_filename(self):
        outpath = Path(self._outfile)
        return outpath.parent / ("__progress_" + outpath.name)

    @property
    def outfile(self):
        return self._outfile

    def __enter__(self):
        if self._outfile:
            if self._use_shelve:
                # Open a shelf file to store each rollout as they come in
                self._shelf = shelve.open(self._outfile)
            else:
                # Original behaviour - keep all rollouts in memory and save
                # them all at the end.
                # But check we can actually write to the outfile before going
                # through the effort of generating the rollouts:
                try:
                    with open(self._outfile, "wb") as _:
                        pass
                except IOError as x:
                    print("Can not open {} for writing - cancelling rollouts.".
                          format(self._outfile))
                    raise x
            if self._write_update_file:
                # Open a file to track rollout progress:
                self._update_file = self._get_tmp_progress_filename().open(
                    mode="w")
        return self

    def __exit__(self, type, value, traceback):
        if self._shelf:
            # Close the shelf file, and store the number of episodes for ease
            self._shelf["num_episodes"] = self._num_episodes
            self._shelf.close()
        elif self._outfile and not self._use_shelve:
            # Dump everything as one big pickle:
            cloudpickle.dump(self._rollouts, open(self._outfile, "wb"))
        if self._update_file:
            # Remove the temp progress file:
            self._get_tmp_progress_filename().unlink()
            self._update_file = None

    def _get_progress(self):
        if self._target_episodes:
            return "{} / {} episodes completed".format(self._num_episodes,
                                                       self._target_episodes)
        elif self._target_steps:
            return "{} / {} steps completed".format(self._total_steps,
                                                    self._target_steps)
        else:
            return "{} episodes completed".format(self._num_episodes)

    def begin_rollout(self):
        self._current_rollout = []

    def end_rollout(self):
        if self._outfile:
            if self._use_shelve:
                # Save this episode as a new entry in the shelf database,
                # using the episode number as the key.
                self._shelf[str(self._num_episodes)] = self._current_rollout
            else:
                # Append this rollout to our list, to save laer.
                self._rollouts.append(self._current_rollout)
        self._num_episodes += 1
        if self._update_file:
            self._update_file.seek(0)
            self._update_file.write(self._get_progress() + "\n")
            self._update_file.flush()

    def append_step(self, obs, action, next_obs, reward, done, info):
        """Add a step to the current rollout, if we are saving them"""
        if self._outfile:
            if self._save_info:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done, info])
            else:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done])
        self._total_steps += 1


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Suppress rendering of the environment.")
    parser.add_argument(
        "--monitor",
        default=False,
        action="store_true",
        help="Wrap environment in gym Monitor to record video. NOTE: This "
        "option is deprecated: Use `--video-dir [some dir]` instead.")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Specifies the directory into which videos of all episode "
        "rollouts will be stored.")
    parser.add_argument(
        "--store-video",
        default=False,
        help="Generate and store video.")
    parser.add_argument(
        "--steps",
        default=10000,
        help="Number of timesteps to roll out (overwritten by --episodes).")
    parser.add_argument(
        "--episodes",
        default=0,
        help="Number of complete episodes to roll out (overrides --steps).")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Gets merged with loaded configuration from checkpoint file and "
        "`evaluation_config` settings therein.")
    parser.add_argument(
        "--save-info",
        default=False,
        action="store_true",
        help="Save the info field generated by the step() method, "
        "as well as the action, observations, rewards and done fields.")
    parser.add_argument(
        "--use-shelve",
        default=False,
        action="store_true",
        help="Save rollouts into a python shelf file (will save each episode "
        "as it is generated). An output filename must be set using --out.")
    parser.add_argument(
        "--track-progress",
        default=False,
        action="store_true",
        help="Write progress to a temporary file (updated "
        "after each episode). An output filename must be set using --out; "
        "the progress file will live in the same folder.")
    parser.add_argument(
        "--env_config",
        default="env_config.json",
        help="Environment specific configuration file")
    parser.add_argument(
        "--generate_figures",
        default=False,
        help="Generate figures")
    return parser


def run(args, parser):
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    # If no pkl file found, require command line `--config`.
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no config given on command line!")
        else:
            config = args.config

    # Load the config from pickled.
    else:
        with open(config_path, "rb") as f:
            config = cloudpickle.load(f)

    # Set num_workers to be at least 2.
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])

    # Make sure worker 0 has an Env.
    config["create_env_on_driver"] = True

    # Merge with `evaluation_config` (first try from command line, then from
    # pkl file).
    evaluation_config = copy.deepcopy(
        args.config.get("evaluation_config", config.get(
            "evaluation_config", {})))
    config = merge_dicts(config, evaluation_config)
    # Merge with command line `--config` settings (if not already the same
    # anyways).
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    env_config = {}
    with open(args.env_config) as json_file:
        env_config = json.load(json_file)

    ray.init()

    env_name = args.env

    #if env_name == "smd_crp_env":
    ModelCatalog.register_custom_model("CRPCustomQModel", CRPCustomQModel)

    def env_creator(_):
        if env_name == "smd_env":
            return GatheringEnv(env_config)
        elif env_name == "smd_crp_env":
            return GatheringCRPEnv(env_config)

    register_env(env_name, env_creator)

    # Create the Trainer from config.
    cls = get_trainable_cls(args.run)
    agent = cls(env=args.env, config=config)
    # Load state from checkpoint.
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    num_episodes = int(args.episodes)

    # Do the actual rollout.
    with RolloutSaver(
            args.out,
            args.use_shelve,
            write_update_file=args.track_progress,
            target_steps=num_steps,
            target_episodes=num_episodes,
            save_info=args.save_info) as saver:
        rollout(agent, args.env, num_steps, num_episodes, saver,
                args.no_render, args.store_video, args.generate_figures)
    agent.stop()


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # if num_episodes is set, this overrides num_steps
    if num_episodes:
        return episodes < num_episodes
    # if num_steps is set, continue until we reach the limit
    if num_steps:
        return steps < num_steps
    # otherwise keep going forever
    return True


def collect_data_for_figures(actions, rewards, env):
    step_info = {}
    tagging_actions = {}
    for agent_id in actions.keys():
        cnt = 1 if actions[agent_id] == 7 else 0
        tagging_actions[agent_id] = cnt

    step_info["tagging_actions"] = tagging_actions
    step_info["rewards"] = rewards

    agent_gatherings = {}
    for agent_id in env.agents.keys():
        agent_gatherings[agent_id] = env.agents[agent_id].gathered_this_turn

    step_info["agent_gatherings"] = agent_gatherings

    agent_starvations = {}
    for agent_id in env.agents.keys():
        agent_starvations[agent_id] = env.agents[agent_id].is_starving

    step_info["agent_starvations"] = agent_starvations

    agent_apples = {}
    for agent_id in env.agents.keys():
        agent_apples[agent_id] = env.agents[agent_id].n_apples

    step_info["agent_apples"] = agent_apples

    agent_removals = {}
    for agent_id in env.agents.keys():
        agent_removals[agent_id] = env.agents[agent_id].is_removed

    step_info["agent_removal"] = agent_removals

    if isinstance(env, GatheringCRPEnv):
        step_info["common_pool"] = env.common_pool

        agent_aid_receptions = {}
        for agent_id in env.agents.keys():
            agent_aid_receptions[agent_id] = env.agents[agent_id].received_aid_this_turn

        step_info["agent_aid_reception"] = agent_aid_receptions

        agent_contributions = {}
        for agent_id in env.agents.keys():
            agent_contributions[agent_id] = env.agents[agent_id].contributed_this_turn

        step_info["agent_contributions"] = agent_contributions

    collect_data_for_figures.steps.append(step_info)


collect_data_for_figures.steps = []


# terrible hack, but for some reason colors
# in the video are mixed up
# the following workaround works only for 4 agents
def agent_color(agent_id):
    id = str(int(agent_id[agent_id.index("-") + 1:]) + 1)
    #id = '5' if id == '4' else id
    if id == '1':
        id = '3'
    elif id == '2':
        id = '5'
    elif id == '3':
        id = '1'
    elif id == '4':
        id = '2'
    else:
        print("Adjust the function agent_color!")

    return DEFAULT_COLOURS[id]


def generate_no_intake(intakes, n_steps_to_starve):
    for agent_id in intakes.keys():
        starvation = []
        last_intake = -1
        for i in range(len(intakes[agent_id])):
            if intakes[agent_id][i] > 0:
                last_intake = i
            starvation.append((i - last_intake) / n_steps_to_starve)

        plt.plot(list(range(1, len(starvation) + 1)), starvation,
                 label="Agent {}".format(agent_id),
                 color=[a / 255 for a in list(agent_color(agent_id))])

    plt.axhline(y=1.0, color='tab:gray', linestyle=':')
    plt.locator_params(axis="both", integer=True, tight=True)

    plt.legend()
    plt.title("Agent without intake ({} steps without apples)".format(n_steps_to_starve))
    plt.savefig('figures/no_intake_{}.png'.format(n_steps_to_starve))
    plt.close()


def generate_starvation_moving_window(starvations, window_size, n_steps_to_starve):
    for agent_id in starvations.keys():
        starvation = []
        for i in range(window_size - 1, len(starvations[agent_id])):
            starvation_window_sum = 0
            for j in range(i - window_size + 1, i):
                starvation_window_sum += starvations[agent_id][j]

            starvation.append(starvation_window_sum / n_steps_to_starve)

        plt.plot(list(range(window_size - 1, len(starvations[agent_id]))), starvation,
                 label="Agent {}".format(agent_id),
                 color=[a / 255 for a in list(agent_color(agent_id))])

    plt.axhline(y=1.0, color='tab:gray', linestyle=':')
    plt.locator_params(axis="both", integer=True, tight=True)

    plt.legend()
    plt.title("Agent starvation ({}-steps window, {} steps to starve)".format(window_size, n_steps_to_starve))
    plt.savefig('figures/starvation_{}_{}.png'.format(window_size, n_steps_to_starve))
    plt.close()


def generate_and_return_figure_rewards(env):
    # rewards
    for agent_id in env.agents.keys():
        rewards_cummulative = []
        last_reward = 0
        for step_info in collect_data_for_figures.steps:
            last_reward += step_info["rewards"][agent_id]
            rewards_cummulative.append(last_reward)

        plt.plot(list(range(1, len(rewards_cummulative) + 1)),
                 rewards_cummulative,
                 label="Agent {}".format(agent_id),
                 color=[a / 255 for a in list(agent_color(agent_id))])

    plt.legend()
    plt.title("Reward per agent (cumulative)")

    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img


def generate_and_return_figure_gathering(env):
    # gathering
    for agent_id in env.agents.keys():
        apples = []
        last_gathering = 0
        for step_info in collect_data_for_figures.steps:
            last_gathering += int(step_info["agent_gatherings"][agent_id])
            apples.append(last_gathering)

        plt.plot(list(range(1, len(apples) + 1)), apples,
                 label="Agent {}".format(agent_id),
                 color=[a / 255 for a in list(agent_color(agent_id))])

    plt.locator_params(axis="both", integer=True, tight=True)

    plt.legend()
    plt.title("Apple gathering per agent (cumulative)")

    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img


def generate_and_return_figure_apples(env):
    # apples
    for agent_id in env.agents.keys():
        apples = []
        for step_info in collect_data_for_figures.steps:
            apples.append(step_info["agent_apples"][agent_id])

        plt.plot(list(range(1, len(apples) + 1)), apples,
                 label="Agent {}".format(agent_id),
                 color=[a / 255 for a in list(agent_color(agent_id))])

    if isinstance(env, GatheringCRPEnv):
        common_pool = []
        for step_info in collect_data_for_figures.steps:
            common_pool.append(step_info["common_pool"])

        plt.plot(list(range(1, len(collect_data_for_figures.steps) + 1)), common_pool,
                 label="Common pool", color="black")

    plt.locator_params(axis="both", integer=True, tight=True)

    plt.legend()
    plt.title("Apple stock per agent")

    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img


def generate_and_return_figure_starvation(env):
    # no intake
    starvations = {}
    for agent_id in env.agents.keys():
        s = []
        for step_info in collect_data_for_figures.steps:
            s.append(int(step_info["agent_starvations"][agent_id]))

        starvations[agent_id] = s

    window_size = 100

    for agent_id in starvations.keys():
        starvation = []
        for i in range(window_size - 1, len(starvations[agent_id])):
            starvation_window_sum = 0
            for j in range(i - window_size + 1, i):
                starvation_window_sum += starvations[agent_id][j]

            starvation.append(starvation_window_sum / window_size)

        plt.plot(list(range(window_size - 1, len(starvations[agent_id]))), starvation,
                 label="Agent {}".format(agent_id),
                 color=[a / 255 for a in list(agent_color(agent_id))])

    plt.locator_params(axis="both", integer=True, tight=True)

    plt.legend()
    plt.title("Agent starvation ({}-steps window)".format(window_size))

    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img


def generate_and_return_figure_aid(env):
    # aid
    if isinstance(env, GatheringCRPEnv):
        for agent_id in env.agents.keys():
            apples = []
            last_aid = 0
            for step_info in collect_data_for_figures.steps:
                last_aid += int(step_info["agent_aid_reception"][agent_id])
                apples.append(last_aid)

            plt.plot(list(range(1, len(apples) + 1)), apples,
                     label="Agent {}".format(agent_id),
                     color=[a / 255 for a in list(agent_color(agent_id))])

        plt.locator_params(axis="both", integer=True, tight=True)

    plt.legend()
    plt.title("Apple aid reception per agent (cumulative)")

    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img


def generate_and_return_figure_contribution(env):
    # contribution
    if isinstance(env, GatheringCRPEnv):
        for agent_id in env.agents.keys():
            apples = []
            last_contribution = 0
            for step_info in collect_data_for_figures.steps:
                last_contribution += int(step_info["agent_contributions"][agent_id])
                apples.append(last_contribution)

            plt.plot(list(range(1, len(apples) + 1)), apples,
                     label="Agent {}".format(agent_id),
                     color=[a / 255 for a in list(agent_color(agent_id))])

        plt.locator_params(axis="both", integer=True, tight=True)

    plt.legend()
    plt.title("Apple contribution per agent (cumulative)")

    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img


def generate_figures_func(env):
    # tagging
    for agent_id in env.agents.keys():
        rewards_cummulative = []
        last_reward = 0
        for step_info in collect_data_for_figures.steps:
            last_reward += step_info["tagging_actions"][agent_id]
            rewards_cummulative.append(last_reward)

        plt.plot(list(range(1, len(rewards_cummulative) + 1)),
                 rewards_cummulative,
                 label="Agent {}".format(agent_id),
                 color=[a / 255 for a in list(agent_color(agent_id))])

    plt.legend()
    plt.title("Tagging actions per agent (cumulative)")
    plt.savefig('figures/tagging.png')
    plt.close()

    # rewards
    for agent_id in env.agents.keys():
        rewards_cummulative = []
        last_reward = 0
        for step_info in collect_data_for_figures.steps:
            last_reward += step_info["rewards"][agent_id]
            rewards_cummulative.append(last_reward)

        plt.plot(list(range(1, len(rewards_cummulative) + 1)),
                 rewards_cummulative,
                 label="Agent {}".format(agent_id),
                 color=[a / 255 for a in list(agent_color(agent_id))])

    plt.legend()
    plt.title("Reward per agent (cumulative)")
    plt.savefig('figures/rewards.png')
    plt.close()

    # gathering
    for agent_id in env.agents.keys():
        apples = []
        last_gathering = 0
        for step_info in collect_data_for_figures.steps:
            last_gathering += int(step_info["agent_gatherings"][agent_id])
            apples.append(last_gathering)

        plt.plot(list(range(1, len(apples) + 1)), apples,
                 label="Agent {}".format(agent_id),
                 color=[a / 255 for a in list(agent_color(agent_id))])

    plt.locator_params(axis="both", integer=True, tight=True)

    plt.legend()
    plt.title("Apple gathering per agent (cumulative)")
    plt.savefig('figures/gathering.png')
    plt.close()

    # removal
    for agent_id in env.agents.keys():
        removals = []
        for step_info in collect_data_for_figures.steps:
            removal = int(step_info["agent_removal"][agent_id])
            removals.append(removal)

        plt.plot(list(range(1, len(removals) + 1)), removals,
                 label="Agent {}".format(agent_id),
                 color=[a / 255 for a in list(agent_color(agent_id))])

    plt.locator_params(axis="both", integer=True, tight=True)

    plt.legend()
    plt.title("Removal per agent")
    plt.savefig('figures/removal.png')
    plt.close()

    # apples
    for agent_id in env.agents.keys():
        apples = []
        for step_info in collect_data_for_figures.steps:
            apples.append(step_info["agent_apples"][agent_id])

        plt.plot(list(range(1, len(apples) + 1)), apples,
                 label="Agent {}".format(agent_id),
                 color=[a / 255 for a in list(agent_color(agent_id))])

    if isinstance(env, GatheringCRPEnv):
        common_pool = []
        for step_info in collect_data_for_figures.steps:
            common_pool.append(step_info["common_pool"])

        plt.plot(list(range(1, len(collect_data_for_figures.steps) + 1)), common_pool,
                 label="Common pool", color="black")

    plt.locator_params(axis="both", integer=True, tight=True)

    plt.legend()
    plt.title("Apple stock per agent")
    plt.savefig('figures/apples.png')
    plt.close()

    if isinstance(env, GatheringCRPEnv):
        # aid
        for agent_id in env.agents.keys():
            apples = []
            last_aid = 0
            for step_info in collect_data_for_figures.steps:
                last_aid += int(step_info["agent_aid_reception"][agent_id])
                apples.append(last_aid)

            plt.plot(list(range(1, len(apples) + 1)), apples,
                     label="Agent {}".format(agent_id),
                     color=[a / 255 for a in list(agent_color(agent_id))])

        plt.locator_params(axis="both", integer=True, tight=True)

        plt.legend()
        plt.title("Apple aid reception per agent (cumulative)")
        plt.savefig('figures/aid.png')
        plt.close()

        # contribution
        for agent_id in env.agents.keys():
            apples = []
            last_contribution = 0
            for step_info in collect_data_for_figures.steps:
                last_contribution += int(step_info["agent_contributions"][agent_id])
                apples.append(last_contribution)

            plt.plot(list(range(1, len(apples) + 1)), apples,
                     label="Agent {}".format(agent_id),
                     color=[a / 255 for a in list(agent_color(agent_id))])

        plt.locator_params(axis="both", integer=True, tight=True)

        plt.legend()
        plt.title("Apple contribution per agent (cumulative)")
        plt.savefig('figures/contribution.png')
        plt.close()

        # no intake
        agent_intakes = {}
        for agent_id in env.agents.keys():
            intakes = []
            for step_info in collect_data_for_figures.steps:
                intakes.append(int(step_info["agent_gatherings"][agent_id] or step_info["agent_aid_reception"][agent_id]))

            agent_intakes[agent_id] = intakes

        generate_no_intake(agent_intakes, 20)
        generate_no_intake(agent_intakes, 50)
        generate_no_intake(agent_intakes, 100)

    else:
        # no intake
        agent_intakes = {}
        for agent_id in env.agents.keys():
            intakes = []
            for step_info in collect_data_for_figures.steps:
                intakes.append(int(step_info["agent_gatherings"][agent_id]))

            agent_intakes[agent_id] = intakes

        generate_no_intake(agent_intakes, 20)
        generate_no_intake(agent_intakes, 50)
        generate_no_intake(agent_intakes, 100)

    # no intake
    agent_starvations = {}
    for agent_id in env.agents.keys():
        starvations = []
        for step_info in collect_data_for_figures.steps:
            starvations.append(int(step_info["agent_starvations"][agent_id]))

        agent_starvations[agent_id] = starvations

    generate_starvation_moving_window(agent_starvations, 10, 5)
    generate_starvation_moving_window(agent_starvations, 20, 10)
    generate_starvation_moving_window(agent_starvations, 50, 25)
    generate_starvation_moving_window(agent_starvations, 100, 50)


def rollout(agent,
            env_name,
            num_steps,
            num_episodes=0,
            saver=None,
            no_render=True,
            store_video=False,
            generate_figures=False):
    policy_agent_mapping = default_policy_agent_mapping

    if saver is None:
        saver = RolloutSaver()

    if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        from gym import envs
        if envs.registry.env_specs.get(agent.config["env"]):
            # if environment is gym environment, load from gym
            env = gym.make(agent.config["env"])
        else:
            # if environment registered ray environment, load from ray
            env_creator = _global_registry.get(ENV_CREATOR,
                                               agent.config["env"])
            env_context = EnvContext(
                agent.config["env_config"] or {}, worker_index=0)
            env = env_creator(env_context)
        multiagent = False
        try:
            policy_map = {DEFAULT_POLICY_ID: agent.policy}
        except AttributeError:
            raise AttributeError(
                "Agent ({}) does not have a `policy` property! This is needed "
                "for performing (trained) agent rollouts.".format(agent))
        use_lstm = {DEFAULT_POLICY_ID: False}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    full_obs = None
    full_fig = []
    if store_video:
        shape = env.base_map.shape
        full_obs = [np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
                    for i in range(num_steps)]

    steps = 0
    episodes = 0
    while keep_going(steps, num_steps, episodes, num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        saver.begin_rollout()
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and keep_going(steps, num_steps, episodes,
                                      num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)

            if generate_figures:
                collect_data_for_figures(action, reward, env)

            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(
                    r for r in reward.values() if r is not None)
            else:
                reward_total += reward
            if not no_render:
                env.render()
            saver.append_step(obs, action, next_obs, reward, done, info)

            if store_video:
                rgb_arr = env.map_to_colors()
                full_obs[steps] = rgb_arr.astype(np.uint8)
                figures = {}
                figures['rewards'] = generate_and_return_figure_rewards(env)
                figures['gathering'] = generate_and_return_figure_gathering(env)
                figures['apples'] = generate_and_return_figure_apples(env)
                figures['aid'] = generate_and_return_figure_aid(env)
                figures['contribution'] = generate_and_return_figure_contribution(env)
                figures['starvation'] = generate_and_return_figure_starvation(env)
                full_fig.append(figures)

            steps += 1
            obs = next_obs
        saver.end_rollout()
        print("Episode #{}: reward: {}".format(episodes, reward_total))
        if done:
            episodes += 1

    print("===========================================")
    print("================= SUMMARY =================")
    print("===========================================")
    num_agents = len(env.agents)
    spawned_per_agent_steps = round(env.total_n_apples_spawned / num_steps / num_agents, 2)
    print("Total number of spawned apples:", env.total_n_apples_spawned, "=", spawned_per_agent_steps, "per agent step")
    total_gathered = sum([
            sum([int(item[1]) for item in step["agent_gatherings"].items()])
            for step in collect_data_for_figures.steps
    ])
    gathered_per_agent_steps = round(total_gathered / num_steps / num_agents, 2)
    print("Total number of gathered apples:", total_gathered, "=", gathered_per_agent_steps, "per agent step")
    total_tagging = sum([
        sum([int(item[1]) for item in step["tagging_actions"].items()])
        for step in collect_data_for_figures.steps
    ])
    print("Total number of tagging actions:", total_tagging)
    total_starving = sum([
        sum([int(item[1]) for item in step["agent_starvations"].items()])
        for step in collect_data_for_figures.steps
    ])
    average_starving_per_agent = round(total_starving / (num_steps * num_agents) * 100, 2)
    print("Total number of starving steps:", total_starving, "=", average_starving_per_agent, "% per agent step")

    if isinstance(env, GatheringCRPEnv):
        total_contributions = sum([
            sum([int(item[1]) for item in step["agent_contributions"].items()])
            for step in collect_data_for_figures.steps
        ])
        average_contributions_per_agent = round(total_contributions / (num_steps * num_agents) * 100, 2)
        print("Total number of contributions:", "=", total_contributions, average_contributions_per_agent, "% per agent step")

        total_aid = sum([
            sum([int(item[1]) for item in step["agent_aid_reception"].items()])
            for step in collect_data_for_figures.steps
        ])
        average_aid_per_agent = round(total_aid / (num_steps * num_agents) * 100, 2)
        print("Total number of aid receptions:", "=", total_aid, average_aid_per_agent, "% per agent step")

    if generate_figures:
        generate_figures_func(env)

    if store_video:
        path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
        if not os.path.exists(path):
            os.makedirs(path)
        images_path = path + '/images/'
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        utility_funcs.make_video_from_rgb_imgs_and_figs(full_obs, full_fig, path)

        # Clean up images
        shutil.rmtree(images_path)

if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    # Old option: monitor, use video-dir instead.
    if args.monitor:
        deprecation_warning("--monitor", "--video-dir=[some dir]")
    # User tries to record videos, but no-render is set: Error.
    if (args.monitor or args.video_dir) and args.no_render:
        raise ValueError(
            "You have --no-render set, but are trying to record rollout videos"
            " (via options --video-dir/--monitor)! "
            "Either unset --no-render or do not use --video-dir/--monitor.")
    # --use_shelve w/o --out option.
    if args.use_shelve and not args.out:
        raise ValueError(
            "If you set --use-shelve, you must provide an output file via "
            "--out as well!")
    # --track-progress w/o --out option.
    if args.track_progress and not args.out:
        raise ValueError(
            "If you set --track-progress, you must provide an output file via "
            "--out as well!")

    run(args, parser)
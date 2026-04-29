# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


INTERNAL_DEFAULT_KEYS = {
    'num_env_steps',
    'num_processes',
    'num_steps',
    'seed',
    'ego_ctrlsim_kl_safe_update',
    'use_editor',
    'use_ego_ctrlsim_kl_loss',
    'use_policy_reweighting',
    'use_plr',
    'tilting_mode',
}


def load_internal_defaults():
    """Load only the argument defaults needed by make_cmd internals."""
    from arguments import parser

    defaults = {}
    for action in parser._actions:
        if action.dest in INTERNAL_DEFAULT_KEYS:
            default = action.default
            if action.type is int and default is not None:
                default = int(default)

            defaults[action.dest] = default

    return defaults


def format_cmd_args(key, value):
    if key.startswith('__'):
        return []

    return [f'--{key}={value}']


def generate_train_cmds(
    params, num_trials=1, start_index=0, newlines=False, 
    xpid_generator=None, xpid_prefix='', xvfb=False, 
    count_set=None):
    separator = ' \\\n' if newlines else ' '
    
    cmds = []

    if xpid_generator:
        params['xpid'] = xpid_generator(params, xpid_prefix)
    xpid_timestamp = time.strftime('%m%d%H%M') if xpid_generator else None

    start_seed = params['seed']

    for t in range(num_trials):
        params['seed'] = start_seed + t + start_index

        if xvfb:
            cmd = [f'xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR -noreset" -- python -m train']
        else:
            cmd = [f'python -m train']

        trial_idx = t + start_index
        for k,v in params.items():
            if k == 'xpid':
                v = f'{v}_{trial_idx}'
                if xpid_timestamp is not None:
                    v = f'{v}-{xpid_timestamp}'

                if count_set is not None:
                    count_set.add(v)

            cmd.extend(format_cmd_args(k, v))

        cmd = separator.join(cmd)

        cmds.append(cmd)

    return cmds


def generate_all_params_for_grid(grid, defaults={}):
    
    def update_params_with_choices(prev_params, param, choices):
        updated_params = []
        for v in choices:
            for p in prev_params:
                updated = p.copy()
                updated[param] = v
                updated_params.append(updated)

        return updated_params

    all_params = [{}]
    for param, choices in grid.items():
        all_params = update_params_with_choices(all_params, param, choices)

    full_params = []
    for p in all_params:
        d = defaults.copy()
        d.update(p)
        full_params.append(d)

    return full_params


def parse_args():
    parser = argparse.ArgumentParser(description='Make commands')
    
    parser.add_argument(
        '--dir',
        type=str,
        default='train_scripts/grid_configs/',
        help='Path to directory with .json configs')

    parser.add_argument(
        '--json',
        type=str,
        default=None,
        help='Name of .json config for hyperparameter search-grid')

    parser.add_argument(
        '--num_trials',
        type=int,
        default=1,
        help='Name of .json config for hyperparameter search-grid')

    parser.add_argument(
        '--start_index',
        default=0,
        type=int,
        help='Starting trial index of xpid runs')

    parser.add_argument(
        '--count',
        action='store_true',
        help='Print number of generated commands at the end of output.')


    parser.add_argument(
        "--checkpoint",
        action='store_true',
        help='Whether to start from checkpoint'
    )

    parser.add_argument(
        '--use_ucb',
        action="store_true",
        help='Whether to include ucb arguments.')

    parser.add_argument(
        '--xvfb',
        action="store_true",
        help='Whether to use xvfb.')

    return parser.parse_args()


def xpid_from_params(p, prefix=''):
    prefix_str = '' if prefix == '' else f'{prefix}-'
    return (
        f'{prefix_str}'
        f'steps{p["num_env_steps"]}'
        f'-proc{p["num_processes"]}'
        f'-roll{p["num_steps"]}'
        f'-plr{int(bool(p["use_plr"]))}'
        f'-edit{int(bool(p["use_editor"]))}'
        f'-tilt{p["tilting_mode"]}'
        f'-kl{int(bool(p["use_ego_ctrlsim_kl_loss"]))}'
        f'-prw{int(bool(p["use_policy_reweighting"]))}'
    )

if __name__ == '__main__':
    args = parse_args()

    params = load_internal_defaults()

    json_filename = args.json
    if not json_filename.endswith('.json'):
        json_filename += '.json'

    grid_path = os.path.join(os.path.expandvars(os.path.expanduser(args.dir)), json_filename)
    config = json.load(open(grid_path))
    grid = config['grid']
    xpid_prefix = '' if 'xpid_prefix' not in config else config['xpid_prefix']

    if args.checkpoint:
        params['checkpoint'] = True

    # Generate all parameter combinations within grid, using defaults for fixed params
    all_params = generate_all_params_for_grid(grid, defaults=params)

    unique_xpids = None
    if args.count:
        unique_xpids = set()

    # Print all commands
    count = 0
    for p in all_params:
        cmds = generate_train_cmds(p,
            num_trials=args.num_trials, 
            start_index=args.start_index, 
            newlines=True, 
            xpid_generator=xpid_from_params, 
            xpid_prefix=xpid_prefix,
            xvfb=args.xvfb,
            count_set=unique_xpids)

        for c in cmds:
            print(c + '\n')
            count += 1

    if args.count:
        print(f'Generated {len(unique_xpids)} unique commands.')

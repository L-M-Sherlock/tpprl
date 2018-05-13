#!/usr/bin/env python

import pandas as pd
import click
import os

INPUT_FILE = "/NL/redqueen/work/rl-broadcast/users-1k-HR-5-followers-pruned-200-own-posts-trimmed-2.dill"
# OUTPUT_DIR = "/NL/crowdjudged/work/rl-broadcast/r_2/"
# OUTPUT_DIR = "/NL/crowdjudged/work/rl-broadcast/r_2-sim-opt-fix/"
OUTPUT_DIR = "/NL/crowdjudged/work/rl-broadcast/top_k-sim-opt-fix/"

@click.command()
@click.argument('in_csv')
@click.option('--dry/--no-dry', help='Dry run.', default=True)
@click.option('--epochs', help='Epochs.', default=25)
@click.option('--reward', 'reward_kind', help='Which reward to use [r_2_reward, top_k_reward].', default='r_2_reward')
@click.option('--output-dir', 'output_dir', help='Where to save the output', default=OUTPUT_DIR)
@click.option('--K', 'k', help='K in top-k loss.', default=1)
@click.option('--mem', 'mem', help='How much memory will each job need (MB).', default=10000)
@click.option('--until', 'until', help='Until which step to run the experiments.', default=1000)
@click.option('--save-every', 'save_every', help='How many epochs to save output at.', default=5)
@click.option('--q', 'q', help='Which q value to use. Negative values imply using the value in the CSV file.', default=-1.0)
def run(in_csv, dry, epochs, k, mem, reward_kind, output_dir, until, q, save_every):
    """Read parameters from in_csv, ignore the host/gpu information, and execute them on using sbatch."""
    os.makedirs(os.path.join(output_dir, 'stdout'), exist_ok=True)
    df = pd.read_csv(in_csv)

    for row_idx, row in df.iterrows():
        stdout_file = f'{output_dir}/stdout/user_idx-{row.idx}.%j.out'

        if q < 0:
            q = row.q

        if reward_kind == 'top_k_reward':
            cmd = (f'sbatch -c 2 --mem={mem} -o "{stdout_file}" ' +
                   f'./top_k_job.sh {row.inp_file} {row.idx} "{output_dir}" ' +
                   f'{row.N} {q} {until} {epochs} {k} {save_every}')
        elif reward_kind == 'r_2_reward':
            cmd = (f'sbatch -c 2 --mem={mem} -o "{stdout_file}" ' +
                   f'./r_2_job.sh {row.inp_file} {row.idx} "{output_dir}" ' +
                   f'{row.N} {q} {until} {epochs} {save_every}')
        else:
            raise ValueError("Unknown reward: {}".format(reward_kind))

        if dry:
            print(cmd)
        else:
            os.system(cmd)


if __name__ == '__main__':
    run()
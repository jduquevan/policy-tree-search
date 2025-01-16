import os
import random

def gen_command(config):
    command = "sbatch sweep/run_job_v1.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'learning_rate': [1e-5, 5e-5, 1e-4],
        'tree_lr': [ 1e-5, 5e-5, 1e-4],
        'br_entropy_beta': [0.001, 0.005, 0.01, 0.05],
        'agent_entropy_beta': [0.001, 0.005, 0.01, 0.05],
        'max_tree_depth': [4, 6, 8],
        'num_br_updates': [1, 2, 3],
        'features': [32, 48, 64],
        'do_self_play': [True, False]
    }

    # sample a random config
    config = {}
    for key, values in hparams.items():
        config[key] = random.choice(values)

    # submit this job using slurm
    command = gen_command(config)
    if fake_submit:
        print('fake submit')
    else:
        os.system(command)
    print(command)

def main(num_jobs: int, fake_submit: bool = True):
    for i in range(num_jobs):
        run_random_job(fake_submit=fake_submit)

if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()
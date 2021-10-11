import argparse
import re
import shutil
import subprocess
import time

import attrdict
import yaml

def get_parser():
    '''
    Returns base parser for scripts
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', help='Path to the benchmark trace', required=True)
    parser.add_argument('--model-path', help='Path to save model checkpoint. If not provided, the model checkpointing is skipped.')
    parser.add_argument('--debug', action='store_true', default=False, help='Faster epochs for debugging')
    parser.add_argument('--config', default='./configs/base.yaml', help='Path to configuration file for the model')
    parser.add_argument('--print-every', type=int, default=None, help='Print updates every this number of steps. Make sure to set when outputting to a file')
    parser.add_argument('--auto-resume', action='store_true', default=False, help='Automatically resume if checkpoint detected')
    parser.add_argument('--checkpoint-every', type=int, default=None, help='Save a resume checkpoint every this number of steps')
    parser.add_argument('--tb-dir', help='Directory to save TensorBoard logs')

    return parser


def load_config(config_path, debug=False):
    '''
    Loads config file and applies any necessary modifications to it
    '''
    # Parse config file
    with open(config_path, 'r') as f:
        config = attrdict.AttrDict(yaml.safe_load(f))

    # If the debug flag was raised, reduce the number of steps to have faster epochs
    if debug:
        config.steps_per_epoch = 2000

    return config


# Used to decorate functions for timing purposes
def timefunction(text=''):
    # Need to do a double decorate since we want the text parameter
    def decorate(f):
        def g(*args, **kwargs):
            start = time.time()
            print(text + '...', end='')
            ret = f(*args, **kwargs)
            end = time.time()
            print('Done in', end - start, 'seconds')
            return ret
        return g

    # Returns the decorating function with the text parameter available via closure
    return decorate


# Create the prefetch file from the given instruction IDs and addresses
# See more at github.com/Quangmire/ChampSim
def create_prefetch_file(prefetch_file, inst_ids, addresses, append=False):
    with open(prefetch_file, 'a' if append else 'w') as f:
        for inst_id, addr in zip(inst_ids, addresses):
            print(inst_id, hex(addr), file=f)


# Modified from https://stackoverflow.com/questions/41634674/tensorflow-on-shared-gpus-how-to-automatically-select-the-one-that-is-unused
def run_command(cmd):
    '''
    Run command, return output as string.
    '''
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode('ascii')


def list_available_gpus():
    '''
    Returns list of available GPU ids.
    '''
    output = run_command('nvidia-smi -L')

    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r'GPU (?P<gpu_id>\d+):')
    result = []
    for line in output.strip().split('\n'):
        m = gpu_regex.match(line)
        assert m, 'Couldnt parse '+line
        result.append(int(m.group('gpu_id')))

    return result


def gpu_memory_map():
    '''
    Returns map of GPU id to memory allocated on that GPU.
    '''
    output = run_command('nvidia-smi')
    gpu_output = output[output.find('GPU Memory'):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r'[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB')
    rows = gpu_output.split('\n')
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}

    for row in gpu_output.split('\n'):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group('gpu_id'))
        gpu_memory = int(m.group('gpu_memory'))
        result[gpu_id] += gpu_memory

    return result


def pick_gpu_lowest_memory():
    '''
    Returns GPU with the least allocated memory
    '''
    if shutil.which('nvidia-smi') is None:
        return ''

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]

    return best_gpu

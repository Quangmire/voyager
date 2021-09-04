import argparse
import os


def submit_jobs(path, start, n, exp_name):
    with open(path, 'r') as f:
        for line in list(f.readlines())[start:start + n]:
            # Default basename is just the filename without '.condor'
            batch_name = os.path.basename(line.strip())[:-len('.condor')]

            # The double underscore is intentional for easy string splitting
            if exp_name is not None:
                batch_name = exp_name + '__' + batch_name
            os.system('condor_submit ' + line.strip() + ' -batch-name ' + batch_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job_file', help='File containing condor submit files, one per  line')
    parser.add_argument('start', type=int, help='Line with first job to submit')
    parser.add_argument('n', type=int, help='Number of jobs / lines to submit')
    parser.add_argument('--exp-name', default=None, help='Name of experiment to differentiate condor files')
    args = parser.parse_args()

    submit_jobs(args.job_file, args.start, args.n, args.exp_name)


if __name__ == '__main__':
    main()

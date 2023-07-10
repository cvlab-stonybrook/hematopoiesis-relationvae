from utility_tools import record_exp_info, utils
import argparse
import sys
import os
import time
import subprocess
import shutil

import json


def generate_resuming_command(sys_argv, info):
    argv = sys_argv.copy()
    
    
    for idx, argv_argument in enumerate(argv):
        if argv_argument == '--result_dir':
            argv.pop(idx)       
            argv.pop(idx)       
            break
    
    passed_comd = ' '.join(argv)
    new_cmd = 'python {} --outdir_path ..'.format(passed_comd)
    return new_cmd

def main():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')  
    parser.add_argument('--dataset', type=str, default='MNIST_FS_raw', help='dataset name')
    parser.add_argument('--result_dir', type=str, default='../results', help='result directory')
    parser.add_argument('--log_path', action='store_true', default=False, help='saving a log file or not')
    parser.add_argument('--setting_file', type=str, default='', help='setting filename')
    parser.add_argument('--tag', type=str, default='', help='information tag')
    parser.add_argument('--n_episodes', type=int, default=200, help='total number of episodes')
    parser.add_argument('--n_episodes_parallel', type=int, default=10, help='number of episodes running parallelly, -1 for unlimited')
    parser.add_argument('--episode_mem_limit', type=int, default=2000, help='Minimum GPU needed (in GB) to run an episode')
    parser.add_argument('--listgpu', type=str, default='', help='string for gpu list')
    parser.add_argument('--outdir_path', type=str, default='', help='the master working directory, if set to non-empty, the prev exp will be continued')
    args, unknown_args = parser.parse_known_args()


    setting_file_path = 'settings/{}/{}/{}'.format(os.path.basename(__file__).split('.py')[0], args.dataset, args.setting_file)
    yml_setting = utils.load_yaml(setting_file_path)


    if args.outdir_path == '':
        parent_base_path = os.path.join(args.result_dir, args.dataset, os.path.basename(__file__).split('.py')[0])
        if not os.path.isdir(parent_base_path):
            os.makedirs(parent_base_path)
        outdir_path = os.path.join(parent_base_path,
                                '{:03}_{}'.format(len(next(os.walk(parent_base_path))[1]),
                                                                time.strftime("%m%d%H%M%S")))
        if not os.path.isdir(outdir_path):
            os.mkdir(outdir_path)


        record_exp_info.copy_src('.', os.path.join(outdir_path, 'source_code'))


        local_data_dir = os.path.join(outdir_path, 'data')
        os.makedirs(local_data_dir)
        shutil.copytree('../data/{}'.format(args.dataset), '{}/{}'.format(local_data_dir, args.dataset))


        with open(os.path.join(outdir_path, 'cmd.sh'), 'w') as fid:
            fid.write('#!/bin/bash\n')
            fid.write(generate_resuming_command(sys.argv, info={'outdir_path': outdir_path}))
    else:
        outdir_path = args.outdir_path

    master_result_file = os.path.join(outdir_path, 'master_result.yml')
    episode_dir = os.path.join(outdir_path, 'episodes')


    orig_stdout = sys.stdout
    logfile_path = os.path.join(outdir_path, 'log.txt')
    if args.log_path == True:
        sys.stdout = record_exp_info.Logger(logfile_path, _write_to_stdout=True)
    else:
        sys.stdout = record_exp_info.Logger(logfile_path, _write_to_stdout=False)


    print(' '.join(sys.argv))
    print(args.setting_file)
    print("Setting\n{}".format(json.dumps(yml_setting, indent=4)))


    gpulist = list(map(int, args.listgpu.split(',')))
    poll_list = []
    for i_episode in range(args.n_episodes):
        while True:
            for poll in poll_list:
                poll_status = poll.poll()
                if poll_status is not None:
                    poll_list.remove(poll)

            if len(poll_list) >= args.n_episodes_parallel:
                time.sleep(30)  
            else:
                break

        time.sleep(10)  
        selected_gpu = utils.get_most_empty_gpu(min_accepted_mem=args.episode_mem_limit, gpulist=gpulist)
        while selected_gpu < 0:
            time.sleep(30)  
            print('.', end='')
            selected_gpu = utils.get_most_empty_gpu(min_accepted_mem=args.episode_mem_limit)

        transfered_argv = unknown_args
        passed_episode_args = ['python', 'class_relation_vae_rtae_crossval_episode.py',
                                '--master_result_file', master_result_file,
                                '--episode_dir', episode_dir,
                                '--setting_file_path', setting_file_path,
                                '--dataset', args.dataset,
                                '--log_path',
                                '--gpu', str(selected_gpu),
                                '--tag', args.tag]
        passed_episode_args.extend(transfered_argv)
        print('Episode: {}'.format(i_episode))
        print(' '.join(passed_episode_args))
        pprocess = subprocess.Popen(passed_episode_args)
        poll_list.append(pprocess)


    sys.stdout.close()
    sys.stdout = orig_stdout

if __name__ == "__main__":
    main()

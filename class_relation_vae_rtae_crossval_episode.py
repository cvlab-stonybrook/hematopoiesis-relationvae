from __future__ import print_function
import argparse

from utility_tools.init_dataset_v2 import init_dataloaders_v3_episode
from models.relation_vae_rtae_model import *
import copy

import numpy as np
import json
import pickle
from utility_tools import record_exp_info, utils

import os
import time
import subprocess
import datetime


def convert_rtae_data(model, data_loader, filename, settings, iepoch, write_to_file):
    generated_list = {}

    for data_batch, label_batch in data_loader:
        data_batch = data_batch.to(settings['device'])

        rtae_data = model.convert_rtae(org_data_batch=data_batch)
        rtae_data = rtae_data.detach().cpu()
        
        

        for rtae_sample_idx in range(rtae_data.shape[0]):
            sample_label = label_batch[rtae_sample_idx].item()
            if sample_label not in generated_list:
                generated_list[sample_label] = []
            rtae_sample = [rtae_data[rtae_sample_idx], sample_label]
            generated_list[sample_label].append(rtae_sample)

    gen_data_file = None

    if write_to_file:
        gen_data_dir = os.path.join(settings['aug_data_dir'], 'e_{:04}'.format(iepoch))
        if not os.path.isdir(gen_data_dir):
            os.makedirs(gen_data_dir)
        gen_data_file = '{}/{}'.format(gen_data_dir, filename)
        pickle.dump(generated_list, open(gen_data_file, "wb"))

    return gen_data_file, generated_list, {}


def generate_data(model, test_loader, filename, settings, iepoch, write_to_file):
    generated_list = {}

    for i, (data_test_batch_parent, data_test_batch_child, label_test_batch_parent, label_test_batch_child) in enumerate(test_loader):


        data_test_batch1 = data_test_batch_child.to(settings['device'])
        data_test_batch2 = data_test_batch_parent.to(settings['device'])
        label_test_batch1 = label_test_batch_child

        gen_data = model.generate_samples_from_fewshot(fewshot_data_batch=data_test_batch1,
                                                        neighbor_data_batch=data_test_batch2,
                                                        n_generated=settings['n_generated'])
        novel_class_label = label_test_batch1[0][0].item()
        generated_list[novel_class_label] = gen_data.detach().cpu().numpy()

    gen_data_file = None

    if write_to_file:
        gen_data_dir = os.path.join(settings['aug_data_dir'], 'e_{:04}'.format(iepoch))
        if not os.path.isdir(gen_data_dir):
            os.makedirs(gen_data_dir)
        
        gen_data_file = '{}/{}'.format(gen_data_dir, filename)
        pickle.dump(generated_list, open(gen_data_file, "wb"))

    return gen_data_file, generated_list, {'DBIndexInversed': 1/(utils.DBindex(generated_list))}

def process_data_dim(data_batch, label_batch, input_dim):
    if len(data_batch.shape) > len(input_dim) + 1:
        data_batch = data_batch.view(data_batch.shape[0] * data_batch.shape[1], -1)
        if len(label_batch.shape) > 1:
            label_batch = label_batch.view(label_batch.shape[0] * label_batch.shape[1], -1)
        else:
            label_batch = label_batch.view(-1,1)
    return data_batch, label_batch




def train(train_loader, test_loader, settings):
    model_settings = {'device': settings['device'],
                      'style_dim': settings['style_dim'],
                      'content_dim': settings['content_dim'],
                      'rtae_dim': settings['rtae_dim'],
                      'input_dim': train_loader.input_dim,
                      'encoder_samplepair_name': settings['encoder_samplepair_name'],
                      'encoder_samplecontent_name': settings['encoder_samplecontent_name'],
                      'decoder_name': settings['decoder_name'],
                      'RTAE_encoder_name': settings['RTAE_encoder_name'],
                      'RTAE_decoder_name': settings['RTAE_decoder_name'],
                      'loss_weight': settings['loss_weight'],
                      'training_settings': settings['training_settings']
                      }
    model = Relation_VAE_RTAE(model_settings).to(settings['device'])
    model.initialize()

    for n, p in model.named_parameters():
        print(n, p.shape)

    max_DB_inversed = 1000000
    best_model_wts = None
    best_epoch = -1
    for iepoch in range(settings['vae_epoch']):
        n_trained_samples = 0.0
        lossinfo = {'loss': 0, 'bce': 0, 'klstyle': 0, 'klcontent1': 0, 'klcontent2': 0, 'inter_class_logvar_loss': 0, 'rtae_mse': 0}
        for batch_idx, (data_batch_parent, data_batch_child, label_batch_parent, label_batch_child) in enumerate(train_loader):

            data_batch1 = data_batch_child.to(settings['device'])
            data_batch2 = data_batch_parent.to(settings['device'])
            label_batch1 = label_batch_child
            label_batch2 = label_batch_parent
            
            train_info = model.train_model(data_batch1, data_batch2, label_batch1, label_batch2, train_settings={'current_epoch': iepoch})
            for lossname in lossinfo.keys():
                lossinfo[lossname] += train_info[lossname]
            n_trained_samples += train_info['n_samples']

        wrt_str = ''
        for lossname in lossinfo.keys():
            lossinfo[lossname] = lossinfo[lossname] / n_trained_samples
            wrt_str += ' - {}: {:05}'.format(lossname, lossinfo[lossname])
        print('====> Epoch: {},  {}'.format(iepoch, wrt_str))


        if (iepoch % settings['test_every_epoch'] == 0) and (settings['use_DBIndex']):
            if max_DB_inversed > lossinfo['rtae_mse']:
                max_DB_inversed = lossinfo['rtae_mse']
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = iepoch
        elif (iepoch % settings['test_every_epoch'] == 0) and (iepoch >= settings['vae_epoch_start_eval']):
            gen_data_file, generated_list, gen_data_info = generate_data(model=model, test_loader=test_loader, 
                                                          filename='novel_gen.pkl', settings=settings, iepoch=iepoch, write_to_file=(not settings['use_DBIndex']))
            if settings['generate_base_samples'] == True:
                gen_base_file, _, _ = generate_data(model=model, test_loader=settings['data_info']['base_test_loader'], 
                                        filename='base_gen.pkl', settings=settings, iepoch=iepoch, write_to_file=(not settings['use_DBIndex']))

            if settings['convert_rtae_base_samples'] == True:
                rtae_base_file, _, _ = convert_rtae_data(model=model, data_loader=settings['data_info']['base_single_sample_dataloader'], 
                                        filename='base_rtae.pkl', settings=settings, iepoch=iepoch, write_to_file=(not settings['use_DBIndex']))
                rtae_novelAll_file, _, _ = convert_rtae_data(model=model, data_loader=settings['data_info']['novel_all_loader'], 
                                    filename='novel_all_rtae.pkl', settings=settings, iepoch=iepoch, write_to_file=(not settings['use_DBIndex']))

            novel_query_data_file, novel_query_generated_list, novel_query_data_info = convert_rtae_data(model=model, data_loader=settings['data_info']['novel_query_loader'], 
                                                          filename='novel_query_rtae.pkl', settings=settings, iepoch=iepoch, write_to_file=(not settings['use_DBIndex']))

            novel_supp_data_file, novel_supp_generated_list, novel_supp_gen_data_info = convert_rtae_data(model=model, data_loader=settings['data_info']['novel_support_loader'], 
                                                          filename='novel_supp_rtae.pkl', settings=settings, iepoch=iepoch, write_to_file=(not settings['use_DBIndex']))


            passed_args = ['python', '-m', 'routines.classifier',
                        '-gpu', str(settings['gpu']),
                        '-datapath', novel_supp_data_file,
                        '-genpath', gen_data_file,
                        '-valpath', novel_query_data_file,
                        '-inputdim', str(settings['rtae_dim']),
                        '-outputdim', str(len(generated_list.keys())),
                        '-epoch', str(settings['classifier_settings']['epoch']),
                        '-batch_size', str(settings['classifier_settings']['batch_size']),
                        '-classifier_name', settings['classifier_settings']['classifier_name'],
                        '-exp_dir', settings['exp_dir'],
                        '-prefix_info', 'Epoch: {},  {}'.format(iepoch, settings['exp_dir']),
                        '-episode_master_result_file', settings['master_result_file']
                        ]
            utils.inject_args_start_thread(passed_args=passed_args, key_arr=['learning_rate', 'epoch_start_eval'], source_dict=settings['classifier_settings'])

            print('Time: ', datetime.datetime.now())
            print(' '.join(passed_args))
            subprocess.call(passed_args)


            if settings['keep_gen_files'] == False:
                os.remove(gen_data_file)
                os.remove(novel_supp_data_file)
                os.remove(novel_query_data_file)
                if settings['generate_base_samples'] == True:
                    os.remove(gen_base_file)
                if settings['convert_rtae_base_samples'] == True:
                    os.remove(rtae_base_file)
                    os.remove(rtae_novelAll_file)


    if settings['use_DBIndex']:
        print('{} - Best epoch by DBIndex: {}'.format(settings['exp_dir'], best_epoch))
        model.load_state_dict(best_model_wts)
        gen_data_file, generated_list, gen_data_info = generate_data(model=model, test_loader=test_loader, 
                                                          filename='novel_gen.pkl', settings=settings, iepoch=best_epoch, write_to_file=True)
        if settings['generate_base_samples'] == True:
            gen_base_file, _, _ = generate_data(model=model, test_loader=settings['data_info']['base_test_loader'], 
                                    filename='base_gen.pkl', settings=settings, iepoch=best_epoch, write_to_file=True)

        
        if settings['convert_rtae_base_samples'] == True:
            rtae_base_file, _, _ = convert_rtae_data(model=model, data_loader=settings['data_info']['base_single_sample_dataloader'], 
                                    filename='base_rtae.pkl', settings=settings, iepoch=iepoch, write_to_file=True)
            rtae_novelAll_file, _, _ = convert_rtae_data(model=model, data_loader=settings['data_info']['novel_all_loader'], 
                                    filename='novel_all_rtae.pkl', settings=settings, iepoch=iepoch, write_to_file=True)
        
        novel_query_data_file, novel_query_generated_list, novel_query_data_info = convert_rtae_data(model=model, data_loader=settings['data_info']['novel_query_loader'], 
                                                        filename='novel_query_rtae.pkl', settings=settings, iepoch=best_epoch, write_to_file=True)
        
        novel_supp_data_file, novel_supp_generated_list, novel_supp_gen_data_info = convert_rtae_data(model=model, data_loader=settings['data_info']['novel_support_loader'], 
                                                        filename='novel_supp_rtae.pkl', settings=settings, iepoch=best_epoch, write_to_file=True)

        
        passed_args = ['python', '-m', 'routines.classifier',
                    '-gpu', str(settings['gpu']),
                    '-datapath', novel_supp_data_file,
                    '-genpath', gen_data_file,
                    '-valpath', novel_query_data_file,
                    '-inputdim', str(settings['rtae_dim']),
                    '-outputdim', str(len(generated_list.keys())),
                    '-epoch', str(settings['classifier_settings']['epoch']),
                    '-batch_size', str(settings['classifier_settings']['batch_size']),
                    '-classifier_name', settings['classifier_settings']['classifier_name'],
                    '-exp_dir', settings['exp_dir'],
                    '-prefix_info', 'Epoch: {},  {}'.format(iepoch, settings['exp_dir']),
                    '-episode_master_result_file', settings['master_result_file']
                    ]
        utils.inject_args_start_thread(passed_args=passed_args, key_arr=['learning_rate', 'epoch_start_eval'], source_dict=settings['classifier_settings'])

        print('Time: ', datetime.datetime.now())
        print(' '.join(passed_args))

        subprocess.call(passed_args)


        if settings['keep_gen_files'] == False:
            os.remove(gen_data_file)
            os.remove(novel_supp_data_file)
            os.remove(novel_query_data_file)
            if settings['generate_base_samples'] == True:
                os.remove(gen_base_file)
            if settings['convert_rtae_base_samples'] == True:
                os.remove(rtae_base_file)
                os.remove(rtae_novelAll_file)

def main():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--vae_epoch', type=int, default=20, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--vae_epoch_start_eval', type=int, default=0, metavar='N', help='first epoch to eval')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--test_every_epoch', type=int, default=1, help='do testing every n epochs')
    parser.add_argument('--style_dim', type=int, default=10, help='dimension of style')
    parser.add_argument('--content_dim', type=int, default=12, help='dimension of content')
    parser.add_argument('--rtae_dim', type=int, default=12, help='dimension of rtae embeddings')
    parser.add_argument('--n_generated', type=int, default=10, help='number of generated samples per class')
    parser.add_argument('--dataset', type=str, default='MNIST_FS_raw', help='dataset name')
    parser.add_argument('--gpu', type=int, default=2, help='gpu id')
    parser.add_argument('--encoder_samplecontent', type=str, default='EncoderSC_Syn_1', help='encoder_samplecontent name')
    parser.add_argument('--encoder_samplepair', type=str, default='EncoderSP_Syn_1', help='encoder_samplepair name')
    parser.add_argument('--decoder', type=str, default='Decoder_MNIST_2', help='decoder name')
    parser.add_argument('--RTAE_encoder_name', type=str, default='Decoder_MNIST_2', help='decoder name')
    parser.add_argument('--RTAE_decoder_name', type=str, default='Decoder_MNIST_2', help='decoder name')
    parser.add_argument('--log_path', action='store_true', default=False, help='saving a log file or not')
    parser.add_argument('--setting_file_path', type=str, default='', help='setting filename')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--generate_base_samples', action='store_true', default=False, help='generating samples for base classes or not')
    parser.add_argument('--convert_rtae_base_samples', action='store_true', default=False, help='converting samples for base classes or not')
    parser.add_argument('--use_DBIndex', action='store_true', default=False, help='choose model to evaluate')
    parser.add_argument('--tag', type=str, default='', help='information tag')
    parser.add_argument('--keep_gen_files', action='store_true', default=False, help='if True, generated files will not be removed')

    parser.add_argument('--n_way', type=int, default=5, help='number of novel classes per episode')
    parser.add_argument('--n_shot', type=int, default=5, help='number of training samples per novel class')
    parser.add_argument('--n_query', type=int, default=15, help='number of testing samples per novel class')
    parser.add_argument('--selected_novel_idx', type=str, default='', help='string for list of novel classes, used for deterministic novel classes and samples')

    
    parser.add_argument('--master_result_file', type=str, help='a common master result file for all episodes')
    parser.add_argument('--episode_dir', type=str, help='episode dir')
    args = parser.parse_args()

    if args.gpu >=0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")





    settings = {'device': device,
                'vae_epoch': args.vae_epoch,
                'vae_epoch_start_eval': args.vae_epoch_start_eval,
                'test_every_epoch': args.test_every_epoch,
                'batch_size': args.batch_size,
                'style_dim': args.style_dim,
                'content_dim': args.content_dim,
                'rtae_dim': args.rtae_dim,
                'n_generated': args.n_generated,
                'encoder_samplecontent_name': args.encoder_samplecontent,
                'encoder_samplepair_name': args.encoder_samplepair,
                'decoder_name': args.decoder,
                'RTAE_encoder_name': args.RTAE_encoder_name,
                'RTAE_decoder_name': args.RTAE_decoder_name,
                'gpu': args.gpu,
                'generate_base_samples': args.generate_base_samples,
                'convert_rtae_base_samples': args.convert_rtae_base_samples,
                'use_DBIndex': args.use_DBIndex,
                'keep_gen_files': args.keep_gen_files
                }
    if args.setting_file_path != '':
        print('Debug', args.setting_file_path)
        yml_setting = utils.load_yaml(args.setting_file_path)
        utils.embed_setting(settings, yml_setting, 'batch_size')
        utils.embed_setting(settings, yml_setting, 'vae_epoch')
        utils.embed_setting(settings, yml_setting, 'vae_epoch_start_eval')
        utils.embed_setting(settings, yml_setting, 'style_dim')
        utils.embed_setting(settings, yml_setting, 'content_dim')
        utils.embed_setting(settings, yml_setting, 'rtae_dim')
        utils.embed_setting(settings, yml_setting, 'encoder_samplecontent_name')
        utils.embed_setting(settings, yml_setting, 'encoder_samplepair_name')
        utils.embed_setting(settings, yml_setting, 'decoder_name')
        utils.embed_setting(settings, yml_setting, 'RTAE_encoder_name')
        utils.embed_setting(settings, yml_setting, 'RTAE_decoder_name')
        utils.embed_setting(settings, yml_setting, 'loss_weight')
        utils.embed_setting(settings, yml_setting, 'classifier_settings')
        utils.embed_setting(settings, yml_setting, 'training_settings')

    train_loader, test_loader, data_info = init_dataloaders_v3_episode(dataset=args.dataset,
                                                            data_settings={'batch_size': settings['batch_size'],
                                                                           'cuda': args.cuda,
                                                                           'no_img_in_pair': yml_setting['no_img_in_pair'],
                                                                           'use_novel_in_train': yml_setting['use_novel_in_train'],
                                                                           'episode_info': {'n_novel_classes': args.n_way,
                                                                                            'support_size': args.n_shot,
                                                                                            'query_size': args.n_query,
                                                                                            'selected_novel_idx': args.selected_novel_idx},
                                                                            'use_backward_relationship': yml_setting['use_backward_relationship'],
                                                                            'generate_base_samples': args.generate_base_samples,
                                                                            'convert_rtae_base_samples': args.convert_rtae_base_samples})
    

    parent_base_path = args.episode_dir
    if not os.path.isdir(parent_base_path):
        try:
            os.makedirs(parent_base_path)
        except:
            print('Looks like parent_base_path exists already')
    outdir_path = os.path.join(parent_base_path,
                               '{:03}_{}'.format(len(next(os.walk(parent_base_path))[1]),
                                                             time.strftime("%m%d%H%M%S")))
    if not os.path.isdir(outdir_path):
        os.mkdir(outdir_path)
    settings['checkpoint_dir'] = os.path.join(outdir_path, 'checkpoint')
    settings['aug_data_dir'] = os.path.join(outdir_path, 'aug_data')
    settings['master_result_file'] = args.master_result_file


    record_exp_info.copy_src('.', os.path.join(outdir_path, 'source_code'))


    orig_stdout = sys.stdout
    logfile_path = os.path.join(outdir_path, 'log.txt')
    settings['exp_dir'] = outdir_path
    if args.log_path == True:
        sys.stdout = record_exp_info.Logger(logfile_path, _write_to_stdout=True)
    else:
        sys.stdout = record_exp_info.Logger(logfile_path, _write_to_stdout=False)


    print(' '.join(sys.argv))
    print(args.setting_file_path)
    print("Setting\n{}".format(json.dumps(yml_setting, indent=4)))


    novel_support_file = os.path.join(outdir_path, 'novel_train.pkl')
    novel_query_file = os.path.join(outdir_path, 'novel_test.pkl')
    with open(novel_support_file, 'wb') as handle:
        pickle.dump(data_info['data_novel_support'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(novel_query_file, 'wb') as handle:
        pickle.dump(data_info['data_novel_query'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    data_info['novel_train_path'] = novel_support_file
    data_info['novel_test_path']  = novel_query_file
    settings['data_info'] = data_info


    train(train_loader, test_loader, settings)



    if settings['keep_gen_files'] == False:
        os.remove(novel_support_file)
        os.remove(novel_query_file)


    sys.stdout.close()
    sys.stdout = orig_stdout




if __name__ == "__main__":
    main()

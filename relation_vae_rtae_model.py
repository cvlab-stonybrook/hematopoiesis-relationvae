import torch
from torch import nn, optim
from torch.nn import functional as F
import sys
sys.path.append('..')
import routines.aux as aux
import scipy
import numpy as np
from .model_architectures import Encoder_SamplePair
from .model_architectures import Encoder_SampleContent
from .model_architectures.Decoder import *
from .model_architectures import RTAE
import copy


class Relation_VAE_RTAE(nn.Module):
    def __init__(self, _model_settings):
        super(Relation_VAE_RTAE, self).__init__()

        
        self.last_updated_epoch = -1
        self.optimizer = None
        self.org_model_settings = _model_settings
        self.model_settings = copy.deepcopy(self.org_model_settings)
        self.update_hyperparam_list(_update_info={'current_epoch': 0}, _update_optim=False)

        self.rtae_enc = RTAE.__dict__[self.model_settings['RTAE_encoder_name']](_settings={'input_dim': self.model_settings['input_dim'], 'embedding_dim': self.model_settings['rtae_dim']})
        self.rtae_dec = RTAE.__dict__[self.model_settings['RTAE_decoder_name']](_settings={'embedding_dim': self.model_settings['rtae_dim'], 'input_dim': self.model_settings['input_dim']})

        encoder_settings = {'input_dim': self.model_settings['rtae_dim'],
                            'style_dim': self.model_settings['style_dim'],
                            'content_dim': self.model_settings['content_dim']}
        self.encoder_samplepair = Encoder_SamplePair.encoder_mapping[self.model_settings['encoder_samplepair_name']](encoder_settings)
        self.encoder_samplecontent = Encoder_SampleContent.encoder_mapping[self.model_settings['encoder_samplecontent_name']](encoder_settings)
        decoder_settings = {'input_dim': self.model_settings['rtae_dim'],
                            'style_dim': self.model_settings['style_dim'],
                            'content_dim': self.model_settings['content_dim'],
                            'bilinear': True}
        self.decoder = decoder_mapping[self.model_settings['decoder_name']](decoder_settings)

        self.inter_class_logvar = torch.nn.Parameter(torch.rand(self.model_settings['content_dim'], requires_grad=True))

        
        self.update_hyperparam_list(_update_info={'current_epoch': 0}, _update_optim=True)

    def initialize(self):
        '''
            This function is depricated - initialization is run automatically in the constructor
        '''
        
        pass


    def encode(self, x):
        pass

    def decode(self, style_z, content_z):
        return self.decoder(style_z, content_z)

    
    def forward(self, x):
        return None

    def encode_data_group(self, x1_org, x2_org, labels1, labels2, sampling_content=True):
        x1 = self.rtae_enc(x1_org)
        x2 = self.rtae_enc(x2_org)

        x1_org_recon = self.rtae_dec(x1)
        x2_org_recon = self.rtae_dec(x2)

        
        style_mu1, style_logvar1, style_mu2, style_logvar2, content_mu2, content_logvar2 = self.encoder_samplepair(x1, x2)

        group_content_mu2, group_content_logvar2 = aux.accumulate_group_evidence(content_mu2, content_logvar2,
                                                                                labels_batch=torch.zeros(content_mu2.shape[0],1),
                                                                                device=self.model_settings['device'])
        style_z2 = aux.reparameterize(training=True, mu=style_mu2, logvar=style_logvar2)
        content_z2 = aux.group_wise_reparameterize(training=sampling_content,
                                                  mu=group_content_mu2, logvar=group_content_logvar2,
                                                  labels_batch=torch.zeros(group_content_mu2.shape[0],1),
                                                  device=self.model_settings['device'])



        content_mu1, content_logvar1 = self.encoder_samplecontent(content_z2, x1)
        group_content_mu1, group_content_logvar1 = aux.accumulate_group_evidence(content_mu1, content_logvar1,
                                                                                 labels_batch=torch.zeros(content_mu1.shape[0], 1),
                                                                                 device=self.model_settings['device'])
        style_z1 = aux.reparameterize(training=True, mu=style_mu1, logvar=style_logvar1)
        content_z1 = aux.group_wise_reparameterize(training=sampling_content,
                                                   mu=group_content_mu1, logvar=group_content_logvar1,
                                                   labels_batch=torch.zeros(group_content_mu1.shape[0], 1),
                                                   device=self.model_settings['device'])

        return {'style_mu1': style_mu1, 'style_logvar1': style_logvar1,
                'group_content_mu1': group_content_mu1, 'group_content_logvar1': group_content_logvar1,
                'style_z1': style_z1, 'content_z1': content_z1,
                'style_mu2': style_mu2, 'style_logvar2': style_logvar2,
                'group_content_mu2': group_content_mu2, 'group_content_logvar2': group_content_logvar2,
                'style_z2': style_z2,  'content_z2': content_z2,
                'x1': x1,
                'x2': x2,
                'x1_org_recon': x1_org_recon,
                'x2_org_recon': x2_org_recon
                }

        

    
    def vae_loss(self, recon_x, x, style_mu, style_logvar, content_mu, content_logvar):
        '''
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD_style = -0.5 * torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())
        KLD_content = -0.5 * torch.sum(1 + content_logvar - content_mu.pow(2) - content_logvar.exp())

        return BCE + KLD_style + KLD_content
        '''
        pass

    def relation_vae_loss(self, vars):
        rtae_mse1 = F.mse_loss(input=vars['x1_org'], target=vars['x1_org_recon'], reduction='sum')
        rtae_mse2 = F.mse_loss(input=vars['x2_org'], target=vars['x2_org_recon'], reduction='sum')
        rtae_mse = rtae_mse1 + rtae_mse2

        BCE1 = F.mse_loss(input=vars['recon_x1'], target=vars['x1'], reduction='sum')
        BCE2 = F.mse_loss(input=vars['recon_x2'], target=vars['x2'], reduction='sum')

        KLD_style1 = -0.5 * torch.sum(1 + vars['style_logvar1'] - vars['style_mu1'].pow(2) - vars['style_logvar1'].exp())
        KLD_style2 = -0.5 * torch.sum(1 + vars['style_logvar2'] - vars['style_mu2'].pow(2) - vars['style_logvar2'].exp())

        sub1 = self.inter_class_logvar - vars['group_content_logvar1']
        sub2 = vars['content_z2'] - vars['group_content_mu1']
        
        KLD_content1_temp = (-sub1).exp() + ((sub2.pow(2)) / (self.inter_class_logvar.exp())) + sub1
        KLD_content1 = 0.5 * torch.sum(KLD_content1_temp)

        KLD_content2 = -0.5 * torch.sum(1 + vars['group_content_logvar2'] - vars['group_content_mu2'].pow(2)
                                        - vars['group_content_logvar2'].exp())

        inter_class_logvar_loss = torch.linalg.norm(self.inter_class_logvar, ord=2)

        lweight = self.model_settings['loss_weight']
        loss = lweight['bce1']*BCE1 + lweight['bce2']*BCE2 + \
               lweight['kl_style1']*KLD_style1 + lweight['kl_style2']*KLD_style2 + \
               lweight['kl_content1']*KLD_content1 + lweight['kl_content2']*KLD_content2 + \
               lweight['inter_class_logvar_loss'] * inter_class_logvar_loss + \
               lweight['rtae_mse_loss'] * rtae_mse

        return {'loss': loss, 'bce': (BCE1+BCE2).item(), 'klstyle': (KLD_style1+KLD_style2).item(),
                'klcontent1': KLD_content1.item(), 'klcontent2': KLD_content2.item(),
                'inter_class_logvar_loss': inter_class_logvar_loss.item(),
                'rtae_mse': rtae_mse.item()}

    def calculate_grad(self, data1, data2, labels1, labels2):
        '''
        :param data1: n_sample x data_dim: data of class 1
        :param data2: n_sample x data_dim: data of class 2
        :param labels1: n_samples:   labels of class 1
        :param labels2: n_samples:   labels of class 2
        :return:
        '''
        
        encoded_info = self.encode_data_group(data1, data2, labels1, labels2, sampling_content=True)

        recon_data1 = self.decode(encoded_info['style_z1'], encoded_info['content_z1'])
        recon_data2 = self.decode(encoded_info['style_z2'], encoded_info['content_z2'])
        encoded_info['x1_org'] = data1
        encoded_info['x2_org'] = data2
        encoded_info['recon_x1'] = recon_data1
        encoded_info['recon_x2'] = recon_data2

        
        lossinfo = self.relation_vae_loss(encoded_info)

        return lossinfo

    def train_model(self, data_batch1, data_batch2, labels_batch1, labels_batch2, train_settings):
        '''
        :param data_batch1: batch_size x n_sample x data_dim: data of class 1, each element of batch is n sample of a single class
        :param data_batch2: batch_size x n_sample x data_dim: data of class 2, each element of batch is n sample of a single class
        :param labels_batch1: batch_size x n_samples:   labels of class 1
        :param labels_batch2: batch_size x n_samples:   labels of class 2
        :return:
        '''
        self.update_hyperparam_list(_update_info=train_settings, _update_optim=True)
        self.train()
        self.optimizer.zero_grad()

        loss = 0
        lossinfo = {'bce': 0, 'klstyle': 0, 'klcontent1': 0, 'klcontent2': 0, 'inter_class_logvar_loss': 0, 'rtae_mse': 0}
        for ipair in range(data_batch1.shape[0]):
            lossinfo_pair = self.calculate_grad(data_batch1[ipair], data_batch2[ipair], labels_batch1[ipair], labels_batch2[ipair])
            loss += lossinfo_pair['loss']
            for lossname in lossinfo.keys():
                lossinfo[lossname] += lossinfo_pair[lossname]

        loss.backward()
        self.optimizer.step()

        lossinfo['loss'] = loss.item()
        lossinfo['n_samples'] = data_batch1.shape[0]*data_batch1.shape[1]
        return lossinfo
        


    def test_model(self, data1, data2,  labels_batch1, labels_batch2):
        self.eval()
        return None


    def generate_samples_from_fewshot(self, fewshot_data_batch, neighbor_data_batch, n_generated):
        '''
                :param fewshot_data: real data from novel classes, fewshot_data should be from the same class
                :param data_neighbor: the images of the neighbor classes, these are for calculating c2. This must have the same size as fewshot_data
                :param n_generated: number of generated data from this class
                :return: a bunch of generated data
                '''
        assert fewshot_data_batch.shape[0] == 1
        assert fewshot_data_batch.shape == neighbor_data_batch.shape

        self.eval()

        fewshot_data = fewshot_data_batch[0]
        neighbor_data = neighbor_data_batch[0]

        labels_batch1 = torch.zeros(len(fewshot_data))
        labels_batch2 = torch.zeros(len(neighbor_data))
        encoded_info = self.encode_data_group(fewshot_data, neighbor_data, labels_batch1, labels_batch2, sampling_content=False)

        
        content_z_to_gen = encoded_info['content_z1'][0].expand(n_generated, -1)

        randoms_np = np.array([np.random.normal(0, 1, self.model_settings['style_dim']) for _ in range(n_generated)])
        randoms = torch.tensor(randoms_np, dtype=torch.float).to(self.model_settings['device'])

        gen_data = self.decode(randoms, content_z_to_gen)

        return gen_data


    def generate_analysis_from_fewshot(self, fewshot_data_batch, neighbor_data_batch, n_generated):
        '''
                :param fewshot_data: real data from novel classes, fewshot_data should be from the same class
                :param data_neighbor: the images of the neighbor classes, these are for calculating c2. This must have the same size as fewshot_data
                :param n_generated: number of generated data from this class
                :return: a bunch of generated data
                '''
        assert fewshot_data_batch.shape[0] == 1
        assert fewshot_data_batch.shape == neighbor_data_batch.shape

        self.eval()

        fewshot_data = fewshot_data_batch[0]
        neighbor_data = neighbor_data_batch[0]

        labels_batch1 = torch.zeros(len(fewshot_data))
        labels_batch2 = torch.zeros(len(neighbor_data))
        encoded_info = self.encode_data_group(fewshot_data, neighbor_data, labels_batch1, labels_batch2, sampling_content=False)

        
        content_z_to_gen = encoded_info['content_z1'][0].expand(n_generated, -1)

        randoms_np = np.array([np.random.normal(0, 1, self.model_settings['style_dim']) for _ in range(n_generated)])
        randoms = torch.tensor(randoms_np, dtype=torch.float).to(self.model_settings['device'])

        gen_data = self.decode(randoms, content_z_to_gen)

        return {'gen_data':gen_data, 'group_content_mu':encoded_info['group_content_mu1'], 'group_content_logvar':encoded_info['group_content_logvar1']}



    def reparameterize_depricated(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def change_learning_rate(self, _learning_rate_coefficient):
        ''' Change learning rates of the optimizer by a coefficient '''
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] * _learning_rate_coefficient

    def convert_rtae(self, org_data_batch):
        return self.rtae_enc(org_data_batch)

    def update_hyperparam(self, hyperparam_dict, dest_hyperparam, _update_info):
        '''
            hyperparam_dict is supposed to be either a value or a list with 'tune_epoch' at the beginning, next element is '0' for starting with epoch 0
        '''
        current_epoch = _update_info['current_epoch']
        
        for hyperparam_subname in hyperparam_dict.keys():
            sub_hyperparam = hyperparam_dict[hyperparam_subname]
            if isinstance(sub_hyperparam, list):
                if sub_hyperparam[0] == 'tune_epoch':
                    
                    epoch_stamp_list = sub_hyperparam[1::2]
                    value_stamp_list = sub_hyperparam[2::2]
                    set_flag = False
                    for epoch_idx, epoch_stamp in enumerate(epoch_stamp_list[:-1]):
                        next_epoch_stamp = epoch_stamp_list[epoch_idx+1]
                        if current_epoch >= epoch_stamp and current_epoch < next_epoch_stamp:
                            print('Update {} from {} to {}'.format(hyperparam_subname, dest_hyperparam[hyperparam_subname], value_stamp_list[epoch_idx]))
                            dest_hyperparam[hyperparam_subname] = value_stamp_list[epoch_idx]
                            set_flag = True
                            print('Set param', hyperparam_subname, value_stamp_list[epoch_idx])
                            break
                    if set_flag == False:
                        print('Update {} from {} to {}'.format(hyperparam_subname, dest_hyperparam[hyperparam_subname], value_stamp_list[-1]))
                        dest_hyperparam[hyperparam_subname] = value_stamp_list[-1]
            elif isinstance(sub_hyperparam, dict):
                self.update_hyperparam(hyperparam_dict=sub_hyperparam, dest_hyperparam=dest_hyperparam[hyperparam_subname], _update_info=_update_info)

    def update_hyperparam_list(self, _update_info, _update_optim):
        if self.last_updated_epoch < _update_info['current_epoch']:
            print('Update params for epoch ', _update_info['current_epoch'])
            self.update_hyperparam(hyperparam_dict=self.org_model_settings, dest_hyperparam=self.model_settings, _update_info=_update_info)

            if _update_optim == True:
                
                if self.optimizer is None:
                    self.optimizer = optim.Adam(self.parameters(), lr=self.model_settings['training_settings']['learning_rate'])
                else:
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.model_settings['training_settings']['learning_rate']
                
                self.last_updated_epoch = _update_info['current_epoch']

            
            if self.model_settings['loss_weight']['rtae_mse_loss'] == 0:
                for name, p in self.named_parameters():
                    if 'rtae' in name:
                        p.requires_grad = False
            else:
                for name, p in self.named_parameters():
                    if 'rtae' in name:
                        p.requires_grad = True
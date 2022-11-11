import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sklearn.metrics as skm
import warnings

from models.resnet import ResNet
from models.cca import CCA
from models.scr import SCR, SelfCorrelationComputation



class MAGF(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        self.encoder = ResNet(args=args)
        self.encoder_dim = 640  #640
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)

        self.scr_module = self._make_scr_layer(planes=[640, 64, 64, 64, 640])  #640
        self.cca_module = CCA(kernel_sizes=[3, 3], planes=[16, 1])
        self.cca_1x1 = nn.Sequential(
            nn.Conv2d(self.encoder_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def _make_scr_layer(self, planes):
        stride, kernel_size, padding = (1, 1, 1), (5, 5), 2
        layers = list()

        if self.args.self_method == 'scr':
            corr_block = SelfCorrelationComputation(kernel_size=kernel_size, padding=padding)
            self_block = SCR(planes=planes, stride=stride)
        else:
            raise NotImplementedError

        if self.args.self_method == 'scr':
            layers.append(corr_block)
        layers.append(self_block)
        return nn.Sequential(*layers)

    def forward(self, input):
        if self.mode == 'fc':
            return self.fc_forward(input)
        elif self.mode == 'encoder':
            return self.encode(input, False)
        elif self.mode == 'cross':
            spt, qry = input
            return self.cross_represent(spt, qry)
        elif self.mode == 'cca':
            spt, qry = input
            return self.cca(spt, qry)
        elif self.mode == 'mid':
            spt, qry = input
            return self.mid_represent(spt, qry)
        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])
        return self.fc(x)

    def cca(self, spt, qry):
        spt = spt.squeeze(0)

        # shifting channel activations by the channel mean
        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry)

        # (S * C * Hs * Ws, Q * C * Hq * Wq) -> Q * S * Hs * Ws * Hq * Wq
        corr4d = self.get_4d_correlation_map(spt, qry)
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()

        # corr4d refinement
        corr4d = self.cca_module(corr4d.view(-1, 1, H_s, W_s, H_q, W_q))
        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)
        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)

        # normalizing the entities for each side to be zero-mean and unit-variance to stabilize training
        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)

        # applying softmax for each side
        corr4d_s = F.softmax(corr4d_s / self.args.temperature_attn, dim=2)
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / self.args.temperature_attn, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)

        # suming up matching scores
        attn_s = corr4d_s.sum(dim=[4, 5])
        attn_q = corr4d_q.sum(dim=[2, 3])

        # applying attention
        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)

        # averaging embeddings for k > 1 shots
        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)

        # In the main paper, we present averaging in Eq.(4) and summation in Eq.(5).
        # In the implementation, the order is reversed, however, those two ways become eventually the same anyway :)
        spt_attended_pooled = spt_attended.mean(dim=[-1, -2])
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2])
        qry_pooled = qry.mean(dim=[-1, -2])

        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)

        if self.training:
            return similarity_matrix / self.args.temperature, self.fc(qry_pooled)
        else:
            return similarity_matrix / self.args.temperature

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def get_4d_correlation_map(self, spt, qry):
        '''
        The value H and W both for support and query is the same, but their subscripts are symbolic.
        :param spt: way * C * H_s * W_s
        :param qry: num_qry * C * H_q * W_q
        :return: 4d correlation tensor: num_qry * way * H_s * W_s * H_q * W_q
        :rtype:
        '''
        way = spt.shape[0]
        num_qry = qry.shape[0]

        # reduce channel size via 1x1 conv
        spt = self.cca_1x1(spt)
        qry = self.cca_1x1(qry)

        # normalize channels for later cosine similarity
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)

        # num_way * C * H_p * W_p --> num_qry * way * H_p * W_p
        # num_qry * C * H_q * W_q --> num_qry * way * H_q * W_q
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)
        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)
        return similarity_map_einsum

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def encode(self, x, do_gap=True):
        x = self.encoder(x)

        if self.args.self_method:
            identity = x
            x = self.scr_module(x)

            if self.args.self_method == 'scr':
                x = x + identity
            x = F.relu(x, inplace=True)

        if do_gap:
            return F.adaptive_avg_pool2d(x, 1)
        else:
            return x

    def get_mid_re_mid_feature(self,data):
        # s1, s2, s3, s4 = x.shape
        # re_mid = torch.empty(s1, s2, s3, s4 )
        H = data.size(-2)
        W = data.size(-1)
        mH = int(H / 2 + 1)
        mW = int(W / 2 + 1)
        mH_t = int((H - mH) / 2)
        mH_b = H - int((H - mH) / 2) - 1
        mW_l = int((W - mW) / 2)
        mW_r = H - int((W - mW) / 2) - 1

        mid = data[:, :, mH_t:mH_b, mW_l:mW_r]
        re_mid_t = data[:, :, :mH_t, mW_l:mW_r]
        re_mid_b = data[:, :, mH_b:, mW_l:mW_r]
        re_mid_tb = torch.cat((re_mid_t, re_mid_b), dim=2)

        re_mid_l = data[:, :, mH_t:mH_b, :mW_l]
        re_mid_r = data[:, :, mH_t:mH_b, mW_r:]
        re_mid_lr = torch.cat((re_mid_l, re_mid_r), dim=-1)

        return mid, re_mid_tb, re_mid_lr

    def cross_represent(self, spt, qry):
        # shifting channel activations by the channel mean
        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry)

        # get spt and qry mid, re_mid_tb, re_mid_lr feature respectively
        spt_mid, spt_re_mid_tb, spt_re_mid_lr = self.get_mid_re_mid_feature(spt)
        qry_mid, qry_re_mid_tb, qry_re_mid_lr = self.get_mid_re_mid_feature(qry)

        # N_s * N_C * H * W --->  N_q * N_s * N_C * H * W ---> N_q * N_s * N_C * HW
        spt_mid = spt_mid.unsqueeze(0).repeat(qry_mid.shape[0], 1, 1, 1, 1)
        spt_re_mid_tb = spt_re_mid_tb.unsqueeze(0).repeat(qry_mid.shape[0], 1, 1, 1, 1)
        spt_re_mid_lr = spt_re_mid_lr.unsqueeze(0).repeat(qry_mid.shape[0], 1, 1, 1, 1)
        N_q, N_s, N_C, _, _ = spt_re_mid_tb.shape
        spt_mid = spt_mid.view(N_q, self.args.shot, self.args.way, N_C, -1)
        spt_re_mid_tb = spt_re_mid_tb.view(N_q, self.args.shot, self.args.way, N_C, -1)
        spt_re_mid_lr = spt_re_mid_lr.view(N_q, self.args.shot, self.args.way, N_C, -1)
        spt_mid = self.gaussian_normalize(spt_mid, dim=-1)
        spt_re_mid_tb = self.gaussian_normalize(spt_re_mid_tb, dim=-1)
        spt_re_mid_lr = self.gaussian_normalize(spt_re_mid_lr, dim=-1)

        spt_re_mid_tblr = torch.cat((spt_re_mid_tb, spt_re_mid_lr), dim=-1)

        # N_q * N_C * H * W --->  N_q * N_s * N_C * H * W ---> N_q * N_s * N_C * HW
        qry_mid = qry_mid.unsqueeze(1).repeat(1, spt_mid.shape[2], 1, 1, 1)

        qry_re_mid_tb = qry_re_mid_tb.unsqueeze(1).repeat(1, spt_mid.shape[2], 1, 1, 1)
        qry_re_mid_lr = qry_re_mid_lr.unsqueeze(1).repeat(1, spt_mid.shape[2], 1, 1, 1)
        if self.args.shot == 1:
            qry_mid = qry_mid.view(N_q, self.args.shot, self.args.way, N_C, -1)
            qry_re_mid_tb = qry_re_mid_tb.view(N_q, self.args.shot, self.args.way, N_C, -1)
            qry_re_mid_lr = qry_re_mid_lr.view(N_q, self.args.shot, self.args.way, N_C, -1)
        if self.args.shot > 1:
            qry_mid = qry_mid.view(N_q, self.args.shot, 1, N_C, -1)
            qry_re_mid_tb = qry_re_mid_tb.view(N_q, self.args.shot, 1, N_C, -1)
            qry_re_mid_lr = qry_re_mid_lr.view(N_q, self.args.shot, 1, N_C, -1)

        qry_mid = self.gaussian_normalize(qry_mid, dim=-1)
        qry_re_mid_tb = self.gaussian_normalize(qry_re_mid_tb, dim=-1)
        qry_re_mid_lr = self.gaussian_normalize(qry_re_mid_lr, dim=-1)

        qry_re_mid_tblr = torch.cat((qry_re_mid_tb, qry_re_mid_lr), dim=-1)

        if self.args.shot >1 :
            qry_re_mid_tblr = qry_re_mid_tblr.mean(dim=2)
            spt_re_mid_tblr = spt_re_mid_tblr.mean(dim=2)
            spt_mid = spt_mid.mean(dim=2)
            qry_mid = qry_mid.mean(dim=2)


        # spt mid --> qry re_mid
        sptmid_qrytblr = torch.cat((spt_mid, qry_re_mid_tblr), dim=-1)

        # qry re_mid --> spt mid
        qrymid_spttblr = torch.cat((qry_mid, spt_re_mid_tblr), dim=-1)


        # normalize to calculate similarity
        if self.args.shot ==1:
            sptmid_qrytblr_pro = sptmid_qrytblr.mean(dim=[1, -1])
            qrymid_spttblr_pro = qrymid_spttblr.mean(dim=[1, -1])
        elif self.args.shot > 1:
            sptmid_qrytblr_pro = sptmid_qrytblr.mean(dim=-1)
            qrymid_spttblr_pro = qrymid_spttblr.mean(dim=-1)

        similarity_matrix = F.cosine_similarity(sptmid_qrytblr_pro, qrymid_spttblr_pro, dim=-1)

        # sptmid_qrytblr_numpy = sptmid_qrytblr.detach().cpu().numpy()
        # qrymid_spttblr_numpy = qrymid_spttblr.detach().cpu().numpy()
        # sptmid_qrytblr_numpy = np.reshape(sptmid_qrytblr_numpy, -1)
        # qrymid_spttblr_numpy = np.reshape(qrymid_spttblr_numpy, -1)
        # # Mi_score = self.MI_score(sptmid_qrytblr_numpy,qrymid_spttblr_numpy)
        # Mi_score = self.hxx(sptmid_qrytblr_numpy, qrymid_spttblr_numpy)
        #return similarity_matrix, Mi_score

        return similarity_matrix

    def mid_represent(self, spt, qry):
        # shifting channel activations by the channel mean
        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry)

        # get spt and qry mid, re_mid_tb, re_mid_lr feature respectively
        spt_mid, NA1, NA2 = self.get_mid_re_mid_feature(spt)
        qry_mid, NA1, NA2= self.get_mid_re_mid_feature(qry)

        N_q,N_c, NA1, NA2= qry_mid.shape

        # unsqueeze
        spt_mid = spt_mid.unsqueeze(0).repeat(qry_mid.shape[0], 1, 1, 1, 1)
        qry_mid = qry_mid.unsqueeze(1).repeat(1, spt_mid.shape[1], 1, 1, 1)

        spt_mid = spt_mid.view(N_q, self.args.shot, self.args.way, N_c, -1)
        qry_mid = qry_mid.view(N_q, self.args.shot, self.args.way, N_c, -1)

        spt_mid = self.gaussian_normalize(spt_mid, dim=-1)
        qry_mid = self.gaussian_normalize(qry_mid, dim=-1)

        # normalize to calculate similarity
        spt_mid_pro = spt_mid.mean(dim=[1, -1])
        qry_mid_pro = qry_mid.mean(dim=[1, -1])

        similarity_matrix_sq = F.cosine_similarity(spt_mid_pro, qry_mid_pro, dim=-1)

        return similarity_matrix_sq

    def MI_score(self, x, y):
        warnings.filterwarnings('ignore')
        MI_sq = skm.mutual_info_score(x, y)
        return round(MI_sq/10, 3)

    def Avg_MI_Score(self, x, y):
        warnings.filterwarnings('ignore')
        num_samples = x.shape[0]
        MI_socre = 0
        for num_sample in range(num_samples):
            x_sample = x[num_sample,:,:]
            y_sample = y[num_sample,:,:]
            MI_sq = skm.mutual_info_score(x_sample, y_sample)
            MI_socre += MI_sq
        MI_socre = round(MI_socre/num_samples/10, 3)
        return MI_socre

    def hxx(self, x, y):
        size = x.shape[-1]
        px = np.histogram(x, 256, (0, 255))[0] / size
        py = np.histogram(y, 256, (0, 255))[0] / size
        hx = - np.sum(px * np.log(px + 1e-8))
        hy = - np.sum(py * np.log(py + 1e-8))

        hxy = np.histogram2d(x, y, 256, [[0, 255], [0, 255]])[0]
        hxy /= (1.0 * size)
        hxy = - np.sum(hxy * np.log(hxy + 1e-8))

        r = hx + hy - hxy
        return round(r, 3)
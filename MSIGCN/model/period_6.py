from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import BaseModel as BaseBlock
import utils.util as util
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.opt import Options
"""
在model1的基础上添加st_gcn,修改 bn 
"""

class MultiStageModel(Module):
    def __init__(self, opt):
        super(MultiStageModel, self).__init__()

        self.opt = opt
        self.kernel_size = opt.kernel_size
        self.d_model = opt.d_model
        # self.seq_in = seq_in
        self.dct_n = opt.dct_n
        # ks = int((kernel_size + 1) / 2)
        assert opt.kernel_size == 10

        self.in_features = opt.in_features
        self.num_stage = opt.num_stage
        self.node_n = self.in_features//3

        self.encoder_layer_num = 1
        self.decoder_layer_num = 2

        self.input_n = opt.input_n
        self.output_n = opt.output_n

        self.gcn_encoder1 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                               node_n=self.node_n,
                                               seq_len=self.dct_n,
                                               p_dropout=opt.drop_out,
                                               num_stage=self.encoder_layer_num)

        self.gcn_decoder1 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                               node_n=self.node_n,
                                               seq_len=self.dct_n*2,
                                               p_dropout=opt.drop_out,
                                               num_stage=self.decoder_layer_num)

        self.gcn_encoder2 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder2 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)

        self.gcn_encoder3 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder3 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)

        self.gcn_encoder4 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder4 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)
        self.gcn_encoder5 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder5 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)
        self.gcn_encoder6 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                  node_n=self.node_n,
                                                  seq_len=self.dct_n,
                                                  p_dropout=opt.drop_out,
                                                  num_stage=self.encoder_layer_num)

        self.gcn_decoder6 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                                  node_n=self.node_n,
                                                  seq_len=self.dct_n * 2,
                                                  p_dropout=opt.drop_out,
                                                  num_stage=self.decoder_layer_num)

        self.mlp1 = BaseBlock.Mlp(in_channal=3, out_channal=self.d_model)
        self.mlp2 = BaseBlock.Mlp(in_channal=3, out_channal=self.d_model)
        self.mlp3 = BaseBlock.Mlp(in_channal=3, out_channal=self.d_model)
        self.mlp4 = BaseBlock.Mlp(in_channal=3, out_channal=self.d_model)
        self.mlp5 = BaseBlock.Mlp(in_channal=3, out_channal=self.d_model)
        self.mlp6 = BaseBlock.Mlp(in_channal=3, out_channal=self.d_model)

    def forward(self, src, input_n=10, output_n=10, itera=1):
        output_n = self.output_n
        input_n = self.input_n

        bs = src.shape[0]
        # [2000,512,22,20]
        dct_n = self.dct_n
        idx = list(range(self.kernel_size)) + [self.kernel_size -1] * output_n
        # [b,20,66]
        input_gcn = src[:, idx].clone()

        dct_m, idct_m = util.get_dct_matrix(input_n + output_n)
        dct_m = torch.from_numpy(dct_m).float().to(self.opt.cuda_idx)
        idct_m = torch.from_numpy(idct_m).float().to(self.opt.cuda_idx)

        # [b,20,66] -> [b,66,20]
        input_gcn_dct = torch.matmul(dct_m[:dct_n], input_gcn).permute(0, 2, 1)


        # [b,66,20]->[b,22,3,20]->[b,3,22,20]->[b,512,22,20]
        input_gcn_dct = input_gcn_dct.reshape(bs, self.node_n, -1, self.dct_n).permute(0, 2, 1, 3)

        #stage1
        rst_1 = torch.zeros_like(input_gcn.reshape(bs, self.dct_n,-1))
        rst_1[:, 1:] = input_gcn[:, :-1] - input_gcn[:, 1:]
        rst_1 = self.mlp1(rst_1.reshape(bs, self.dct_n, -1, 3))
        latent_gcn_dct = self.gcn_encoder1(input_gcn_dct)
        #[b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, rst_1.transpose(1,-1)), dim=3)
        # latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)
        output_dct_1 = self.gcn_decoder1(latent_gcn_dct)[:, :, :, :dct_n]

        #stage2
        rst_2 = torch.zeros_like(output_dct_1)
        rst_2[:,1:] = output_dct_1[:,:-1] - output_dct_1[:,1:]
        rst_2 = self.mlp2(rst_2.reshape(bs, self.dct_n,-1,3))
        latent_gcn_dct = self.gcn_encoder2(output_dct_1)
        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, rst_2.transpose(1,-1)), dim=3)
        # latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)
        output_dct_2 = self.gcn_decoder2(latent_gcn_dct)[:, :, :, :dct_n]

        # #stage3
        rst_3 = torch.zeros_like(output_dct_2)
        rst_3[:, 1:] = output_dct_2[:, :-1] - output_dct_2[:, 1:]
        rst_3 = self.mlp3(rst_3.reshape(bs, self.dct_n, -1, 3))
        latent_gcn_dct = self.gcn_encoder3(output_dct_2)
        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, rst_3.transpose(1,-1)), dim=3)
        # latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)
        output_dct_3 = self.gcn_decoder3(latent_gcn_dct)[:, :, :, :dct_n]

        #stage4
        rst_4 = torch.zeros_like(output_dct_3)
        rst_4[:, 1:] = output_dct_3[:, :-1] - output_dct_3[:, 1:]
        rst_4 = self.mlp4(rst_4.reshape(bs, self.dct_n, -1, 3))
        latent_gcn_dct = self.gcn_encoder4(output_dct_3)
        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, rst_4.transpose(1, -1)), dim=3)
        # latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)
        output_dct_4 = self.gcn_decoder4(latent_gcn_dct)[:, :, :, :dct_n]

        # stage5
        rst_5 = torch.zeros_like(output_dct_4)
        rst_5[:, 1:] = output_dct_4[:, :-1] - output_dct_4[:, 1:]
        rst_5 = self.mlp5(rst_5.reshape(bs, self.dct_n, -1, 3))
        latent_gcn_dct = self.gcn_encoder5(output_dct_4)
        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, rst_5.transpose(1, -1)), dim=3)
        # latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)
        output_dct_5 = self.gcn_decoder5(latent_gcn_dct)[:, :, :, :dct_n]

        # #stage6
        rst_6 = torch.zeros_like(output_dct_5)
        rst_6[:, 1:] = output_dct_5[:, :-1] - output_dct_5[:, 1:]
        rst_6 = self.mlp6(rst_6.reshape(bs, self.dct_n, -1, 3))
        latent_gcn_dct = self.gcn_encoder6(output_dct_5)
        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, rst_6.transpose(1, -1)), dim=3)
        # latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)
        output_dct_6 = self.gcn_decoder6(latent_gcn_dct)[:, :, :, :dct_n]


        output_dct_1 = output_dct_1.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_2 = output_dct_2.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_3 = output_dct_3.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_4 = output_dct_4.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_5 = output_dct_5.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_6 = output_dct_6.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)

        # [b,20 66]->[b,20 66]
        output_1 = torch.matmul(idct_m[:, :dct_n], output_dct_1.permute(0, 2, 1))
        output_2 = torch.matmul(idct_m[:, :dct_n], output_dct_2.permute(0, 2, 1))
        output_3 = torch.matmul(idct_m[:, :dct_n], output_dct_3.permute(0, 2, 1))
        output_4 = torch.matmul(idct_m[:, :dct_n], output_dct_4.permute(0, 2, 1))
        output_5 = torch.matmul(idct_m[:, :dct_n], output_dct_5.permute(0, 2, 1))
        output_6 = torch.matmul(idct_m[:, :dct_n], output_dct_6.permute(0, 2, 1))

        return output_6, output_5, output_4, output_3, output_2, output_1


if __name__ == '__main__':
    option = Options().parse()
    option.d_model = 16
    model = MultiStageModel(opt=option).cuda()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    src = torch.FloatTensor(torch.randn((32, 35, 66))).cuda()
    output = model(src)

import torch
import torch.nn as nn
from pytorch_transformers import RobertaModel
from pytorch_transformers.modeling_bert import BertLayerNorm
from config.cfg import cfg, path, hyper_roberta
from torch.autograd import Variable
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gpu_id'])
device = torch.device(cfg['device'])

class Com(nn.Module):
    def __init__(self):
        super(Com, self).__init__()
        self.roberta = RobertaModel.from_pretrained(path['roberta_path'])

    def forward(self, pos, neg, input_x):
        mask1 = (input_x != 1).type(torch.long)

        input_x = self.roberta(input_x, attention_mask=mask1)
        input_x = input_x[0]

        pos = self.roberta(pos, attention_mask=mask1)
        pos = pos[0]

        neg = self.roberta(neg, attention_mask=mask1)
        neg = neg[0]

        dis = []

        dis1 = torch.pairwise_distance(input_x, pos, p=1)
        dis.append(dis1)
        dis2 = torch.pairwise_distance(input_x, neg, p=1)
        dis.append(dis2)

        dis = torch.tensor(dis).to(device)
        return dis







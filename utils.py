import argparse     
from transformers import AutoTokenizer,AutoModel
import torch
parser = argparse.ArgumentParser(description='parser example')

parser.add_argument('--batch_size', default=8, type=int, help='batch size')

parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')

parser.add_argument('--B1', default=0.9, type=float, help='B1 parameter of AdamW optimizer')

parser.add_argument('--B2', default=0.99, type=float, help='B2 parameter of AdamW optimizer.')

parser.add_argument('--epoch', default=10, type=int, help='epoch of train')

parser.add_argument('--data_path', default='./data/', type=str, help='dataset path')

parser.add_argument('--data_name', default='ReClor', type=str, help='ReClor or LogiQA')

parser.add_argument('--label_path', default='./label.npy', type=str, help='label of test sets')

parser.add_argument('--model_path', default='./net_params.pkl', type=str, help='path for saving the best parameter of Beta-LR')

parser.add_argument('--pretrained_model_path', default='roberta-large', type=str, help='path of pre-trained model from huggingface')

parser.add_argument('--embed_method', default='beta', type=str, help='beta or point, beta means training on Beta-LR, point means training on single vector for ablation study')

parser.add_argument('--operator_method', default='logic', type=str, help='logic or arithmetic, the way of operations in the intergation of logical information')

parser.add_argument('--union_method', default='renew', type=str, help='renew or initial, whether renew initial beta distributions or not')

parser.add_argument('--attention_layer_num', default=2, type=int, help='the number of attention layer in logical intersection operation')

parser.add_argument('--embedding_size', default=1024, type=int, help='embedding size of pre-trained model')

parser.add_argument('--beta_size', default=512, type=int, help='embedding size of beta distributions')

args = parser.parse_args()

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrain_model=AutoModel.from_pretrained('roberta-large').to(device)

tokenizer=AutoTokenizer.from_pretrained('roberta-large')

label_path=args.label_path

model_path=args.model_path

def collate_batch(data):
    unit_ids,unit_mask,unit_idx,unit_label=[],[],[],[]
    for unit in data:
        unit_ids.append(unit[0])
        unit_mask.append(unit[1])
        unit_idx.append(unit[2])
        unit_label.append(unit[3])
    return  unit_ids,unit_mask,unit_idx,unit_label
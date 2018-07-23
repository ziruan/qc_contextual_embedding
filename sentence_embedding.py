import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')
# Model parameters.
parser.add_argument('--data', type=str, default='./data/my_wechat_title',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./char_model1.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()


def get_representation(model, strings, detach_from_lm=True):
    ''''
    @parameters
    strings: [[word1, ..., wordn]]
    '''
    sequences_as_char_indices = []
    unk_idx = model.dictionary.word2idx.get("<unk>")
    # 一次处理多个句子
    for string in strings:
        char_indices = [model.dictionary.word2idx.get(char, unk_idx) for char in string]
        sequences_as_char_indices.append(char_indices)

    batch = Variable(torch.LongTensor(sequences_as_char_indices).transpose(0, 1))

    if torch.cuda.is_available():
        batch = batch.cuda()

    hidden = model.init_hidden(len(strings))
    prediction, rnn_output, hidden = model.forward(batch, hidden, True)

    if detach_from_lm:
        rnn_output = model.repackage_hidden(rnn_output)

    return rnn_output


def get_sent_pair_rep(sent_pair):
    left_sent = sent_pair[0]
    right_sent = sent_pair[1]
    left_sent_output = get_representation(model, left_sent)
    right_sent_output = get_representation(model, right_sent)

    return left_sent_output, right_sent_output


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

# 读取数据
sentences_lst = []
test_data_file = "test_sentences_pair.dat"
with open(test_data_file) as fin:
    for line in fin:
        arr = line.strip().split("\t")
        if len(arr) != 2:
            continue
        sentences_lst.append([[item.split()] for item in arr])

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
for sent_pair in sentences_lst:
    lout, rout = get_sent_pair_rep(sent_pair)
    #if lout.size(0) != rout.size(0):
    #    continue

    sim_score = cos(lout[lout.size(0) - 1], rout[rout.size(0) - 1])
    print(sent_pair)
    print(sim_score)






import torch
import torch.optim as optim
from collections import defaultdict
from tqdm import trange

from model import *
from utils.dataset import gen_data_list, gen_dataloader
from utils.load_data import load_data
from train_test import train, test

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ', device)

nets = defaultdict()
nets['agnn'] = AGNN
nets['gat'] = GAT
nets['gcn'] = GCN
nets['gin'] = GIN
nets['set2set'] = Set2SetNet
nets['sgcn'] = SGCN
nets['tag'] = TAG
nets['tag_jk'] = TAGWithJK
nets['tag_jk_learn'] = TAGWithJKLearn
nets['tag_learn'] = TAGLearn
nets['gcn_learn'] = GCNLearn
nets['sgcn_learn'] = SGCNLearn


def generate_data_info(subject_id, edge_type='corr', feature_type='psd_group'):
    print(f'Current directory: s{subject_id}')

    # load data from '*.npy' file
    x_train, x_test, y_train, y_test = load_data(subject_id)
    # num_classes = 2
    train_num, test_num = x_train.shape[0], x_test.shape[0]

    print('Loading training data...')
    train_list = gen_data_list(x_train, y_train,
                               edge_type=edge_type, feature_type=feature_type)
    print('Loading testing data...')
    test_list = gen_data_list(x_test, y_test,
                              edge_type=edge_type, feature_type=feature_type)

    data_info = defaultdict()
    data_info['subject_id'] = subject_id
    data_info['edge_type'] = edge_type
    data_info['feature_type'] = feature_type
    data_info['num_features'] = train_list[0].num_features
    # data_info['num_classes'] = num_classes
    data_info['train_num'] = train_num
    data_info['test_num'] = test_num
    data_info['train_lis'] = train_list
    data_info['test_lis'] = test_list

    return data_info


def main(nets, net_name, data_info, batch_size, epochs, num_iteration, logged=False):
    # num_classes = date_info['num_classes']
    subject_id = data_info['subject_id']
    edge_type = data_info['edge_type']
    feature_type = data_info['feature_type']
    num_features = data_info['num_features']

    train_loader = gen_dataloader(data_info['train_lis'],
                                  batch_size=batch_size)
    test_loader = gen_dataloader(data_info['test_lis'],
                                 batch_size=batch_size)

    for i in trange(num_iteration):
        # model initiation
        if net_name == 'tag_jk' or net_name == 'tag_jk_learn':
            model = nets[net_name](num_features, 4).to(device)
        else:
            model = nets[net_name](num_features).to(device)

        # criterion = torch.nn.BCELoss()
        # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.5]).to(device))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        log = []
        if logged:
            log.append(f's{subject_id:02d}\t{edge_type:<6s}\t{feature_type:<6s}\t'
                       f'{net_name:<8s}\t{batch_size:<4d}\t{epochs:<4d}\t')

        train(model, criterion, optimizer, train_loader, device,
              data_info['train_num'], epochs, logged)

        test(model, criterion, test_loader, device,
             data_info['test_num'], log, logged)


if __name__ == '__main__':
    """
    hyper-parameter search
    """
    _subject_id = [i for i in range(6, 14)]
    # _subject_id = [8, 12]
    # _edge_type = ['corr', 'wpli', 'plv', 'cg']
    _edge_type = ['cg']
    _feature_type = ['psd_group']
    # _feature_type = ['psd_group']
    _batch_size = [2, 4, 8, 16]
    # _net_name = [name for name in nets.keys()]
    _net_name = ['tag_learn']
    _epochs = [20, 30, 40, 60, 80]
    num_iteration = 5
    logged = True
    for subject_id in _subject_id:
        for edge_type in _edge_type:
            for feature_type in _feature_type:
                data_info = generate_data_info(subject_id, edge_type, feature_type)
                for batch_size in _batch_size:
                    for net_name in _net_name:
                        for epochs in _epochs:
                            main(nets, net_name, data_info, batch_size,
                                 epochs, num_iteration, logged)

    """
    single iteration test
    """
    # net_name = 'gcn'
    # subject_id = 12
    # batch_size = 8
    # data_info = generate_data_info(subject_id)
    #
    # epochs = 80
    # logged = False
    # # logged = True
    # main(nets, net_name, data_info, epochs, 1, logged)

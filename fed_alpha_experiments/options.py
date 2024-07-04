import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_rounds', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_clients', type=int, default=100, help="number of clients: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--bs', type=int, default=8, help="batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--wd', type=float, default=0.0, help="weight decay")

    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--dirichlet_beta', type=float, default='1.0', help="dirichlet sampling parameter")
    parser.add_argument('--config_file', type=str, default=None, help="path to config file")

    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    args = parser.parse_args()
    return args
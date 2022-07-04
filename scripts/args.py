from argparse import ArgumentParser, RawTextHelpFormatter, Namespace


def get_parser():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-m', '--model',    dest="model",   type=str, default='rdf',    help="choose model to train", choices=['ann', 'rdf', 'lgr'])
    parser.add_argument('-n', '--name',     dest="name",    type=str, default='ml',     help="model name")
    parser.add_argument('-d', '--data',     dest="data",    type=str, default='../data/elite.csv',     help="data source path")
    parser.add_argument('-ne', '--epochs',     dest="epochs",    type=int,  default=50,     help="epochs")
    parser.add_argument('-bs', '--batchsize',     dest="batch_size",    type=int, default=512,     help="batch size")
    return parser


def parse_args_to_dict(ns: Namespace):
    res = {k : v for k, v in ns._get_kwargs()}
    return res

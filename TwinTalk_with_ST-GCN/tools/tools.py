# st-gcn/tools/tools.py

import argparse
from torchlight import str2bool

def get_parser():
    parser = argparse.ArgumentParser(description='ST-GCN')

    parser.add_argument('--work_dir', default=None)
    parser.add_argument('--config', '-c', help='path to the configuration file')
    parser.add_argument('--model_saved_name', default=None)

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save_score', type=str2bool, default=False)
    parser.add_argument('--print_log', type=str2bool, default=True)

    # debug
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--test_feeder_args.debug', type=str2bool, default=False)
    parser.add_argument('--train_feeder_args.debug', type=str2bool, default=False)

    return parser

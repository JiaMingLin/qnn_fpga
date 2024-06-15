import argparse
parser = argparse.ArgumentParser(description="Extreme quantization of RNN-based models")
parser.add_argument('dataset', type=str, default='mnist', choices=['mnist', 'emg', 'ecg'])
parser.add_argument('batch', type=int, default='50')
parser.add_argument('epoch', type=int, default='10')
parser.add_argument('hidden', type=int, default='128')

parser.add_argument('--settings', nargs='+')

import os
import re
import argparse
from model_multi import Feature_colorization
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='horse', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# images in batch')
parser.add_argument('--dim', dest='dim', type=int, default=64)
parser.add_argument('--is_grayscale', dest='is_grayscale', type=int, default=False)
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--output_size', dest='output_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input color image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--train_type', dest='train_type', default='e_loss', help='which step to train')
parser.add_argument('--num_gpus', dest='num_gpus', default=2, help='number of gpus')

args = parser.parse_args()


def main(_):
	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)

	if not os.path.exists(args.test_dir):
		os.makedirs(args.test_dir)

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		model = Feature_colorization(
		sess, 
		train_type = args.train_type,
		checkpoint_dir = args.checkpoint_dir,
		test_dir = args.test_dir,
		dataset = args.dataset,
		is_grayscale = args.is_grayscale,
		learning_rate = args.learning_rate,
		epoch = args.epoch,
		dim = args.dim,
		fine_size = args.fine_size, 
		output_size = args.output_size, 
		batch_size = args.batch_size,
		input_c_dim = args.input_nc,
		output_c_dim = args.output_nc,
		num_gpus = args.num_gpus
		)
		if args.phase == 'train':
			model.train()
		else:
			model.test()

if __name__ == '__main__':
	tf.app.run()

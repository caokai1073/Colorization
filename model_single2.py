from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from ops import *
from six.moves import xrange

class Feature_colorization(object):
	def __init__(self, 
		sess, 
		train_type = 'e_loss',
		checkpoint_dir = None,
		test_dir = None,
		dataset = 'horse',
		is_grayscale = False,
		learning_rate = 0.0002,
		epoch = 200,
		dim = 64,
		fine_size = 256, 
		output_size = 256, 
		batch_size = 5,
		input_c_dim = 3,
		output_c_dim = 3
		):
		self.sess = sess
		self.train_type = train_type
		self.dataset = dataset
		self.test_dir = test_dir
		self.checkpoint_dir = checkpoint_dir
		self.is_grayscale = is_grayscale
		self.learning_rate = learning_rate
		self.epoch = epoch
		self.dim = dim
		self.fine_size = fine_size
		self.output_size = output_size
		self.batch_size = batch_size
		self.input_c_dim = input_c_dim
		self.output_c_dim = output_c_dim

		self.en_e1 = batch_norm(name='en_e1')
		self.en_e2 = batch_norm(name='en_e2')
		self.en_e3 = batch_norm(name='en_e3')
		self.en_e4 = batch_norm(name='en_e4')
		self.en_e5 = batch_norm(name='en_e5')
		self.en_e6 = batch_norm(name='en_e6')

		self.de_d0 = batch_norm(name='de_d0')
		self.de_d1 = batch_norm(name='de_d1')
		self.de_d2 = batch_norm(name='de_d2')
		self.de_d3 = batch_norm(name='de_d3')
		self.de_d4 = batch_norm(name='de_d4')
		self.de_d5 = batch_norm(name='de_d5')
		self.de_d6 = batch_norm(name='de_d6')
		
		self.f_f1 = batch_norm(name='f_f1')
		self.f_f2 = batch_norm(name='f_f2')
		self.f_q1 = batch_norm(name='f_q1')
		self.f_q2 = batch_norm(name='f_q2')

		self.bulid_encoder_model()

	def bulid_encoder_model(self):
		self.image_G = tf.placeholder(tf.float32, [self.batch_size, self.fine_size, self.fine_size, self.input_c_dim])
		self.image_C = tf.placeholder(tf.float32, [self.batch_size, self.fine_size, self.fine_size, self.input_c_dim])
		
		self.feature_G, self.e6, self.e5, self.e4, self.e3, self.e2, self.e1, self.e0 = self.Autoencoder(self.image_G, False)
		#feature layer: gray to color
		self.feature_G_ = self.Feature_G2C(self.feature_G)
		
		if self.train_type == 'e_loss':
			self.image_G_ = self.Autodecoder(self.feature_G, self.e6, self.e5, self.e4, self.e3, self.e2, self.e1, self.e0, False)
		else:
			self.image_G_ = self.Autodecoder(self.feature_G_, self.e6, self.e5, self.e4, self.e3, self.e2, self.e1, self.e0, False)	

		self.feature_C, self.e6, self.e5, self.e4, self.e3, self.e2, self.e1, self.e0 = self.Autoencoder(self.image_C, True)
		self.image_C_ = self.Autodecoder(self.feature_C, self.e6, self.e5, self.e4, self.e3, self.e2, self.e1, self.e0, True)

		self.encoder_loss = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.image_G_, self.image_C),2.0))+ 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.image_C_,self.image_C),2.0))
		self.feature_loss = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.feature_C, self.feature_G_),2.0))
		self.total_loss = self.encoder_loss + self.feature_loss

		t_vars = tf.trainable_variables()

		self.feature_vars = [var for var in t_vars if 'f_' in var.name]

		self.saver = tf.train.Saver()

	def Feature_G2C(self, feature_G):
		with tf.variable_scope("f_G2C") as scope:	
			#s = feature_G.get_shape()[1]
			s, dim = int(np.shape(feature_G)[1]), int(np.shape(feature_G)[3])
			s2, s4 = int(s/2), int(s/4)

			f0 = lrelu(conv2d(feature_G, dim*4, d_h=1, d_w=1, name='f_conv0'))
			# f0 is (2 x 2 x 2048)
			f1 = lrelu(self.f_f1(conv2d(f0, dim*8, name='f_conv1')))
			# f1 is (1 x 1 x 4096)
			#f2 = lrelu(self.f_f2(conv2d(f1, dim*4, name='f_conv2')))

			#q2 = lrelu(self.f_q2(deconv2d(f2, [self.batch_size, s4, s4, dim*2], name='f_deconv2')))
			q1 = lrelu(self.f_q1(deconv2d(f1, [self.batch_size, s2, s2, dim*4], name='f_deconv1')))
			# q1 is (2 x 2 x 2048)
			q0 = lrelu(deconv2d(q1, [self.batch_size, s, s, dim],  d_h=1, d_w=1, name='f_deconv0'))
			# q0 is (2 x 2 x 512)
			
			return q0		

	def Autoencoder(self, image, reuse=False):

		with tf.variable_scope("Autoencoder") as scope:
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False

			s = self.output_size
			s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
			#encoder
			e0 = lrelu(conv2d(image, self.dim, d_h=1, d_w=1, name='e0_conv'))
			# e0 is (256 x 256 x 64)
			e1 = lrelu(self.en_e1(conv2d(e0, self.dim, name='e1_conv')))
			# e1 is (128 x 128 x 64)
			e2 = lrelu(self.en_e2(conv2d(e1, self.dim*2, name='e2_conv')))
			# e2 is (64 x 64 x 128)
			e3 = lrelu(self.en_e3(conv2d(e2, self.dim*4, name='e3_conv')))
			# e3 is (32 x 32 x 256)
			e4 = lrelu(self.en_e4(conv2d(e3, self.dim*8, name='e4_conv')))
			# e4 is (16 x 16 x 512)
			e5 = lrelu(self.en_e5(conv2d(e4, self.dim*8, name='e5_conv')))
			# e5 is (8 x 8 x 512)
			e6 = lrelu(self.en_e6(conv2d(e5, self.dim*8, name='e6_conv')))
			# e6 is (4 x 4 x 512)
			e7 = lrelu(conv2d(e6, self.dim*8, name='e7_conv'))
			# e7 is (2 x 2 x 512)
		
			return e7, e6, e5, e4, e3, e2, e1, e0
	

	def Autodecoder(self, feature, e6, e5, e4, e3, e2, e1, e0, reuse=False):

		with tf.variable_scope("Autodecoder") as scope:
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False

			s = self.output_size
			s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

			#decoder
			d6 = lrelu(self.de_d6(deconv2d(feature, [self.batch_size, s64, s64, self.dim*8], name='d6_deconv')))
			# d6 is (4 x 4 x 512)
			d5 = lrelu(self.de_d5(deconv2d(tf.concat([e6,d6],3), [self.batch_size, s32, s32, self.dim*8], name='d5_deconv')))
			# d5 is (8 x 8 x 512)
			d4 = lrelu(self.de_d4(deconv2d(tf.concat([e5,d5],3), [self.batch_size, s16, s16, self.dim*8], name='d4_deconv')))
			# d4 is (16 x 16 x 512)
			d3 = lrelu(self.de_d3(deconv2d(tf.concat([e4,d4],3), [self.batch_size, s8, s8, self.dim*4], name='d3_deconv')))
			# d3 is (32 x 32 x 256)
			d2 = lrelu(self.de_d2(deconv2d(tf.concat([e3,d3],3), [self.batch_size, s4, s4, self.dim*2], name='d2_deconv')))
			# d2 is (64 x 64 x 128)
			d1 = lrelu(self.de_d1(deconv2d(tf.concat([e2,d2],3), [self.batch_size, s2, s2, self.dim], name='d1_deconv')))
			# d1 is (128 x 128 x 64)
			d0 = lrelu(self.de_d0(deconv2d(tf.concat([e1,d1],3), [self.batch_size, s, s, self.dim], name='d0_deconv')))
			# d0 is (256 x 256 x 64)
			output = lrelu(deconv2d(d0, [self.batch_size, s, s, self.output_c_dim], d_h=1, d_w=1, name='output_deconv'))
			# output is (256 x 256 x 3)
			return output


	def imread(self, path, is_grayscale = False):
		if (is_grayscale):
			return scipy.misc.imread(path, flatten = True).astype(np.float)
		else:
			return scipy.misc.imread(path).astype(np.float)

	def load_data(self, path, is_grayscale=False):
		img = imread(path, is_grayscale)
		img = scipy.misc.imresize(img, [self.fine_size, self.fine_size])
		img = img/127.5 - 1.
		return img

	def train(self):
		train_encoder = tf.train.AdamOptimizer(self.learning_rate).minimize(self.encoder_loss)
		train_feature = tf.train.AdamOptimizer(self.learning_rate).minimize(self.feature_loss, var_list = self.feature_vars)
		train_total = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)        
		
		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		counter = 1
		start_time = time.time()

		for epo in xrange(self.epoch):
			data_G = glob('./datasets/{}/gray/train/*.jpg'.format(self.dataset))
			data_G.sort()
			data_C = glob('./datasets/{}/color/train/*.jpg'.format(self.dataset))
			data_C.sort()
			batch_idxs = len(data_G) // self.batch_size
			for idx in xrange(batch_idxs):
				a = np.random.randint(low=0, high=batch_idxs)
				batch_files_G = data_G[a*self.batch_size:(a+1)*self.batch_size]
				batch_files_C = data_C[a*self.batch_size:(a+1)*self.batch_size]
				batch_G = [self.load_data(batch_file) for batch_file in batch_files_G]
				batch_C = [self.load_data(batch_file) for batch_file in batch_files_C]
				
				if (self.is_grayscale):
					batch_images_G = np.array(batch_G).astype(np.float32)[:, :, :, None]
					batch_images_C = np.array(batch_C).astype(np.float32)
				else:
					batch_images_G = np.array(batch_G).astype(np.float32)
					batch_images_C = np.array(batch_C).astype(np.float32)

				if self.train_type == 'e_loss':
					_ , e_loss = self.sess.run([train_encoder, self.encoder_loss] , feed_dict={self.image_G: batch_images_G, self.image_C: batch_images_C})
				elif self.train_type == 'f_loss':
					_ , f_loss = self.sess.run([train_feature, self.feature_loss] , feed_dict={self.image_G: batch_images_G, self.image_C: batch_images_C})
				else:
					_ , total_loss = self.sess.run([train_total, self.total_loss] , feed_dict={self.image_G: batch_images_G, self.image_C: batch_images_C})

				counter += 1
				if np.mod(counter, 100) == 2:

					self.save(self.checkpoint_dir, counter)


				if self.train_type == 'e_loss':
					print("Epoch: [%2d] [%4d/%4d] time: %4.4f, encoder_loss: %.8f" \
					% (epo, idx, batch_idxs,
						time.time() - start_time, e_loss))
				elif self.train_type == 'f_loss':
					print("Epoch: [%2d] [%4d/%4d] time: %4.4f, feature_loss: %.8f" \
					% (epo, idx, batch_idxs,
						time.time() - start_time, f_loss))
				else:
					print("Epoch: [%2d] [%4d/%4d] time: %4.4f, total_loss: %.8f" \
					% (epo, idx, batch_idxs,
						time.time() - start_time, total_loss))
				
			if epo % 3 ==1:
				a = np.random.randint(low=0, high=batch_idxs)
				batch_files_G = data_G[a*self.batch_size:(a+1)*self.batch_size]
				batch_G = [self.load_data(batch_file) for batch_file in batch_files_G]
				batch_files_C = data_C[a*self.batch_size:(a+1)*self.batch_size]
				batch_C = [self.load_data(batch_file) for batch_file in batch_files_C]

				if (self.is_grayscale):
					batch_images_G = np.array(batch_G).astype(np.float32)[:, :, :, None]
					batch_images_C = np.array(batch_C).astype(np.float32)
				else:
					batch_images_G = np.array(batch_G).astype(np.float32)
					batch_images_C = np.array(batch_C).astype(np.float32)

				decoder_imgs_G, decoder_imgs_C = self.sess.run([self.image_G_, self.image_C_], feed_dict={self.image_G: batch_images_G, self.image_C: batch_images_C})
				
				rand = np.random.randint(low=0, high=len(decoder_imgs_G))
				if self.train_type == 'e_loss':
					for i,decoder_img_G in enumerate(decoder_imgs_G):
						if i==rand:
							scipy.misc.imsave('./datasets/{}/test/test_eG/test_G_{:04d}.jpg'.format(self.dataset, epo), decoder_img_G)

					for i,decoder_img_C in enumerate(decoder_imgs_C):
						if i==rand:
							scipy.misc.imsave('./datasets/{}/test/test_eC/test_C_{:04d}.jpg'.format(self.dataset, epo), decoder_img_C)

				elif self.train_type == 'f_loss':
					for i,decoder_img_G in enumerate(decoder_imgs_G):
						if i==rand:
							scipy.misc.imsave('./datasets/{}/test/test_fG/test_G_{:04d}.jpg'.format(self.dataset, epo), decoder_img_G)

					for i,decoder_img_C in enumerate(decoder_imgs_C):
						if i==rand:
							scipy.misc.imsave('./datasets/{}/test/test_fC/test_C_{:04d}.jpg'.format(self.dataset, epo), decoder_img_C)

				else:
					for i,decoder_img_G in enumerate(decoder_imgs_G):
						if i==rand:
							scipy.misc.imsave('./datasets/{}/test/test_tG/test_G_{:04d}.jpg'.format(self.dataset, epo), decoder_img_G)

					for i,decoder_img_C in enumerate(decoder_imgs_C):
						if i==rand:
							scipy.misc.imsave('./datasets/{}/test/test_tC/test_C_{:04d}.jpg'.format(self.dataset, epo), decoder_img_C)
	

	def save(self, checkpoint_dir, step):
		model_name = "Feature_colorization.model"
		model_dir = "%s_%s" % (self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
			os.path.join(checkpoint_dir, model_name),
 			global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoint...")

		model_dir = "%s_%s" % (self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False
	'''
	def test(self):
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
  
		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		start_time = time.time()

		data_G_test = glob('./datasets/{}/gray/test/*.jpg'.format(self.dataset))

		idxs = len(data_G_test) // self.batch_size
		for idx in range(idxs):
			batch_files_G = data_G_test[idx*self.batch_size:(idx+1)*self.batch_size]
			batch_G_test = [self.load_data(batch_file) for batch_file in batch_files_G]
			
		
			if (self.is_grayscale):
				batch_images_G = np.array(batch_G_test).astype(np.float32)[:, :, :, None]
			else:
				batch_images_G = np.array(batch_G_test).astype(np.float32)
			
			decoder_imgs_G = self.sess.run(self.image_G_, feed_dict={self.image_G: batch_images_G})
			for i,decoder_img_G in enumerate(decoder_imgs_G):
				scipy.misc.imsave('./datasets/{}/test/test_final/test_{:04d}_{:04d}.jpg'.format(self.dataset, idx, i), decoder_img_G)
	'''
	def test(self):
		self.image_G = tf.placeholder(tf.float32, [self.batch_size, self.fine_size, self.fine_size, self.input_c_dim])
		self.image_C = tf.placeholder(tf.float32, [self.batch_size, self.fine_size, self.fine_size, self.input_c_dim])

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
  
		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		start_time = time.time()
		
		data_G_test = glob('./datasets/{}/gray/test/*.jpg'.format(self.dataset))
		data_C_test = glob('./datasets/{}/color/test/*.jpg'.format(self.dataset))

		idxs = len(data_G_test) // (self.batch_size)
		for idx in range(idxs):
			batch_files_G = data_G_test[idx*self.batch_size:(idx+1)*self.batch_size]
			batch_files_C = data_C_test[idx*self.batch_size:(idx+1)*self.batch_size]
			batch_G_test = [self.load_data(batch_file) for batch_file in batch_files_G]
			batch_C_test = [self.load_data(batch_file) for batch_file in batch_files_C]

			if (self.is_grayscale):
				batch_images_G = np.array(batch_G_test).astype(np.float32)[:, :, :, None]
				batch_images_C = np.array(batch_C_test).astype(np.float32)
			else:
				batch_images_G = np.array(batch_G_test).astype(np.float32)
				batch_images_C = np.array(batch_C_test).astype(np.float32)
	
			decoder_imgs_G, decoder_imgs_C, encoder_loss, feature_loss = self.sess.run([self.image_G_, self.image_C_, self.encoder_loss, self.feature_loss], feed_dict={self.image_G: batch_images_G, self.image_C: batch_images_C})
			print("encoder_loss: %.8f, feature_loss :%.8f"% (encoder_loss, feature_loss))
			for i,decoder_img_G in enumerate(decoder_imgs_G):
				scipy.misc.imsave('./datasets/{}/test/test_final/testG_{:04d}_{:04d}.jpg'.format(self.dataset, idx, i), decoder_img_G)
			for i,decoder_img_C in enumerate(decoder_imgs_C):
				scipy.misc.imsave('./datasets/{}/test/test_final/testC_{:04d}_{:04d}.jpg'.format(self.dataset, idx, i), decoder_img_C)

			

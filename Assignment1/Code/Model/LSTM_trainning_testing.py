
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf


import reader_LSTM




flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "medium",
    "model type")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS



class Model(object):


  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      f = open('new_glove_3.txt', 'r')
      words = []
      vectors = []
      idx = -1
      count =0
      word_dict = {}
      word_dict_reverse = {}
      for l in f:
        idx += 1
        line = l
        strings = line.split()
        if len(strings)==301:
          word = strings[0]
          vector = [np.float32(num) for num in strings[1:]]
          if len(vector)==300:
            word_dict[word] = idx
            word_dict_reverse[idx] = word
            
            if (idx%10000) == 0:
              print ('Processed words: ', idx)
            words.append(word)
            vector = np.array(vector).reshape((1,-1))
            vectors.append(vector)
        else :
          count+=1
      f.close()
      
      matrix = np.vstack(tuple(vectors))
      W1 = tf.Variable(matrix,name="W1")
      
      inputs = tf.nn.embedding_lookup(W1, self.input_data)
      
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.5
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 300
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 12935


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 2.2
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 30
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 109510 +10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 100
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000 +109510


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 300
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 5764




def run_epoch(session, m, data, eval_op,str2, verbose=False):
  
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  
  start_time = time.time()
  costs = 0.0
  iters = 0
  t1 = 'test'
  
  state  = session.run(m.initial_state)
  
  
  for step, (x, y) in enumerate(reader.iterator(data, m.batch_size,
                                                    m.num_steps)):
    
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost
    
    iters += m.num_steps
    
    
    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))
    
  return np.exp(costs / iters)





def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
	

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to data directory")

  raw_data = reader.model_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data1, test_data2, test_data3, test_data4, test_data5, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    
    print(type(initializer))
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = Model(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = Model(is_training=False, config=config)
      mtest = Model(is_training=False, config=eval_config)
      mtest1 = Model(is_training=False, config=eval_config)
      mtest2 = Model(is_training=False, config=eval_config)
      mtest3 = Model(is_training=False, config=eval_config)
      mtest4 = Model(is_training=False, config=eval_config)



    tf.initialize_all_variables().run()

    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i-10 - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      
      train_perplexity = run_epoch(session, m, train_data, m.train_op,'train',
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      


    with open("MSR_Sentence_Completion_Challenge_V1/Data/perplexities.txt", "w") as p :   
    	with open("MSR_Sentence_Completion_Challenge_V1/Data/answer.txt", "w") as answer :
    		with open("MSR_Sentence_Completion_Challenge_V1/Data/questions.txt") as f:
    			i = 0
    			k = 1
    			low = 999999999999.0
    			for line in f:
    				#i = i+1
    				if i>=(5195):
    					break
    				#line = line.split(' ', 1)[1]
    				raw_data = reader.model_raw_data(FLAGS.data_path)
    				train_data, valid_data, test_data1, test_data2, test_data3, test_data4, test_data5, _ = raw_data
    				
    				if i%5==0 :
    					f1 = open("MSR_Sentence_Completion_Challenge_V1/Data/test1.txt", "w+")
    						
    					f1.write(line)
    					
    					f1.close()
    					test_perplexity1 = run_epoch(session, mtest, test_data1, tf.no_op(),'test')
    					p.write(str(test_perplexity1)+'  ')
    					
    					low = test_perplexity1
    					
    					ans = 'a'
    				elif i%5==1:
    					f2 = open("MSR_Sentence_Completion_Challenge_V1/Data/test2.txt", "w+")
    						
    					f2.write(line)
    					
    					f2.close()
    					test_perplexity2 = run_epoch(session, mtest, test_data2, tf.no_op(),'test')
    					p.write(str(test_perplexity2)+'  ')
    					
    					if test_perplexity2<low :
    					
    						low = test_perplexity2
    						ans ='b'
    				elif i%5==2:
    					f3 = open("MSR_Sentence_Completion_Challenge_V1/Data/test3.txt", "w+")
    						
    					f3.write(line)
    					
    					f3.close()
    					test_perplexity3 = run_epoch(session, mtest, test_data3, tf.no_op(),'test')
    					p.write(str(test_perplexity3)+'  ')
    					
    					if test_perplexity3<low :
    					
    						low = test_perplexity3
    						ans = 'c'
    				elif i%5==3:
    					f4 = open("MSR_Sentence_Completion_Challenge_V1/Data/test4.txt", "w+")
    						
    					f4.write(line)
    					
    					f4.close()
    					test_perplexity4 = run_epoch(session, mtest, test_data4, tf.no_op(),'test')
    					p.write(str(test_perplexity4)+'  ')
    					
    					if test_perplexity4<low :
    					
    						low = test_perplexity4
    						ans = 'd'
    				elif i%5==4:
    					f5 = open("MSR_Sentence_Completion_Challenge_V1/Data/test5.txt", "w+")
    						
    					f5.write(line)
    					
    					f5.close()
    					test_perplexity5 = run_epoch(session, mtest, test_data5, tf.no_op(),'test')
    					p.write(str(test_perplexity5)+'  ')
    					
    					p.write('\n')
    					if test_perplexity5<low :
    					
    						low = test_perplexity5
    						ans = 'e'
    					answer.write(str(k))
    					answer.write(str(ans)+')')
    					answer.write('\n')
    					low = 999999999999.0
    					k=k+1
    					print(k)
    				i=i+1
	

if __name__ == "__main__":
  tf.app.run()

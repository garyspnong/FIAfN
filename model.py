import os
import numpy as np
import tensorflow as tf
from time import time
from sklearn.metrics import roc_auc_score, log_loss
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


def normalize(inputs, epsilon=1e-8):
    '''
    Applies layer normalization
    Args:
        inputs: A tensor with 2 or more dimensions
        epsilon: A floating number to prevent Zero Division
    Returns:
        A tensor with the same shape and data dtype
    '''
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta

    return outputs

        
def AFT_simple(queries,
               keys,
               values,
               num_units=None,
               num_heads=1,
               dropout_keep_prob=1,
               is_training=True,
               has_residual=True):
	
    if num_units is None:
        num_units = queries.get_shape().as_list[-1]

    # Linear projections
    Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
    K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
    V = tf.layers.dense(values, num_units, activation=tf.nn.relu)
    if has_residual:
        V_res = tf.layers.dense(values, num_units, activation=tf.nn.relu)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

    # Multiplication
    weights = tf.multiply(K_, V_)

    # Scale
    weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)

    # Activation
    weights = tf.nn.softmax(weights)
    Q_ = tf.sigmoid(Q_)

    # Dropouts
    weights = tf.layers.dropout(weights, rate=1-dropout_keep_prob,
                                        training=tf.convert_to_tensor(is_training))

    # Weighted sum
    outputs = tf.multiply(weights, Q_)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

    # Residual connection
    if has_residual:
        outputs += V_res

    outputs = tf.nn.relu(outputs)
    # Normalize
    outputs = normalize(outputs)
        
    return outputs


class FIAfN():
    def __init__(self, args, feature_size, run_cnt):

        self.feature_size = feature_size        
        self.field_size = args.field_size            # denote as m, number of total feature fields
        self.embedding_size = args.embedding_size    # denote as d, size of the feature embedding
        self.blocks = args.blocks                    # number of the blocks
        self.heads = args.heads                      # number of the heads
        self.block_shape = args.block_shape
        self.output_size = args.block_shape[-1] 
        self.has_residual = args.has_residual
        self.deep_layers = args.deep_layers          # whether to joint train with deep networks 


        self.lstm_unit_num = 16                      #unit num of dynamic_rnn
        self.movie_title_max = 5216
        self.batch_norm = args.batch_norm
        self.batch_norm_decay = args.batch_norm_decay
        self.drop_keep_prob = args.dropout_keep_prob
        self.l2_reg = args.l2_reg
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.optimizer_type = args.optimizer_type

        self.save_path = args.save_path + str(run_cnt) + '/'
        self.is_save = args.is_save
        if (args.is_save == True and os.path.exists(self.save_path) == False):
            os.makedirs(self.save_path)	

        self.verbose = args.verbose
        self.random_seed = args.random_seed
        self.loss_type = args.loss_type
        self.eval_metric = roc_auc_score
        self.best_loss = 1.0
        self.greater_is_better = args.greater_is_better
        self.train_result, self.valid_result = [], []
        self.train_loss, self.valid_loss = [], []
        
        self._init_graph()
        
    def movie_title_lstm_layer(self, title_index, title_length, dropout_keep_prob):
        with tf.variable_scope('movie_title_embed'):
            movie_title_embed_matrix = tf.get_variable('title_embed_matrix', [self.movie_title_max , self.embedding_size],
                                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, title_index,
                                                             name='title_lookup')
    
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_unit_num, forget_bias=0.0)
    
        with tf.name_scope("movie_title_dropout"):
            lstm_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
    
            # 根据输入动态决定对应的batch_size大小
            batch_size_ = tf.shape(title_index)[0]
            init_state = lstm_cell_dropout.zero_state(batch_size_, dtype=tf.float32)
    
        # 步长根据标题长度动态变化，dynamic_rnn会将填充长度输出置为0
        
        
        lstm_output, final_state = tf.nn.dynamic_rnn(lstm_cell_dropout,
                                                     movie_title_embed_layer,
                                                     sequence_length=title_length,
                                                     initial_state=init_state,
                                                     scope='movie_title_rnn')
        # 根据标题长度计算平均值，除数是标题的真实长度
        with tf.name_scope('movie_title_avg_pool'):
            lstm_output = tf.reduce_sum(lstm_output, 1) / title_length[:, None]
    
        return lstm_output,final_state
        
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            # placeholder for single-value field.
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_index")  # None * 6
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                                 name="feat_value")  # None * 6

            # placeholder for multi-value field. (movielens dataset genre field)
            self.genre_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="genre_index") # None * 6
            self.genre_value = tf.placeholder(tf.float32, shape=[None, None],
                                                 name="genre_value") # None * 6
               
            # placeholder for multi-value field. (movielens dataset title field)
            self.title_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="title_index") # None * 15
            self.title_length = tf.placeholder(tf.float32, shape=[None],
                                                 name="title_value") # None * 1
            self.embeddings_t,self.state = self.movie_title_lstm_layer(self.title_index, self.title_length, 0.5)
            
            
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1

            # In our implementation, the shape of dropout_keep_prob is [3], used in 3 different places.
            self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_prob")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                             self.feat_index)  # None * 6 * d
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size-2, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)      # None * 6 * d

            # for multi-value field
            self.embeddings_m = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                             self.genre_index) # None * 6 * d
            genre_value = tf.reshape(self.genre_value, shape=[-1, 6, 1])
            self.embeddings_m  = tf.multiply(self.embeddings_m, genre_value)
            self.embeddings_m = tf.reduce_sum(self.embeddings_m, axis=1) # None * d 
            self.embeddings_m = tf.div(self.embeddings_m, tf.reduce_sum(self.genre_value, axis=1, keep_dims=True)) #求平均 None * d
            
            #concatenate single-value field with multi-value field
            self.embeddings = tf.concat([self.embeddings, tf.expand_dims(self.embeddings_m, 1)], 1) # None * 7 * d
            self.embeddings = tf.concat([self.embeddings, tf.expand_dims(self.embeddings_t, 1)], 1) # None * 8 * d
            self.embeddings = tf.nn.dropout(self.embeddings, self.dropout_keep_prob[1]) # None * 8 * d

            # ---------- main part of FIAfN-------------------
            self.y_deep = self.embeddings # None * 8 * d
            
            for i in range(self.blocks):   
                self.y_deep = AFT_simple(queries=self.y_deep,
                                                  keys=self.y_deep,
                                                  values=self.y_deep,
                                                  num_units=self.block_shape[i],
                                                  num_heads=self.heads,
                                                  dropout_keep_prob=self.dropout_keep_prob[0],
                                                  is_training=self.train_phase,
                                                  has_residual=self.has_residual)
            
            self.flat = tf.reshape(self.y_deep, 
                                   shape=[-1, self.output_size * self.field_size]) 

            
            self.out = tf.add(tf.matmul(self.flat, self.weights["prediction"]), 
                              self.weights["prediction_bias"], name='logits')  # None * 1
            
            # joint training with feedforward nn
            if self.deep_layers != None:
                self.y_dense = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
                for i in range(0, len(self.deep_layers)):
                    self.y_dense = tf.add(tf.matmul(self.y_dense, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i]
                    if self.batch_norm:
                        self.y_dense = self.batch_norm_layer(self.y_dense, train_phase=self.train_phase, scope_bn="bn_%d" %i)
                    self.y_dense = tf.nn.relu(self.y_dense)
                    self.y_dense = tf.nn.dropout(self.y_dense, self.dropout_keep_prob[2])
                self.y_dense = tf.add(tf.matmul(self.y_dense, self.weights["prediction_dense"]),
                                      self.weights["prediction_bias_dense"], name='logits_dense')  # None * 1
                self.out += self.y_dense
        
            # ---------- Compute the loss ----------
            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out, name='pred')
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # l2 regularization on weights
            if self.l2_reg > 0:
                if self.deep_layers != None:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                                                    self.l2_reg)(self.weights["layer_%d"%i])
         
           
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.var1 = [v for v in tf.trainable_variables() if v.name != 'feature_bias:0']
            self.var2 = [tf.trainable_variables()[1]]    # self.var2 = [feature_bias]

            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                    beta1=0.9, beta2=0.999, epsilon=1e-8).\
                                                    minimize(self.loss, global_step=self.global_step)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).\
                                                           minimize(self.loss, global_step=self.global_step)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).\
                                                                   minimize(self.loss, global_step=self.global_step)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).\
                                                            minimize(self.loss, global_step=self.global_step)

            # init
            self.saver = tf.train.Saver(max_to_keep=5)
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

        

    def _init_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)


    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size(n) * d
  
        input_size = self.output_size * self.field_size

        # FNN layers
        if self.deep_layers != None:
            num_layer = len(self.deep_layers)
            layer0_size = self.field_size * self.embedding_size
            glorot = np.sqrt(2.0 / (layer0_size + self.deep_layers[0]))
            weights["layer_0"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(layer0_size, self.deep_layers[0])), dtype=np.float32)
            weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                            dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
                weights["layer_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                    dtype=np.float32)  # layers[i-1] * layers[i]
                weights["bias_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                    dtype=np.float32)  # 1 * layer[i]
            glorot = np.sqrt(2.0 / (self.deep_layers[-1] + 1))
            weights["prediction_dense"] = tf.Variable(
                                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[-1], 1)),
                                dtype=np.float32, name="prediction_dense")
            weights["prediction_bias_dense"] = tf.Variable(
                                np.random.normal(), dtype=np.float32, name="prediction_bias_dense")


        #---------- prediciton weight ------------------#                                
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["prediction"] = tf.Variable(
                            np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                            dtype=np.float32, name="prediction")
        weights["prediction_bias"] = tf.Variable(
                            np.random.normal(), dtype=np.float32, name="prediction_bias")

        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    
    def get_batch(self, Xi, Xv, Xi_title, Xv_title, Xi_genre, Xv_genre, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], Xi_title[start:end], Xv_title[start:end], Xi_genre[start:end], Xv_genre[start:end], [[y_] for y_ in y[start:end]]


    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c, d, e, f, g):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(e)   
        np.random.set_state(rng_state)
        np.random.shuffle(f)  
        np.random.set_state(rng_state)
        np.random.shuffle(g)               


    def fit_on_batch(self, Xi, Xv, Xi_title, Xv_title, Xi_genre, Xv_genre, y):
        title_lenth=np.sum(Xv_title, axis=1)
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.title_index: Xi_title,
                     self.title_length: title_lenth,
                     self.genre_index: Xi_genre,
                     self.genre_value: Xv_genre,
                     self.label: y,
                     self.dropout_keep_prob: self.drop_keep_prob,
                     self.train_phase: True}
        step, loss, opt = self.sess.run((self.global_step, self.loss, self.optimizer), feed_dict=feed_dict)
        return step, loss

    # Since the train data is very large, they can not be fit into the memory at the same time.
    # We separate the whole train data into several files and call "fit_once" for each file.
    def fit_once(self, Xi_train, Xv_train, Xi_train_title,Xv_train_title, Xi_train_genre, Xv_train_genre, y_train,
                 epoch, Xi_valid=None, 
	             Xv_valid=None, Xi_valid_title=None,Xv_valid_title=None,Xi_valid_genre=None, Xv_valid_genre=None, y_valid=None,
                 early_stopping=False):
        
        has_valid = Xv_valid is not None
        last_step = 0
        t1 = time()
        self.shuffle_in_unison_scary(Xi_train, Xv_train, Xi_train_title,Xv_train_title,Xi_train_genre, Xv_train_genre, y_train)
        total_batch = int(len(y_train) / self.batch_size)
        for i in range(total_batch):
            Xi_batch, Xv_batch, Xi_batch_title, Xv_batch_title, Xi_batch_genre, Xv_batch_genre, y_batch = self.get_batch(Xi_train, Xv_train, Xi_train_title, Xv_train_title, Xi_train_genre, Xv_train_genre, y_train, self.batch_size, i)
            step, loss = self.fit_on_batch(Xi_batch, Xv_batch, Xi_batch_title, Xv_batch_title, Xi_batch_genre, Xv_batch_genre, y_batch)
            last_step = step
        # evaluate training and validation datasets
        train_result, train_loss = self.evaluate(Xi_train, Xv_train, Xi_train_title,Xv_train_title, Xi_train_genre, Xv_train_genre, y_train)
        self.train_result.append(train_result)
        self.train_loss.append(train_loss)
        if has_valid:
            valid_result, valid_loss = self.evaluate(Xi_valid, Xv_valid, Xi_valid_title, Xv_valid_title, Xi_valid_genre, Xv_valid_genre, y_valid)
            self.valid_result.append(valid_result)
            self.valid_loss.append(valid_loss)
            if valid_loss < self.best_loss and self.is_save == True:
                old_loss = self.best_loss
                self.best_loss = valid_loss
                self.saver.save(self.sess, self.save_path + 'model.ckpt',global_step=last_step)
                print("[%d] model saved!. Valid loss is improved from %.4f to %.4f" 
                      % (epoch, old_loss, self.best_loss))

        if self.verbose > 0 and ((epoch-1)*9) % self.verbose == 0:
            if has_valid:
                print("[%d] train-result=%.4f, train-logloss=%.4f, valid-result=%.4f, valid-logloss=%.4f [%.1f s]" % (epoch, train_result, train_loss, valid_result, valid_loss, time() - t1))
            else:
                print("[%d] train-result=%.4f [%.1f s]" \
                    % (epoch, train_result, time() - t1))
        if has_valid and early_stopping and self.training_termination(self.valid_loss):
            return False
        else:
            return True



    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False


    def predict(self, Xi, Xv, Xi_title, Xv_title, Xi_genre, Xv_genre):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, Xi_batch_title, Xv_batch_title, Xi_batch_genre, Xv_batch_genre, y_batch = self.get_batch(Xi, Xv,Xi_title, Xv_title, Xi_genre, Xv_genre, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            title_lenth=np.sum(Xv_batch_title, axis=1)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.title_index: Xi_batch_title,
                         self.title_length: title_lenth,
                         self.genre_index: Xi_batch_genre,
                         self.genre_value: Xv_batch_genre,
                         self.label: y_batch,
                         self.dropout_keep_prob: [1.0] * len(self.drop_keep_prob),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, Xi_batch_title, Xv_batch_title, Xi_batch_genre, Xv_batch_genre, y_batch = self.get_batch(Xi, Xv, Xi_title, Xv_title, Xi_genre, Xv_genre, dummy_y, self.batch_size, batch_index)

        return y_pred


    def evaluate(self, Xi, Xv, Xi_title, Xv_title, Xi_genre, Xv_genre, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv, Xi_title, Xv_title, Xi_genre, Xv_genre)
        y_pred = np.clip(y_pred,1e-6,1-1e-6)
        return self.eval_metric(y, y_pred), log_loss(y, y_pred)


    def restore(self, save_path=None):
        if (save_path == None):
            save_path = self.save_path
        ckpt = tf.train.get_checkpoint_state(save_path)  
        if ckpt and ckpt.model_checkpoint_path:  
            self.saver.restore(self.sess, ckpt.model_checkpoint_path) 
            if self.verbose > 0:
                print ("restored from %s" % (save_path))
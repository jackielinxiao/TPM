import numpy as np
import pandas as pd
import hashlib
import math
import random
import tensorflow as tf

from model_zoo import tree_model_fasttest
from utils import label_encoding, get_label_encoding_loss, get_encoded_playtime
from metrics import xauc_score

# params
var_w = 1.0
mse_weight = 0.2
wr_bucknum = 16


# setting
learning_rate = 1e-3
epochs = 1
batch_size = 512
embdim = 16
feat_num = 5000000

# data prepare
print("loading dataset")
train_feats_selected = np.load("train_clip.npy")
np.random.shuffle(train_feats_selected)

# get percentile of play_time
play_time = train_feats_selected[:,-3]
percen_value = np.percentile(play_time, np.linspace(0.0, 100.0, num=wr_bucknum+1).astype(np.float32)).tolist()
peercentiles_by_duration = percen_value

begins_tensor = tf.constant(peercentiles_by_duration[:-1], shape=[1, wr_bucknum])
ends_tensor = tf.constant(peercentiles_by_duration[1:], shape=[1, wr_bucknum])

# loading test
test_feats_selected = np.load("test_clip.npy")

# hash feature
def get_md5_featID(feat_str):
  md5 = hashlib.md5()
  md5.update(feat_str.encode("UTF-8"))
  return int(md5.hexdigest(),16) % feat_num

# return feats and label
def data_func(apd, is_train):
  i = 0
  while True:
    if is_train and i>=apd.shape[0]:
      i = i % apd.shape[0]
    if not is_train and i>=apd.shape[0]:
      break
    alist = apd[i,:].tolist()
    inputs_fea = [get_md5_featID("query_%d" % i) for i in alist[:10]]
    inputs_fea += [get_md5_featID("item_%d" % i) for i in alist[10:20]]
    yield inputs_fea[:20], alist[20:30], alist[30:40], [apd[i,-3]]
    i = i + 1


sess = tf.Session()
# train dataset
dataset = tf.data.Dataset.from_generator(data_func, args=[train_feats_selected, True], output_types=(tf.int32,tf.float32,tf.float32,tf.float32), output_shapes = (tf.TensorShape([20]), tf.TensorShape([10]), tf.TensorShape([10]), tf.TensorShape([1])), )
dataset = dataset.batch(batch_size).prefetch(batch_size)

# test dataset
dataset_test = tf.data.Dataset.from_generator(data_func, args=[test_feats_selected, False], output_types=(tf.int32,tf.float32,tf.float32,tf.float32), output_shapes = (tf.TensorShape([20]), tf.TensorShape([10]), tf.TensorShape([10]), tf.TensorShape([1])), )
dataset_test = dataset_test.batch(batch_size).prefetch(batch_size)

# Create an iterator over the dataset
iterator = dataset.make_initializable_iterator()
iterator_test = dataset_test.make_initializable_iterator()
sess.run(iterator.initializer)
sess.run(iterator_test.initializer)

X,qmsk,imsk,Ytime = iterator.get_next()
X_test,qmsk_test, imsk_test, Ytime_test= iterator_test.get_next()

# build model graph
pred_train = tree_model_fasttest(X, class_num=wr_bucknum-1, dropout=0.2, field_num=20, feat_num=feat_num, feat_dim=embdim, qmsk=qmsk,imsk=imsk,reuse=False, is_training=True)
pred_test = tree_model_fasttest(X_test, class_num=wr_bucknum-1, dropout=0.2, field_num=20, feat_num=feat_num, feat_dim=embdim, qmsk=qmsk_test,imsk=imsk_test, reuse=True, is_training=False)

# training : get pred time and variance
encoded_playtime, var = get_encoded_playtime(pred_train, wr_bucknum, begins_tensor, ends_tensor)

# testing : get pred time and variance
encoded_playtime_test, var_test = get_encoded_playtime(pred_test, wr_bucknum, begins_tensor, ends_tensor)

# get tree-like loss
label_dict, weight_dict = label_encoding(wr_bucknum, Ytime, begins_tensor, ends_tensor)
loss_op = get_label_encoding_loss(label_dict, weight_dict, pred_train, wr_bucknum)

# get mse loss
loss_op_mse = tf.reduce_sum(
  tf.square(Ytime-encoded_playtime)
)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op+loss_op_mse*mse_weight+var*var_w)

init = tf.global_variables_initializer()
sess.run(init)

# Training cycle
num_steps = int(epochs * train_feats_selected.shape[0] / batch_size )
for step in range(1, num_steps + 1):
    # Run optimization
    sess.run(train_op)
        
    if step % 100 == 0 or step == 1:
        var111, loss, lossmse, ptime , ytime= sess.run([var, loss_op,loss_op_mse,encoded_playtime, Ytime])
        print("Step " + str(step) + ", Minibatch TimeLoss= " + str(loss) + ", Minibatch MseLoss= " + str(lossmse) + "var: "+ str(var111))
        print(ytime[:5],ptime[:5])

print("Optimization Finished!")

###############################################testing###############
# Testing
# rnd Testing
all_pred_list = []
all_ground_list = []
all_dur_list = []
for step in range(1, int(test_feats_selected.shape[0] / batch_size )+1):
  ptime_test, yy = sess.run([encoded_playtime_test, Ytime_test])
  all_pred_list.append(ptime_test)
  all_ground_list.append(yy)
all_pred = np.concatenate(all_pred_list, axis=0).reshape(-1)
all_yy = np.concatenate(all_ground_list, axis=0).reshape(-1)
print("time info is ",all_pred[:5], all_yy[:5])
xxx = round(xauc_score(all_yy,all_pred), 4)
print("time XAUC is ", xxx)
mmm = round(np.mean(np.fabs(all_pred-all_yy)), 4)
print("time MAE is ", mmm)

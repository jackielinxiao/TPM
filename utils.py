import math
import tensorflow as tf

def label_encoding(tree_num_intervals,cmp_ratio, begins, ends, name="label_encoding"):
  label_dict = {}
  weight_dict = {}
  height = int(math.log2(tree_num_intervals))
  with tf.name_scope(name):
    for i in range(height):
      for j in range(2**i):
        temp_ind = max(int(tree_num_intervals*1.0/(2**i)*j)-1,0)
        if j==0:
          weight_temp = tf.where(tf.less(cmp_ratio, tf.reshape(begins[:, temp_ind], [-1,1])), tf.zeros_like(cmp_ratio), tf.ones_like(cmp_ratio))
        else:
          weight_temp = tf.where(tf.less(cmp_ratio, tf.reshape(ends[:, temp_ind], [-1,1])), tf.zeros_like(cmp_ratio), tf.ones_like(cmp_ratio))
        temp_ind = max(int(tree_num_intervals*1.0/(2**i)*(j+1))-1,0)
        weight_temp = tf.where(tf.less(cmp_ratio, tf.reshape(ends[:, temp_ind], [-1,1])), weight_temp, tf.zeros_like(cmp_ratio))
        temp_ind = max(int(tree_num_intervals*(1.0/(2**i)*j+1.0/(2**(i+1))))-1,0)
        label_temp = tf.where(tf.less(cmp_ratio, tf.reshape(ends[:, temp_ind], [-1,1])), tf.zeros_like(cmp_ratio), tf.ones_like(cmp_ratio))
        label_dict[1000*i+j] = label_temp
        weight_dict[1000*i+j] = weight_temp
  return label_dict, weight_dict


def get_label_encoding_loss(label_dict, weight_dict, label_encoding_predict, tree_num_intervals=32):
  auxiliary_loss_  = 0.0
  height = int(math.log2(tree_num_intervals))
  for i in range(height):
      for j in range(2**i):
        interval_label = label_dict[1000*i+j]
        interval_weight = weight_dict[1000*i+j]
        interval_preds = tf.reshape(label_encoding_predict[:, 2**i-1+j], [-1, 1])
        interval_loss = tf.losses.log_loss(labels=interval_label,
                                         predictions=interval_preds,
                                         weights=interval_weight,
                                         reduction=tf.losses.Reduction.SUM)
        auxiliary_loss_ = auxiliary_loss_ + interval_loss
  return auxiliary_loss_ / (tree_num_intervals - 1.0)


def get_encoded_playtime(label_encoding_predict, tree_num_intervals, begins, ends, name="encoded_playtime"):
  height = int(math.log2(tree_num_intervals))
  encoded_prob_list = []
  temp_encoded_playtime = (begins + ends) / 2.0  # bsx32
  encoded_playtime = temp_encoded_playtime
  
  for i in range(tree_num_intervals):
      temp = 0.0
      cur_code = 2**height - 1 + i
      for j in range(1, 1+height):
          classifier_branch = cur_code % 2
          classifier_idx = int((cur_code - 1) / 2)
          # update cur_code
          cur_code = classifier_idx
          if classifier_branch == 1:
            temp += tf.log(1.0 - label_encoding_predict[:,classifier_idx]+ 0.00001)
          else:
            temp += tf.log(label_encoding_predict[:,classifier_idx]+0.00001)
      encoded_prob_list.append(temp)

  encoded_prob = tf.exp(tf.stack(encoded_prob_list,axis=1))  # bs*tree_num_intervals
  encoded_playtime = tf.reduce_sum(encoded_playtime*encoded_prob,axis=-1,keepdims=True)

  e_x2 = tf.reduce_sum(tf.square(encoded_playtime)*encoded_prob, axis=-1,keepdims=True)
  square_of_e_x = tf.square(encoded_playtime)
  var = tf.sqrt(e_x2 - square_of_e_x)
  return encoded_playtime, tf.reduce_sum(var)



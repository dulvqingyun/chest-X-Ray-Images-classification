import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from lib.data_loader import ChestXRayDataLoader
from lib.config import ConfigReader, TrainNetConfig, TestNetConfig, DataConfig
from lib.network import NetWork


def train():
    tf.reset_default_graph()

    config_reader = ConfigReader('./utils/config.yml')
    train_config = TrainNetConfig(config_reader.get_train_config())
    data_config = DataConfig(config_reader.get_train_config())

    train_log_dir = './logs/train/'
    val_log_dir = './logs/val/'

    
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if not os.path.exists(val_log_dir):
        os.makedirs(val_log_dir)

    net = NetWork(train_config)

    with tf.name_scope('input'):
        data_loader = ChestXRayDataLoader(data_config,is_train=False, is_shuffle=True)
        train_image_batch, train_label_batch = data_loader.generate_batch(data_category='train')
        
        val_image_batch, val_label_batch = data_loader.generate_batch(data_category='val')

    net.build_model()    
    train_op = net.optimize()
    summaries = net.get_summary()

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge(summaries)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    try:
        for step in np.arange(train_config.max_step):
            if coord.should_stop():
                break

            train_image, train_label = sess.run([train_image_batch, train_label_batch])
#            image1 = train_image[0:1,:,:,:]
#            image1 = image1.reshape((224,224,3))
#            plt.imshow(image1)
            _,train_logits, train_loss, train_acc = sess.run([train_op,net.logits, net.loss, net.accuracy],
                                                feed_dict={net.x: train_image, net.y: train_label})

            if step % 10 == 0 or step + 1 == train_config.max_step:
                print('===TRAIN===: Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, train_loss, train_acc))
                summary_str = sess.run(summary_op, feed_dict={net.x: train_image, net.y: train_label})
                train_summary_writer.add_summary(summary_str, step)
            if step % 100 == 0 or step + 1 == train_config.max_step:
                val_image, val_label = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([net.loss, net.accuracy], feed_dict={net.x: val_image, net.y: val_label})
                print('====VAL====: Step %d, val loss = %.4f, val accuracy = %.4f%%' % (step, val_loss, val_acc))
                summary_str = sess.run(summary_op, feed_dict={net.x: val_image, net.y: val_label})
                val_summary_writer.add_summary(summary_str, step)
            if step % 1000 == 0 or step + 1 == train_config.max_step:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('===INFO====: Training completed, reaching the maximum number of steps')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

        
def test():
    
    
    tf.reset_default_graph()
    config_reader = ConfigReader('./utils/config.yml')
    test_config = TestNetConfig(config_reader.get_test_config())
    data_config = DataConfig(config_reader.get_test_config())
    test_loader = ChestXRayDataLoader(data_config, is_train=False, is_shuffle=True)
    
    BATCH_SIZE =test_loader.batch_size
    test_data,test_label = test_loader.readData()
#    test_data,test_label,_ = test_loader.shuffleData(test_data,test_label)
    num_batch = test_data.shape[0]//BATCH_SIZE #batch size
    
    net = NetWork(test_config)
    net.build_model()


    saver = tf.train.Saver(tf.global_variables())

    
    ckpt_path = test_config.model_path
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
    saver.restore(sess, ckpt.model_checkpoint_path)

    total_test_acc=0
    tn_all=0
    fp_all=0
    fn_all=0
    tp_all=0
    for batch_idx in range(num_batch):
        start_idx = batch_idx*BATCH_SIZE
        end_idx = (batch_idx+1)*BATCH_SIZE
        
        current_data = test_data[start_idx:end_idx,:,:,:]
        current_label = test_label[start_idx:end_idx]
        
        # turn label to one hot
        current_label_one_hot = test_loader.dense_to_one_hot(current_label,2)
        
        test_logits, test_loss, test_acc = sess.run([net.logits,net.loss, net.accuracy], 
                                         feed_dict={net.x: current_data, net.y: current_label_one_hot})
        print('====test====: Step %d, test loss = %.4f, test accuracy = %.4f%%' % (batch_idx, test_loss, test_acc))
        total_test_acc+=test_acc
        test_logits = np.argmax(test_logits,1)
        test_label_argmax = np.argmax(current_label_one_hot,1)
            
#            recall = recall_score(test_label_argmax,test_logits)
        for i in test_logits:
            if test_logits[i]==0 and test_label_argmax[i]==0:
                tn_all+=1
            if test_logits[i]==0 and test_label_argmax[i]==1:
                fn_all+=1
            if test_logits[i]==1 and test_label_argmax[i]==0:
                fp_all+=1
            if test_logits[i]==1 and test_label_argmax[i]==1:
                tp_all+=1
                                           

    print('===test====: Testing completed,total accuracy is %.4f' % (total_test_acc/num_batch))     
    print('recall is %.4f:' % (tp_all/(tp_all+fn_all)))
    print('precision is %.4f:'% (tp_all/(tp_all+fp_all)))
        

if __name__ == '__main__':
    train()
#    test()

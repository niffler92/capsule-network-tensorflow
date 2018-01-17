import os
import argparse
import uuid
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

import models
from common.utils import *
from params import params


def train(dataset):
    train_dataset, num_tr_batch, val_dataset, num_val_batch = build_dataset(dataset, True)
    train_iterator = train_dataset.make_initializable_iterator()
    train_batch = train_iterator.get_next()
    model = find_class_by_name([models], 'CapsNet')(params)
    model.build_graph()

    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        log_device_placement=False,
        allow_soft_placement=True)
    )
    unique_key = uuid.uuid1().hex[:6]
    saver = set_saver(sess, params)
    summary_writer = tf.summary.FileWriter(
        os.path.join('checkpoint', unique_key), flush_secs=10)

    valid_acc_best = -1e10

    for epoch in range(params['epochs']):
        total_loss = 0
        margin_loss = 0
        train_acc = 0
        st = time.time()
        y_preds = []
        y_trues = []
        sess.run(train_iterator.initializer)

        for _ in range(num_tr_batch):

            batch_x, batch_y = sess.run(train_batch)
            _, acc_, total_loss_, margin_loss_, step, summary, y_pred= \
                sess.run(
                    [model.train_op, model.accuracy, model.total_loss, model.margin_loss,
                    model.global_step, model.summary_train, model.y_pred],
                    feed_dict={model.x: batch_x, model.y: batch_y}
                )

            total_loss += total_loss_ / num_tr_batch
            margin_loss += margin_loss_ / num_tr_batch
            train_acc += acc_ / num_tr_batch
            y_preds += y_pred.ravel().tolist()
            y_trues += batch_y.ravel().tolist()

            if step % params['step_save_summaries'] == 0:
                print("Step:  {:5d} | batch total loss: {:.5f} | batch margin loss: {:.5f} | batch acc: {:.5f}".format(
                        int(step), total_loss_, margin_loss_, acc_))
                summary_writer.add_summary(summary, global_step=step)

        val_acc, val_total_loss, val_margin_loss = evaluate(step, model, val_dataset, num_val_batch, sess, summary_writer)

        if val_acc > valid_acc_best:
            tf.logging.info("Saving best ckpt....")
            ckpt_filename = os.path.join('checkpoint', unique_key,
                                         "capsnet-{}".format(datetime.now().strftime("%y%m%d%H%M%S"))
                                         )
            saver.save(sess, ckpt_filename)

        elapsed_time = time.time() - st
        real_epoch = int(step / num_tr_batch)
        print("Step: {:5d} | Epoch: {:3d} | Elapsed time: {:3.2f} | "
              "train_total_loss: {:.5f} | train_margin_loss: {:.5f} | train_acc: {:.5f} | "
              "valid_total_loss: {:.5f} | valid_margin_loss: {:.5f} | valid_acc: {:.5f}".format(
               int(step), real_epoch, elapsed_time, total_loss, margin_loss, train_acc,
               val_total_loss, val_margin_loss, val_acc))

    print("Training Finished!")
    summary_writer.close()
    sess.close()

    return unique_key


def evaluate(step, model, val_dataset, num_val_batch, session, summary_writer):
    val_iterator = val_dataset.make_initializable_iterator()
    val_batch = val_iterator.get_next()
    total_loss = 0
    margin_loss = 0
    acc = 0
    y_preds = []
    y_trues = []

    session.run(val_iterator.initializer)
    for _ in range(num_val_batch):
        batch_x, batch_y = session.run(val_batch)
        summary, total_loss_, margin_loss_, acc_, y_pred = session.run(
            [model.summary_valid, model.total_loss, model.margin_loss, model.accuracy,
             model.y_pred], feed_dict={model.x: batch_x, model.y: batch_y})

        total_loss += total_loss_ / num_val_batch
        margin_loss += margin_loss_ / num_val_batch
        acc += acc_ / num_val_batch
        y_preds.extend(y_pred.ravel().tolist())
        y_trues.extend(batch_y.ravel().tolist())

    if (summary_writer is not None) and (step % params['step_save_summaries'] == 0):
        summary_writer.add_summary(summary, global_step=step)

    return acc, total_loss, margin_loss


def inference(dataset, unique_key):
    tf.reset_default_graph()
    test_dataset, num_te_batch = build_dataset(dataset, False)
    test_iterator = test_dataset.make_initializable_iterator()
    test_batch = test_iterator.get_next()

    model = find_class_by_name([models], 'CapsNet')(params)
    model.build_graph()


    session = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        device_count={'GPU': 2},
        log_device_placement=False,
        allow_soft_placement=True)
    )
    restore_session(session, unique_key)
    y_trues = []
    y_preds = []

    session.run(test_iterator.initializer)
    st = time.time()
    for i in range(num_te_batch):
        batch_x, batch_y = session.run(test_batch)
        y_pred,  acc_, total_loss_, margin_loss = \
        session.run([model.y_pred,  model.accuracy, model.total_loss, model.margin_loss],
                    feed_dict={model.x: batch_x, model.y: batch_y})  # FIXME

        y_trues.extend(batch_y.ravel().tolist())
        y_preds.extend(y_pred.ravel().tolist())

    session.close()

    print("Model CapsNet was inferenced in {} seconds.".format(time.time() - st))
    for line in classification_report(y_trues, y_preds).split("\n"):
        print(line)
    print(confusion_matrix(y_trues, y_preds))
    test_acc = np.mean([y_true == y_preds[i] for i, y_true in enumerate(y_trues)])
    print("Test Accuracy: {}".format(test_acc))


def build_dataset(dataset, is_training):
    if is_training:
        train_X, train_Y, num_tr_batch, val_X, val_Y, num_val_batch = \
            load_data(dataset, params['batch_size'], is_training)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_X, train_Y)
        ).shuffle(55000).batch(params['batch_size'])
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_X, val_Y)
        ).shuffle(5000).batch(params['batch_size'])
        return train_dataset, num_tr_batch, val_dataset, num_val_batch
    else:
        test_X, test_Y, num_te_batch = \
            load_data(dataset, params['batch_size'], is_training)

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_X, test_Y)
        ).shuffle(10000).batch(params['batch_size'])
        return test_dataset, num_te_batch


def set_saver(session, args):
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    if args['checkpoint_path'] is not "":
        if os.path.isdir(args['checkpoint_path']):
            old_checkpoint_path = args['checkpoint_path']
            args['checkpoint_path'] = tf.train.latest_checkpoint(args['checkpoint_path'])
            tf.logging.info("Update checkpoint_path: {} -> {}".format(
                old_checkpoint_path, args['checkpoint_path'])
            )
        saver.restore(session, args['checkpoint_path'])
        tf.logging.info("Restore from {}".format(args['checkpoint_path']))
    else:
        tf.logging.info("No designated checkpoint path. Initializing weights randomly.")

    return saver


def restore_session(session, unique_key):
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    ckpt_path = os.path.join("checkpoint", unique_key)
    try:
        if os.path.isdir(ckpt_path):
            old_checkpoint_path = ckpt_path
            ckpt_path = tf.train.latest_checkpoint(ckpt_path)
            tf.logging.info("Update checkpoint_path: {} -> {}".format(
                old_checkpoint_path, ckpt_path)
            )
        tf.logging.info("Restoring from {}".format(ckpt_path))
        saver.restore(session, ckpt_path)
    except:
        raise Exception("Something is wrong with ckpt.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu_device", default=0, type=int)
    parser.add_argument("--dataset", default='mnist', type=str, choices=['mnist', 'fashion_mnist'])
    args = parser.parse_args()
    print("Params:")
    [print("{}={}".format(k, v)) for k, v in sorted(params.items())]
    with tf.device("/gpu:{}".format(args.gpu_device)):
        unique_key = train(args.dataset)
        inference(args.dataset, unique_key)

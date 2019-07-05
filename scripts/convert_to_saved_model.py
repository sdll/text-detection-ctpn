# coding=utf-8
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from nets import model_train as model


def convert_to_saved_model(checkpoint_path, output_path):
    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name="input_image"
        )

        input_im_info = tf.placeholder(
            tf.float32, shape=[None, 3], name="input_im_info"
        )

        inputs = {
            "input_image": tf.saved_model.utils.build_tensor_info(input_image),
            "input_im_info": tf.saved_model.utils.build_tensor_info(input_im_info),
        }

        bbox_pred, _, cls_prob = model.model(input_image)

        outputs = {
            "bbox_pred": tf.saved_model.utils.build_tensor_info(bbox_pred),
            "cls_prob": tf.saved_model.utils.build_tensor_info(cls_prob),
        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
        )

        global_step = tf.get_variable(
            "global_step", [], initializer=tf.constant_initializer(0), trainable=False
        )
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)

        saver = tf.train.Saver(variable_averages.variables_to_restore())
        builder = tf.saved_model.builder.SavedModelBuilder(args.output_path)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(
                checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path)
            )
            saver.restore(sess, model_path)
            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
                },
                strip_default_attrs=True,
            )
            builder.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the CTPN checkpoint to SavedModel"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="dist/data", help="The checkpoint path"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="dist/saved_model",
        help="The SavedModel path",
    )
    args = parser.parse_args()
    convert_to_saved_model(
        checkpoint_path=args.checkpoint_path, output_path=args.output_path
    )

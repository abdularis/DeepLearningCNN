# model_exporter.py
# Created by abdularis on 12/07/18


import argparse
import importlib


def export_model(model_arch_module, model_path, output_path):
    import tensorflow as tf

    model = model_arch_module.build_model_arch()
    extractor = model.stored_ops['features']

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        ti_input_images = tf.saved_model.utils.build_tensor_info(model.x)
        ti_output_probs = tf.saved_model.utils.build_tensor_info(model.prediction)
        ti_output_features = tf.saved_model.utils.build_tensor_info(extractor)

        pred_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': ti_input_images},
            outputs={'probs': ti_output_probs, 'features': ti_output_features},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        builder = tf.saved_model.builder.SavedModelBuilder(output_path)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images': pred_signature,
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: pred_signature
            }
        )
        builder.save()
        print("Export model selesai! - %s" % output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model checkpoint ke tensorflow saved model')
    parser.add_argument('--model-module', type=str, help='Python module string untuk model cnn', required=True)
    parser.add_argument('--model-path', type=str, help='Path model CNN', required=True)
    parser.add_argument('--output-path', type=str, help='Output path saved model', required=True)

    args = parser.parse_args()

    export_model(
        importlib.import_module(args.model_module),
        args.model_path,
        args.output_path
    )

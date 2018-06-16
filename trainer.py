# trainer.py
# Created by abdularis on 16/05/18

import argparse

INPUT_SHAPE = [None, 128, 128, 3]
NUM_CLASSES = 6
LEARNING_RATE = 1e-4


def get_mean_op():
    import tensorflow as tf

    accuracies = tf.placeholder(tf.float32, shape=[None])
    mean_accuracy = tf.reduce_mean(accuracies)

    losses = tf.placeholder(tf.float32, shape=[None])
    mean_loss = tf.reduce_mean(losses)

    train_summary = tf.summary.merge(
        [
            tf.summary.scalar('train_accuracy_per_epoch', mean_accuracy),
            tf.summary.scalar('train_cross_entropy_per_epoch', mean_loss)
        ]
    )

    val_summary = tf.summary.merge(
        [
            tf.summary.scalar('val_accuracy_per_epoch', mean_accuracy),
            tf.summary.scalar('val_cross_entropy_per_epoch', mean_loss)
        ]
    )

    train_fetches = [mean_accuracy, mean_loss, train_summary]
    val_fetches = [mean_accuracy, mean_loss, val_summary]

    return accuracies, losses, train_fetches, val_fetches


def run_trainer(model_ver, num_epocs, batch_size, dataset_path, model_name, run_name):

    import tensorflow as tf
    import cnn
    import data_reader
    import data_config as cfg

    if model_ver == 2:
        model = cnn.build_model_arch_v2(INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE)
    else:
        model = cnn.build_model_arch(INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE)
    train_data, val_data, test_data = data_reader.read_data_set_dir(dataset_path, cfg.one_hot, batch_size)

    accuracies_input, losses_input, train_mean, val_mean = get_mean_op()

    global_step = 0
    file_writer = tf.summary.FileWriter('logs/%s/' % run_name)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        file_writer.add_graph(sess.graph)

        try:
            for epoch in range(num_epocs):

                train_accuracies = []
                train_losses = []
                for step in range(train_data.batch_count):
                    batch_images, batch_labels = train_data.next_batch()
                    batch_images = batch_images / 255.0

                    if step % 10 == 0:
                        train_accuracy, loss, summary = model.train_step(sess, batch_images, batch_labels,
                                                                         run_summary=True)
                        file_writer.add_summary(summary, global_step)

                        print('Epoch %d, step %d, global step %d, training accuracy: %f, training loss %f'
                              % (epoch, step, global_step, train_accuracy, loss))
                    else:
                        train_accuracy, loss = model.train_step(sess, batch_images, batch_labels, run_summary=False)

                    train_accuracies.append(train_accuracy)
                    train_losses.append(loss)
                    global_step += 1

                mean_acc, mean_loss, summ = sess.run(train_mean,
                                                     feed_dict={accuracies_input: train_accuracies,
                                                                losses_input: train_losses})
                print('Epoch %d: Training accuracy: %f, loss %f' % (epoch, mean_acc, mean_loss))
                file_writer.add_summary(summ, epoch)

                val_accuracies = []
                val_losses = []
                for step in range(val_data.batch_count):
                    batch_images, batch_labels = val_data.next_batch()
                    batch_images = batch_images / 255.0

                    val_accuracy, loss, summary = model.evaluate(sess, batch_images, batch_labels)
                    val_accuracies.append(val_accuracy)
                    val_losses.append(loss)

                mean_acc, mean_loss, summ = sess.run(val_mean,
                                                     feed_dict={accuracies_input: val_accuracies,
                                                                losses_input: val_losses})
                print('Epoch %d: Validation accuracy: %f, loss %f' % (epoch, mean_acc, mean_loss))
                file_writer.add_summary(summ, epoch)

                saver.save(sess, save_path='./model/%s' % model_name, global_step=epoch)
        except KeyboardInterrupt:
            print('Training cancelled intentionally.')

        print('Stop training at %d steps' % global_step)
        file_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latih model CNN')
    parser.add_argument('--model-version', type=int, help='Versi model cnn', required=True)
    parser.add_argument('--num-epochs', type=int, help='Jumlah epoch', required=True)
    parser.add_argument('--batch-size', type=int, help='Ukuran batch/jumlah data per batch/iterasi', required=True)
    parser.add_argument('--dataset-path', type=str, help='Path ke dataset', required=True)
    parser.add_argument('--model-name', type=str, help='Nama model output', required=True)
    parser.add_argument('--run-name', type=str, help='Nama run untuk trainer ini dijalankan', required=True)

    args = parser.parse_args()

    print('Run trainer:')
    print('\tModel version: %d' % args.model_version)
    print('\tNum epoch: %d' % args.num_epochs)
    print('\tBatch size: %d' % args.batch_size)
    print('\tDataset path: %s' % args.dataset_path)
    print('\tModel name: %s' % args.model_name)
    print('\tRun name: %s' % args.run_name)

    run_trainer(
        model_ver=args.model_version,
        num_epocs=args.num_epochs,
        batch_size=args.batch_size,
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        run_name=args.run_name
    )

import tensorflow as tf
import inputs
import model
import sys
from evaluation import *


def eval_batch(eval_csv_path, sess, labels, predictions, csv_record, step, summary_writer, class_map, eval_loss_tensor):
     # Read in the validation CSV, skipping the header.
    eval_records = open(eval_csv_path).readlines()[1:]
    # Shuffle the lines so that as we print incremental stats, we get good
    # coverage acrosst classes and get a quick initial impression of how well
    # the model is doing across classes well before evaluation is completed.
    random.shuffle(eval_records)

    lwlrap = Lwlrap(class_map)
    total_loss = []
    for (i, record) in enumerate(eval_records):
        record = record.strip()
        print("[%d of %d]" % (i + 1, len(eval_records)), record)
        sys.stdout.flush()

        actual, predicted, loss = sess.run([labels, predictions, eval_loss_tensor], {csv_record: record})

        total_loss += [loss]
        # By construction, actual consists of identical rows, where each row is
        # the same 1-hot label (because we are looking at features from the same
        # clip). So we can just use the first row as the ground truth.
        actual_labels = actual[0]

        # We make a clip prediction by averaging the prediction scores across
        # all examples for the clip.
        predicted_labels = np.average(predicted, axis=0)

        # Update eval metric.
        lwlrap.accumulate(actual_labels[np.newaxis, :], predicted_labels[np.newaxis, :])

        # For quick feedback, print running lwlrap periodically and generate a
        # partial lwlrap summary from 5% of the eval data.
        if i % 10 == 0:
            print('\n', lwlrap, '\n', sep='')
            sys.stdout.flush()
        if i == int(0.05 * len(eval_records)):
            lwlrap_summary = make_scalar_summary('Lwlrap-5%', lwlrap.overall_lwlrap())
            summary_writer.add_summary(lwlrap_summary, step)
            summary_writer.flush()

    print('\nFINAL LWLRAP:\n\n', lwlrap, sep='')
    sys.stdout.flush()

    total_loss = np.mean(np.array(total_loss))
    loss_summary = make_scalar_summary('loss', total_loss)
    summary_writer.add_summary(loss_summary, step)

    lwlrap_summary = make_scalar_summary('Lwlrap', lwlrap.overall_lwlrap())
    summary_writer.add_summary(lwlrap_summary, step)
    summary_writer.flush()

    return lwlrap


def train_and_evaluate(model_name=None, hparams=None, class_map_path=None, train_csv_path=None, train_clip_dir=None,
                       train_dir=None, epoch_batches=None, warmstart_checkpoint=None,
                       warmstart_include_scopes=None, warmstart_exclude_scopes=None,
                       eval_csv_path=None, eval_clip_dir=None, eval_dir=None):
    """Runs the training loop."""
    print('\nTraining model:{} with hparams:{} and class map:{}'.format(model_name, hparams, class_map_path))
    print('Training data: clip dir {} and labels {}'.format(train_clip_dir, train_csv_path))
    print('Training dir {}\n'.format(train_dir))

    class_map = {int(row[0]): row[1] for row in csv.reader(open(class_map_path))}

    with tf.Graph().as_default():
        # Create the input pipeline.
        features, labels, num_classes, input_init = inputs.train_input(
            train_csv_path=train_csv_path, train_clip_dir=train_clip_dir, class_map_path=class_map_path,
            hparams=hparams)
        # Create the model in training mode.
        global_step, prediction, loss_tensor, train_op = model.define_model(
            model_name=model_name, features=features, labels=labels, num_classes=num_classes,
            hparams=hparams, epoch_batches=epoch_batches, training=True)

        # evaluation graph
        label_class_index_table, num_classes = inputs.get_class_map(class_map_path)
        csv_record = tf.placeholder(tf.string, [])  # fed during evaluation loop.

        eval_features, eval_labels = inputs.record_to_labeled_log_mel_examples(
            csv_record, clip_dir=eval_clip_dir, hparams=hparams,
            label_class_index_table=label_class_index_table, num_classes=num_classes)

        # Create the model in prediction mode.
        global_step, eval_predictions, eval_loss_tensor, _ = model.define_model(
            model_name=model_name, features=eval_features, labels=eval_labels, num_classes=num_classes,
            hparams=hparams, training=False, evaluating=True)

        # Write evaluation graph to checkpoint directory.
        tf.train.write_graph(tf.get_default_graph().as_graph_def(add_shapes=True),
                             eval_dir, 'eval.pbtxt')

        eval_writer = tf.summary.FileWriter(eval_dir, tf.get_default_graph())

        # Define our own checkpoint saving hook, instead of using the built-in one,
        # so that we can specify additional checkpoint retention settings.
        saver = tf.train.Saver(
            max_to_keep=10, keep_checkpoint_every_n_hours=0.25)
        saver_hook = tf.train.CheckpointSaverHook(
            save_steps=100, checkpoint_dir=train_dir, saver=saver)

        summary_op = tf.summary.merge_all()
        summary_hook = tf.train.SummarySaverHook(
            save_steps=10, output_dir=train_dir, summary_op=summary_op)

        if hparams.warmstart:
            var_include_scopes = warmstart_include_scopes
            if not var_include_scopes: var_include_scopes = None
            var_exclude_scopes = warmstart_exclude_scopes
            if not var_exclude_scopes: var_exclude_scopes = None
            restore_vars = tf.contrib.framework.get_variables_to_restore(
                include=var_include_scopes, exclude=var_exclude_scopes)
            # Only restore trainable variables, we don't want to restore
            # batch-norm or optimizer-specific local variables.
            trainable_vars = set(tf.contrib.framework.get_trainable_variables())
            restore_vars = [var for var in restore_vars if var in trainable_vars]

            print('Warm-start: restoring variables:\n%s\n' % '\n'.join([x.name for x in restore_vars]))
            print('Warm-start: restoring from ', warmstart_checkpoint)
            assert restore_vars, 'No warm-start variables to restore!'
            restore_op, feed_dict = tf.contrib.framework.assign_from_checkpoint(
                model_path=warmstart_checkpoint, var_list=restore_vars, ignore_missing_vars=True)

            scaffold = tf.train.Scaffold(
                init_fn=lambda scaffold, session: session.run(restore_op, feed_dict),
                summary_op=summary_op, saver=saver)
        else:
            scaffold = None

        with tf.train.SingularMonitoredSession(hooks=[saver_hook, summary_hook],
                                               checkpoint_dir=train_dir,
                                               scaffold=scaffold,
                                               config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.raw_session().run(input_init)
            while not sess.should_stop():

                # train
                step, _, pred, loss = sess.run([global_step, train_op, prediction, loss_tensor])
                print(step, loss)
                sys.stdout.flush()

                # evaluates every 100 steps
                if step > 0 and step % 100 == 0:
                    # Loop through all checkpoints in the training directory.
                    checkpoint_state = tf.train.get_checkpoint_state(train_dir)

                    lwlrap = eval_batch(eval_csv_path, sess, eval_labels, eval_predictions, csv_record, step, eval_writer, class_map, eval_loss_tensor)

                        
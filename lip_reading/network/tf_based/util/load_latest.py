import tensorflow as tf


def LoadLatestWeight(model, weight_dir):
    latest = tf.train.latest_checkpoint(weight_dir)

    if latest is not None:
        model.load_weights(latest)
        print("Loading latest weight from path: "+str(latest))
    else:
        print("No existing weight. Strat training from scratch")

    return model



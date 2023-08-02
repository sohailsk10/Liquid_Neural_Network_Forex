import numpy as np
import pandas as pd
import os
import tensorflow as tf
import argparse
import datetime as dt
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse
import datetime as dt
from tensorflow.keras.models import save_model
import h5py
from tensorflow.keras.models import Model


def load_forex_data():
    # Load your Forex data from the CSV file
    df = pd.read_csv("data/forex.csv")

    # Preprocess your data as needed
    bid_values = df["bid"].values.astype(np.float32)
    # Normalize the bid values
    bid_values -= np.mean(bid_values)
    bid_values /= np.std(bid_values)

    # Extract datetime features
    date_time = df["datetime"].values
    # Assuming the datetime format is "yyyy-mm-dd hh:mm:ss"
    date_time = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f") for d in date_time]
    weekday = np.array([d.weekday() for d in date_time]).astype(np.float32)
    noon = np.array([d.hour for d in date_time]).astype(np.float32)
    noon = np.sin(noon * np.pi / 24)

    # Add dummy features to match the model's input shape
    holiday = np.zeros_like(weekday)
    temp = np.zeros_like(weekday)
    rain = np.zeros_like(weekday)
    snow = np.zeros_like(weekday)
    clouds = np.zeros_like(weekday)

    features = np.stack([holiday, temp, rain, snow, clouds, weekday, noon], axis=-1)

    return features, bid_values

def cut_in_sequences(x, y, seq_len, inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)

class ForexData:
    def __init__(self, seq_len=32):

        x, y = load_forex_data()

        train_x, train_y = cut_in_sequences(x, y, seq_len, inc=4)

        self.train_x = np.stack(train_x, axis=0)
        self.train_y = np.stack(train_y, axis=0)
        total_seqs = self.train_x.shape[1]
        print("Total number of training sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)

        self.valid_x = self.train_x[:, permutation[:valid_size]]
        self.valid_y = self.train_y[:, permutation[:valid_size]]
        self.test_x = self.train_x[:, permutation[valid_size : valid_size + test_size]]
        self.test_y = self.train_y[:, permutation[valid_size : valid_size + test_size]]
        self.train_x = self.train_x[:, permutation[valid_size + test_size :]]
        self.train_y = self.train_y[:, permutation[valid_size + test_size :]]

    def iterate_train(self, batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = self.train_x[:, permutation[start:end]]
            batch_y = self.train_y[:, permutation[start:end]]
            yield (batch_x, batch_y)




class ForexModel:
    def save_best_model_to_h5(self, h5_file_path):
        with h5py.File(h5_file_path, "w") as h5_file:
        # Save the model architecture and weights
            model_config = self.model.to_json()
            h5_file.attrs["model_config"] = model_config
            self.model.save_weights(h5_file)
        # Save any additional data required for model inference (if needed)
        # For example, you can save the mean and std values for input normalization

        print("Best model saved to", h5_file_path)

    def __init__(self, model_type, model_size, learning_rate=0.01):
        self.model = None
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 7])
        self.target_y = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.model_size = model_size
        head = self.x
        if model_type == "lstm":
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)

            head, _ = tf.nn.dynamic_rnn(
                self.fused_cell, head, dtype=tf.float32, time_major=True
            )
        elif model_type.startswith("ltc"):
            learning_rate = 0.01  # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if model_type.endswith("_rk"):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif model_type.endswith("_ex"):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head, _ = tf.nn.dynamic_rnn(
                self.wm, head, dtype=tf.float32, time_major=True
            )
            self.constrain_op = self.wm.get_param_constrain_op()
        elif model_type == "node":
            self.fused_cell = NODE(model_size, cell_clip=10)
            head, _ = tf.nn.dynamic_rnn(
                self.fused_cell, head, dtype=tf.float32, time_major=True
            )
        elif model_type == "ctgru":
            self.fused_cell = CTGRU(model_size, cell_clip=-1)
            head, _ = tf.nn.dynamic_rnn(
                self.fused_cell, head, dtype=tf.float32, time_major=True
            )
        elif model_type == "ctrnn":
            self.fused_cell = CTRNN(model_size, cell_clip=-1, global_feedback=True)
            head, _ = tf.nn.dynamic_rnn(
                self.fused_cell, head, dtype=tf.float32, time_major=True
            )
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        target_y = tf.expand_dims(self.target_y, axis=-1)
        self.y = tf.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(),
        )(head)
        print("logit shape: ", str(self.y.shape))
        self.loss = tf.reduce_mean(tf.square(target_y - self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        self.accuracy = tf.reduce_mean(tf.abs(target_y - self.y))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join(
            "results", "forex", "{}_{}.csv".format(model_type, model_size)
        )
        if not os.path.exists("results/forex"):
            os.makedirs("results/forex")
        if not os.path.isfile(self.result_file):
            with open(self.result_file, "w") as f:
                f.write(
                    "best epoch, train loss, train mae, valid loss, valid mae, test loss, test mae\n"
                )

        self.checkpoint_path = os.path.join(
            "tf_sessions", "forex", "{}".format(model_type)
        )
        if not os.path.exists("tf_sessions/forex"):
            os.makedirs("tf_sessions/forex")

        self.saver = tf.train.Saver()

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)

    def fit(self, gesture_data, epochs, verbose=True, log_period=50):

        best_valid_loss = np.PINF
        best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
        self.save()
        for e in range(epochs):
            if verbose and e % log_period == 0:
                test_acc, test_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: gesture_data.test_x, self.target_y: gesture_data.test_y},
                )
                valid_acc, valid_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: gesture_data.valid_x, self.target_y: gesture_data.valid_y},
                )
                # MSE metric -> less is better
                if (valid_loss < best_valid_loss and e > 0) or e == 1:
                    best_valid_loss = valid_loss
                    best_valid_stats = (
                        e,
                        np.mean(losses),
                        np.mean(accs),
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc,
                    )
                    self.save()

            losses = []
            accs = []
            for batch_x, batch_y in gesture_data.iterate_train(batch_size=16):
                acc, loss, _ = self.sess.run(
                    [self.accuracy, self.loss, self.train_step],
                    {self.x: batch_x, self.target_y: batch_y},
                )
                if not self.constrain_op is None:
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if verbose and e % log_period == 0:
                print(
                    "Epochs {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.2f}, valid mae: {:0.2f}, test loss: {:0.2f}, test mae: {:0.2f}".format(
                        e,
                        np.mean(losses),
                        np.mean(accs),
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc,
                    )
                )
            if e > 0 and (not np.isfinite(np.mean(losses))):
                break
        self.restore()
        (
            best_epoch,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc,
            test_loss,
            test_acc,
        ) = best_valid_stats
        print(
            "Best epoch {:03d}, train loss: {:0.3f}, train mae: {:0.3f}, valid loss: {:0.3f}, valid mae: {:0.3f}, test loss: {:0.3f}, test mae: {:0.3f}".format(
                best_epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                test_loss,
                test_acc,
            )
        )
        with open(self.result_file, "a") as f:
            f.write(
                "{:08d}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}\n".format(
                    best_epoch,
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                    test_loss,
                    test_acc,
                )
            )
        
        # Export the best model to .h5 file
        best_model_h5_path = "best_model.h5"
        self.save_best_model_to_h5(best_model_h5_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ltc")
    parser.add_argument("--log", default=1, type=int)
    parser.add_argument("--size", default=32, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    args = parser.parse_args()

    forex_data = ForexData()
    model = ForexModel(model_type=args.model, model_size=args.size)

    model.fit(forex_data, epochs=args.epochs, log_period=args.log)

    # Export the best model to .h5 file
    model.save_best_model_to_h5("best_model.h5")


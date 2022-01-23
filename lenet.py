import numpy as np
import time
import pickle
import os
from tqdm.contrib import tzip

from utils import *


# 在原 LeNet-5上进行少许修改后的 网路结构
"""
conv1: in_channels: 1, out_channel:6, kernel_size=(5x5), pad=0, stride=1, activation: relu
avgpool1: in_channels: 6, out_channels:6, kernel_size = (2x2), stride=2
conv2: in_channels: 6, out_channel:16, kernel_size=(5x5), pad=0, stride=1, activation: relu
avgpool2: in_channels: 16, out_channels:16, kernel_size = (2x2), stride=2
flatten
fc1: in_channel: 256, out_channels: 128, activation: relu
fc2: in_channel: 128, out_channels: 64, activation: relu
fc3: in_channel: 64, out_channels: 10, activation: relu
softmax:

tensor: (1x28x28)   --conv1    -->  (6x24x24)
tensor: (6x24x24)   --avgpool1 -->  (6x12x12)
tensor: (6x12x12)   --conv2    -->  (16x8x8)
tensor: (16x8x8)    --avgpool2 -->  (16x4x4)
tensor: (16x4x4)    --flatten  -->  (256)
tensor: (256)       --fc1      -->  (128)
tensor: (128)       --fc2      -->  (64)
tensor: (64)        --fc3      -->  (10)
tensor: (10)        --softmax  -->  (10)
"""


class Sigmoid(object):
    def __init__(self):
        super(self).__init__()
        self.output = None

    def forward(self, x):
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, grad_y):
        return grad_y * self.output * (1 - self.output)


class Relu(object):
    def __init__(self):
        self.output = None

    def forward(self, x):  # x: ndarray
        self.output = np.maximum(x, 0)  # need to remember output for current batch
        return self.output  # ndarray with the same shape as x

    def backward(self, grad_y):  # grad_y: ndarray
        return np.multiply(np.sign(self.output), grad_y)  # ndarray with the same shape as grad_y


class Conv(object):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 pad: int = 0,
                 stride: int = 1,
                 activation : str = "relu"
                 ):
        # parameters
        self.in_channels = in_channels  # number of input channels
        self.out_channels = out_channels  # number of output channels
        self.kernel_size = kernel_size
        self.pad = pad
        self.stride = stride
        
        # init weight
        self.weight = np.random.rand(out_channels, in_channels, kernel_size, kernel_size) - 1 / 2  # uniform [-0.5, 0.5]

        # activation
        assert activation in ["sigmoid", "relu"], activation
        if activation == "relu":
            self.act = Relu()
        else:
            self.act = Sigmoid()
        
        # store input for backward
        self.input_padded = None

    def forward(self, x):  # x: ndarray with shape (B, C1, H1, W1) or (B, H1, W1)
        """
        Args:
        - x: ndarray with shape (B, C1, H1, W1) or (B, H1, W1)

        Returns:
        - output: ndarray with shape (B, C2, H2, W2)
        """
        st = self.stride
        sz = self.kernel_size
        pad = self.pad
        b = x.shape[0]  # number of images in a single batch
        c2 = self.out_channels  # number of output channels

        if len(x.shape) == 3:  # only one input channel
            c1 = 1
            h1, w1 = x.shape[1: 3]
            x = np.reshape(x, [b, 1, h1, w1])
        else:  # more than one input channels
            assert len(x.shape) == 4, x.shape
            c1, h1, w1 = x.shape[1: 4]

        assert self.in_channels == c1, "{}, {}".format(self.in_channels, c1)

        # pad zeros
        h1 += 2 * pad
        w1 += 2 * pad
        x_padded = np.zeros([b, c1, h1, w1])
        x_padded[:, :, pad: h1 - pad, pad: w1 - pad] = x

        self.input_padded = x_padded  # need to remember input_padded for current batch

        # compute output
        h2 = int((h1 - self.kernel_size) / self.stride) + 1
        w2 = int((w1 - self.kernel_size) / self.stride) + 1
        output = np.zeros([b, c2, h2, w2])

        for p in range(b):
            for q in range(c2):
                for r in range(h2):
                    for s in range(w2):
                        output[p, q, r, s] = \
                            np.sum(np.multiply(self.weight[q, :, :, :],
                                               x_padded[p, :, r * st: r * st + sz, s * st: s * st + sz]))

        return output

    def backward(self, lr, grad_y):
        """
        Args:
        - lr: learning rate
        - grad_y: ndarray with shape (B, C2, H2, W2)

        Returns:
        - grad_x: ndarray with shape (B, C1, H1, W1) or (B, H1, W1)
        """
        b, c2, h2, w2 = grad_y.shape
        c1 = self.in_channels
        h1 = self.kernel_size + (h2 - 1) * self.stride  # padded
        w1 = self.kernel_size + (w2 - 1) * self.stride  # padded
        st = self.stride
        sz = self.kernel_size
        pad = self.pad

        # compute grad_x and grad_w
        grad_x = np.zeros([b, c1, h1, w1])
        grad_w = np.zeros([c2, c1, sz, sz])
        for p in range(b):
            for r in range(h2):
                for s in range(w2):
                    grad_x[p, :, r * st: r * st + sz, s * st: s * st + sz] += \
                        np.tensordot(grad_y[p, :, r, s], self.weight, axes=([0], [0]))
                    grad_w += \
                        grad_y[p, :, r, s][:, np.newaxis, np.newaxis, np.newaxis] \
                        * self.input_padded[p, :, r * st: r * st + sz, s * st: s * st + sz][np.newaxis, :, :, :]

        h1 -= 2 * pad
        w1 -= 2 * pad
        grad_x = grad_x[:, :, pad: h1 + pad, pad: w1 + pad]
        if c1 == 1:
            grad_x = np.reshape(grad_x, [b, h1, w1])
        
        # update parameters
        self.weight -= lr * grad_w
        
        return grad_x


class AvgPool(object):
    def __init__(self, pool_size: int):
        self.pool_size = pool_size

    def forward(self, x):
        """
        Args:
        - x: ndarray with shape (B, C, H1, W1)

        Returns:
        - output: ndarray with shape (B, C, H2, W2)
        """
        b, c = x.shape[0: 2]
        sz = self.pool_size
        h2 = int(x.shape[2] / sz)
        w2 = int(x.shape[3] / sz)
        output = np.zeros([b, c, h2, w2])

        for i in range(b):
            for j in range(c):
                for k in range(h2):
                    for l in range(w2):
                        output[i, j, k, l] = x[i, j, sz * k: sz * k + sz, sz * l: sz * l + sz].mean()

        return output

    def backward(self, grad_y):
        """
        Args:
        - grad_y: ndarray with shape (B, C, H2, W2)

        Returns:
        - grad_x: ndarray with shape (B, C, H1, W1)
        """
        sz = self.pool_size
        amp = 1 / (sz ** 2)
        b, c, h2, w2 = grad_y.shape

        grad_x = np.ones([b, c, h2 * sz, w2 * sz])

        for k in range(h2):
            for l in range(w2):
                grad_x[:, :, k * sz: k * sz + sz, l * sz: l * sz + sz] = amp * grad_y[:, :, k, l][:, :, np.newaxis, np.newaxis]

        return grad_x


class Flatten(object):
    def __init__(self):
        self.in_shape = None

    def forward(self, x):
        """
        Args:
        - x: ndarray with shape (B, C1, H, W)

        Returns:
        - output: ndarray with shape (B, C2)
        """
        self.in_shape = x.shape
        output = np.reshape(x, [x.shape[0], -1])
        return output

    def backward(self, grad_y):
        """
        Args:
        - grad_y: ndarray with shape (B, C2)

        Returns:
        - grad_x: ndarray with shape (B, C1, H, W)
        """
        grad_x = np.reshape(grad_y, self.in_shape)
        return grad_x


class FC(object):
    def __init__(self, in_channels, out_channels):
        # parameters
        self.c1 = in_channels
        self.c2 = out_channels

        # init weight and bias
        self.weight = (np.random.rand(self.c1, self.c2) - 1 / 2) / 2  # uniform [-0.25, 0.25]
        self.bias = np.zeros(self.c2) - 1 / 2  # uniform [-0.5, -0.5]

        # store input for backward
        self.input = None

    def forward(self, x):
        """
        Args:
        - x: ndarray with shape (B, C1)

        Returns:
        - output: ndarray with shape (B, C2)
        """
        self.input = x
        output = x.dot(self.weight) + self.bias[np.newaxis, :]

        return output

    def backward(self, lr, grad_y):
        """
        Args:
        - lr: learning rate
        - grad_y: ndarray with shape (B, C2)

        Returns:
        - grad_x: ndarray with shape (B, C1)
        """
        # compute gradients
        grad_x = grad_y.dot(self.weight.T)
        grad_w = self.input.T.dot(grad_y)

        # update parameters
        self.weight -= lr * grad_w
        self.bias -= lr * np.sum(grad_y, axis=0)

        return grad_x


class SoftMax(object):
    def __init__(self):
        self.output = None

    def forward(self, x):
        """
        Args:
        - x: ndarray with shape (B, NumCategories)
        
        Returns:
        - output: ndarray with shape (B, NumCategories)
        """
        output = np.exp(x)
        output /= np.sum(output, axis=1)[:, np.newaxis]
        self.output = output

        return output 

    def backward(self, gt_label):
        """
        Args:
        - gt_label: one-hot ndarray with shape (B, NumCategories)
        
        Returns:
        - grad_x: ndarray with shape (B, NumCategories)
        """
        grad_x = self.output - gt_label
        return grad_x


class LeNet(object):
    def __init__(self, n_categories: int = 10, save_dir: str = "model"):
        self.conv1 = Conv(in_channels=1, out_channels=6, kernel_size=5)
        self.act1 = Relu()
        self.avgp1 = AvgPool(pool_size=2)
        self.conv2 = Conv(in_channels=6, out_channels=16, kernel_size=5)
        self.act2 = Relu()
        self.avgp2 = AvgPool(pool_size=2)
        self.flat = Flatten()
        self.fc1 = FC(in_channels=256, out_channels=128)
        self.act3 = Relu()
        self.fc2 = FC(in_channels=128, out_channels=64)
        self.act4 = Relu()
        self.fc3 = FC(in_channels=64, out_channels=n_categories)
        self.soft = SoftMax()

        self.model = []

        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        print("model initialized")

    def init_weight(self):
        pass

    def forward(self, x):
        """
        Args:
        - x: a batch of images, shape (B, C, H, W)

        Returns:
        - x: predicted posterior (not one-hot), shape (B, 10)
        """

        x = self.conv1.forward(x)
        x = self.act1.forward(x)
        x = self.avgp1.forward(x)
        x = self.conv2.forward(x)
        x = self.act2.forward(x)
        x = self.avgp2.forward(x)
        x = self.flat.forward(x)
        x = self.fc1.forward(x)
        x = self.act3.forward(x)
        x = self.fc2.forward(x)
        x = self.act4.forward(x)
        x = self.fc3.forward(x)
        x = self.soft.forward(x)

        return x

    def backward(self, gt, lr=1.0e-3):
        """
        Args:
        - gt: ground truth (one-hot), shape (B, 10)
        - lr: learning rate
        """

        grad = self.soft.backward(gt)
        grad = self.fc3.backward(lr, grad)
        grad = self.act4.backward(grad)
        grad = self.fc2.backward(lr, grad)
        grad = self.act3.backward(grad)
        grad = self.fc1.backward(lr, grad)
        grad = self.flat.backward(grad)
        grad = self.avgp2.backward(grad)
        grad = self.act2.backward(grad)
        grad = self.conv2.backward(lr, grad)
        grad = self.avgp1.backward(grad)
        grad = self.act1.backward(grad)
        grad = self.conv1.backward(lr, grad)
    
    def predict(self, x):
        """
        Args:
        - x: input images with shape (N, C, H, W)
        
        Returns:
        - pred_labels: predicted labels, shape (N)
        """
        posterior = self.forward(x)
        pred_labels = distribution_to_num(posterior)
        return pred_labels

    def evaluate(self, x, labels):
        """
        Args:
        - x: input images, shape (B, H, W)
        - labels: ground truth (one-hot), shape (B, 10)

        Returns:
        - acc: test accuracy
        - eval_time: evaluation time
        """

        t0 = time.time()
        print("evaluating...")
        pred = self.predict(x)
        num_correct = 0
        for n in range(labels.shape[0]):
            num_correct += labels[n, pred[n]]
        acc = num_correct / labels.shape[0]
        eval_time = time.time() - t0

        return acc, eval_time
    
    def compute_loss(self, y_pred, y_gt):
        """
        Args:
        - y_pred: predicted posterior (not one-hot), shape (N, 10)
        - y_gt: ground truth (one-hot), shape (N, 10)

        Returns:
        - loss
        """
        loss = (-np.log(y_pred) * y_gt).sum(axis=1)
        return loss

    def data_augmentation(self, images):
        """
        数据增强，可选操作，非强制，但是需要合理
        一些常用的数据增强选项： ramdom scale， translate， color(grayscale) jittering， rotation, gaussian noise,
        这一块儿允许使用 opencv 库或者 PIL image库
        比如把6旋转90度变成了9，但是仍然标签为6 就不合理了
        """
        return images

    def save_model(self, epoch: int):
        name = self.save_dir + "/model_" + str(epoch) + ".pkl"
        model = {'conv1': self.conv1.weight,
                 'conv2': self.conv2.weight,
                 'fc1': [self.fc1.weight, self.fc1.bias],
                 'fc2': [self.fc2.weight, self.fc2.bias],
                 'fc3': [self.fc3.weight, self.fc3.bias]
                 }
        
        with open(name, "wb") as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        
        print("model saved to " + name)

    def load_model(self, epoch: int):
        name = self.save_dir + "/model_" + str(epoch) + ".pkl"
        with open(name, "rb") as f:
            model = pickle.load(f)
            self.conv1.weight = model['conv1']
            self.conv2.weight = model['conv2']
            self.fc1.weight, self.fc1.bias = model['fc1']
            self.fc2.weight, self.fc2.bias = model['fc2']
            self.fc3.weight, self.fc3.bias = model['fc3']
        print("model loaded from " + name)

    def fit(
            self,
            train_image,
            train_label,
            test_image=None,
            test_label=None,
            prev_epochs: int = 0,
            epochs: int = 10,
            batch_size: int = 16,
            lr: float = 1.0e-3
    ):
        sum_time = 0
        avg_time = 0
        test_accuracies = []
        num_batches = int(train_image.shape[0] / batch_size)
        file = open("train_log.txt", "w+")

        train_image = self.data_augmentation(train_image)
        
        evaluate_before_training = False
        if evaluate_before_training:
            test_accuracy, eval_time = self.evaluate(test_image, test_label)
            test_accuracies.append(test_accuracy)
            str2 = "epoch {}, test accuracy: {:.2%}, evaluation time: {:.2f} seconds".format(prev_epochs, test_accuracy, eval_time)
            file.write(str2 + "\n")
            print(str2)

        for epoch in range(prev_epochs + 1, epochs + 1):
            # shuffle index
            batch_images = []
            batch_labels = []
            num_images = train_image.shape[0]
            idx = np.linspace(start=0, stop=num_images - 1, num=num_images).astype(int)
            np.random.shuffle(idx)
            
            # get batches
            for i in range(num_batches):
                curr_id = idx[i * batch_size : (i + 1) * batch_size]
                curr_batch_images = train_image[curr_id, :, :]
                curr_batch_labels = train_label[curr_id, :]
                batch_images.append(curr_batch_images)
                batch_labels.append(curr_batch_labels)

            last = time.time()  # 计时开始

            print("training epoch {}".format(epoch))
            time.sleep(0.01)
            for imgs, labels in tzip(batch_images, batch_labels):
                self.forward(imgs)
                self.backward(labels, lr)

            self.save_model(epoch)

            # lr = lr * 9 / 10
            duration = time.time() - last
            sum_time += duration

            str1 = "epoch {} finished, training time: {:.2f} seconds".format(epoch, duration)
            file.write(str1 + "\n")
            print(str1)

            test_accuracy, eval_time = self.evaluate(test_image, test_label)
            test_accuracies.append(test_accuracy)

            str2 = "epoch {}, test accuracy: {:.2%}, evaluation time: {:.2f} seconds".format(epoch, test_accuracy, eval_time)
            file.write(str2 + "\n")
            print(str2)

        if epochs != prev_epochs:
            avg_time = sum_time / (epochs - prev_epochs)

        return avg_time, test_accuracies

import numpy as np
import random
import matplotlib.pyplot as plt

from lenet import LeNet
from utils import *


def prepare_data(show_information: bool = False, show_an_image: bool = False):
    """
    Load data, normalize the images, and make the labels one-hot.

    Returns:
    - data: dict, where labels are one-hot
    """
    # load data
    with np.load('mnist.npz', allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]
    
    # show information about the data
    if show_information:
        print("x_train: ", x_train.shape, np.min(x_train), np.max(x_train))
        print("y_train: ", y_train.shape, np.min(y_train), np.max(y_train))
        print("x_test: ", x_test.shape, np.min(x_test), np.max(x_test))
        print("y_test: ", y_test.shape, np.min(y_test), np.max(y_test))
    
    # show an image
    if show_an_image:
        plt.imshow(x_train[5432], cmap="gray")
        plt.show()

    # format the data
    x_train = normalize_image(x_train)
    x_test = normalize_image(x_test)
    y_train = one_hot_labels(y_train)
    y_test = one_hot_labels(y_test)
    
    # pack as dict
    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }

    return data


def train_model(data: dict, prev_epochs: int = 0, total_epochs: int = 10):
    """
    Train the LeNet model.

    Args:
    - data: dict, where labels are one-hot
    - prev_epochs: how many epochs have been trained before
    - total_epochs: total number of epochs to train

    Returns:
    - net: the trained LeNet model
    """
    x_train, y_train, x_test, y_test = data.values()
    net = LeNet()

    if prev_epochs > 0:
        print("continue from the last epoch")
        net.load_model(prev_epochs)
    else:
        net.save_model(0)
        print("train from the beginning")

    avg_time, acc_list = net.fit(
        x_train, y_train, x_test, y_test, prev_epochs, epochs=total_epochs, batch_size=32, lr=1e-3
    )

    best_acc = np.max(acc_list)
    final_acc = acc_list[-1]
    print("best accuracy {:.2%}, final accuracy {:.2%}; average training time {:.2f} seconds"
          .format(best_acc, final_acc, avg_time))
    
    return net


def simply_load_model(epoch: int = 10):
    """
    Load the LeNet model from a local file.

    Args:
    - epoch: the model to load

    Returns:
    - net: the model
    """
    net = LeNet()
    net.load_model(epoch)
    return net


def simply_evaluate(data: dict, net: LeNet):
    """
    Evaluate the model with the test set.

    Args:
    - data: dict, where labels are one-hot
    - net: the model
    """
    images = data["x_test"]
    labels = data["y_test"]
    accuracy, eval_time = net.evaluate(images, labels)
    print("test accuracy {:.2%}, evaluation time {:.2f}".format(accuracy, eval_time))


def demo_some_results(data: dict, net: LeNet, demo_set: str = "test"):
    """
    Pick some images and demo predictons.

    Args:
    - data: dict, where labels are one-hot
    - net: the model
    - demo_set: from which set the images are picked
    - demo_id: 
    """
    assert demo_set in ["train", "test"], demo_set
    if demo_set == "train":
        images = data["x_train"]
        labels = data["y_train"]
    else:
        images = data["x_test"]
        labels = data["y_test"]
    
    max_id = images.shape[0] - 1
    num = 3
    demo_id = []
    while len(demo_id) < num:
        demo_id.append(random.randint(0, max_id))

    demo_img = []
    demo_gt = []
    
    for idx in demo_id:
        plt.figure()
        plt.imshow(images[idx], cmap="gray")
        plt.title("demo: " + demo_set + " image id " + str(idx))
        plt.show()
        demo_img.append(images[idx])
        demo_gt.append(labels[idx])

    demo_img = np.array(demo_img)  # (B, H, W)
    demo_gt = np.array(demo_gt)  # (B, 10), one-hot
    demo_gt = distribution_to_num(demo_gt)  # (B,)
    demo_pred = net.predict(demo_img)  # (B,)

    for n in range(len(demo_gt)):
        print("demo: " + demo_set + " image id {}, predicted to be {}, ground truth is {}"
              .format(demo_id[n], demo_pred[n], demo_gt[n]))


def main():
    data = prepare_data()
    net = train_model(data, prev_epochs=0, total_epochs=10)
    # net = simply_load_model(epoch=10)
    # simply_evaluate(data, net)
    demo_some_results(data, net)


if __name__ == "__main__":
    main()

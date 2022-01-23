import numpy as np


def normalize_image(images):
    """ 对图像做归一化处理 """
    return images * (2 / 255) - 1


def one_hot_labels(labels):
    """
    将labels 转换成 one-hot向量
    eg:  label: 3 --> [0,0,0,1,0,0,0,0,0,0]
    """
    num_images = labels.size  # 60000
    num_categories = np.max(labels) + 1  # 10
    one_hot = np.zeros([num_images, num_categories])
    for k in range(num_images):
        one_hot[k][labels[k]] = 1
    return one_hot


def find_first_pos(tgt, x):
    """
    Args:
    - tgt: the number to find
    - x: 1D array

    Returns:
    - pos: the position where `tgt` appears the first time in `x`
    """
    assert x.ndim == 1, x.ndim
    for k in range(x.shape[0]):
        if x[k] == tgt:
            return k
    
    return -1


def opt_val_and_pos(x, is_max: bool = True):
    """
    Find maximum or minimum and its first position.
    Conducted in each row if `x` is 2D.

    Args:
    - x: 1D or 2D array. if 2D, the max or min axis is 1
    - is_max: bool, indicating max or min

    Returns:
    - m_val: max or min value
    - m_pos: the position where max or min value is met the first time
    """
    assert x.ndim in [1, 2], x.ndim
    if x.ndim == 1:
        m_val = np.max(x) if is_max else np.min(x)
        m_pos = find_first_pos(m_val, x)
    else:
        m_val = np.max(x, axis=1) if is_max else np.min(x, axis=1)
        m_pos = np.zeros(x.shape[0]).astype(int)
        for k in range(x.shape[0]):
            m_pos[k] = find_first_pos(m_val[k], x[k, :])
    
    return m_val, m_pos


def distribution_to_num(x):
    """
    Convert distributions to numbers.

    Args:
    - x: 2D array, each row a distribution

    Returns:
    - y: 1D array, the position where maximum appears the first time
    """
    assert (x >= 0).all() and (abs(np.sum(x, axis=1) - 1) < 1e-5).all(), "Not a distribution"
    _, y = opt_val_and_pos(x)

    return y

import os
import gc
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Cap low-level library threading before importing heavy deps to avoid oversubscribing
# Allow overriding via DATA_WORKERS env for easier debugging or tighter CPU limits.
DEFAULT_WORKERS = max(1, int(os.environ.get("DATA_WORKERS", min(4, (os.cpu_count() or 1)))))
os.environ.setdefault("OMP_NUM_THREADS", str(DEFAULT_WORKERS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(DEFAULT_WORKERS))
os.environ.setdefault("MKL_NUM_THREADS", str(DEFAULT_WORKERS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(DEFAULT_WORKERS))

import pickle
import numpy as np
import psutil
import scipy
import scipy.io as scio
from sklearn.model_selection import train_test_split
from utils.dataset import ToDataLoader, set_seed
from utils.preprocess import preprocessing


# CPU kernel limitation
def LimitCpu():
    # 获取可用的CPU核心数
    num_cores = psutil.cpu_count(logical=False)
    available_memory = psutil.virtual_memory().available
    # 假设使用32位浮点数(4bytes),1/16减少内存开销占用
    max_chunk_size = int((available_memory // (4 * num_cores)) // 16)
    return max_chunk_size


global max_chunk_size
max_chunk_size = LimitCpu()


def _prepare_loaders(tx, vx, test_x, ts, vs, test_y, batchsize, pin_memory):
    tx, vx, test_x = [np.expand_dims(x, axis=1) for x in [tx, vx, test_x]]
    loaders = [
        ToDataLoader(x, y, mode, batch_size=batchsize, pin_memory=pin_memory)
        for x, y, mode in zip([tx, vx, test_x], [ts, vs, test_y], ["train", "test", "test"])
    ]
    return loaders


def _process_in_chunks(data, processor):
    processed_data = np.empty_like(data)
    chunk_size = min(data.shape[0], max_chunk_size)
    with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as executor:
        futures = []
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i : i + chunk_size]
            future = executor.submit(processor, chunk)
            futures.append((i, future))
        for i, future in futures:
            processed_data[i : i + chunk_size] = future.result()
    return processed_data


# Get Filter-bank EEG
def SubBandSplit(data: np.ndarray, freq_start: int = 4, freq_end: int = 40, bandwidth: int = 4, fs: int = 250):
    """
    优化后的子带切分函数
    data(batch,channel,time) --> sub_band_data(batch,(channel*nBands),time)
    """

    @lru_cache(maxsize=32)
    def get_sos_coeffs(freq_low, freq_high, fs):
        """缓存并返回 SOS 滤波器系数"""
        return scipy.signal.butter(6, [2.0 * freq_low / fs, 2.0 * freq_high / fs], "bandpass", output="sos")

    def process_single_band(args):
        """处理单个频带的数据"""
        data, freq_low, freq_high = args
        sos = get_sos_coeffs(freq_low, freq_high, fs)
        return scipy.signal.sosfilt(sos, data, axis=-1)

    subbands = np.arange(freq_start, freq_end + 1, bandwidth)
    with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as executor:
        band_args = [(data, low_freq, high_freq) for low_freq, high_freq in zip(subbands[:-1], subbands[1:])]
        results = list(executor.map(process_single_band, band_args))

    sub_band_data = np.stack(results, axis=1).astype(np.float32)
    del results
    gc.collect()
    # return rearrange(sub_band_data, 'b c t n -> b (c n) t')
    return sub_band_data


def GetLoader14xxx(seed, split: str = "001", batchsize: int = 64, pin_memory: bool = True):
    data = scio.loadmat(f"/mnt/data1/tyl/UserID/dataset/mydata/ori_{split}.mat")
    data1, label1 = data["ori_train_x"], data["ori_train_s"]
    data2, label2 = data["ori_test_x"], data["ori_test_s"]
    label1, label2 = [label.squeeze() for label in [label1, label2]]

    DataProcessor = preprocessing(fs=250)
    data1, data2 = [DataProcessor.EEGpipline(x) for x in [data1, data2]]
    tx, vx, ty, vy = train_test_split(data1, label1, test_size=0.2, random_state=seed, stratify=label1)

    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{data2.shape}")

    trainloader, validateloader, testloader = _prepare_loaders(
        tx,
        vx,
        data2,
        ty,
        vy,
        label2,
        batchsize,
        pin_memory,
    )
    del data1, data2, label1, label2, tx, vx, ty, vy
    gc.collect()
    return trainloader, validateloader, testloader


def GetloaderLJ30(seed, batchsize: int = 64, pin_memory: bool = True):
    data = scio.loadmat("/mnt/data1/tyl/UserID/dataset/mydata/ori_LingJiu30.mat")
    data1, data2 = data["ori_train_x"], data["ori_test_x"]
    data1, data2 = [x.reshape((-1, x.shape[-2], x.shape[-1])) for x in [data1, data2]]
    label1, label2 = data["ori_train_s"], data["ori_test_s"]
    label1, label2 = [s.reshape(-1) for s in [label1, label2]]

    DataProcessor = preprocessing(fs=300)
    data1, data2 = [DataProcessor.EEGpipline(x) for x in [data1, data2]]
    tx, vx, ts, vs = train_test_split(data1, label1, test_size=0.2, random_state=seed, stratify=label1)

    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{data2.shape}")

    trainloader, validateloader, testloader = _prepare_loaders(
        tx,
        vx,
        data2,
        ts,
        vs,
        label2,
        batchsize,
        pin_memory,
    )
    del data1, data2, label1, label2, tx, vx, ts, vs
    gc.collect()
    return trainloader, validateloader, testloader


# MI:(54,200,62,4000) --> downsample: (54,200,62,1000)
# SSVEP:(54,200,62,4000) --> downsample : (54,200,62,1000)
# ERP:(54,4140,62,800) --> downsample : (54,200,62,800)
def GetLoaderOpenBMI(seed, Task: str = "MI", batchsize: int = 64, is_task: bool = True, pin_memory: bool = True):
    def load_data(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    if is_task:
        data_train = load_data(f"/mnt/data1/tyl/data/OpenBMI/Task/{Task}/train.pkl")
        data_test = load_data(f"/mnt/data1/tyl/data/OpenBMI/Task/{Task}/test.pkl")
        train_x = data_train["data"].astype(np.float32)
        test_x = data_test["data"].astype(np.float32)
        train_y = data_train["label"].astype(np.int16)
        test_y = data_test["label"].astype(np.int16)
    else:
        data_train = load_data(f"/mnt/data1/tyl/data/OpenBMI/processed/{Task}/train.pkl")
        data_test = load_data(f"/mnt/data1/tyl/data/OpenBMI/processed/{Task}/test.pkl")
        train_x = data_train["ori_train_x"].astype(np.float32)
        test_x = data_test["ori_test_x"].astype(np.float32)
        train_y = (data_train["ori_train_s"] - 1).astype(np.int16)
        test_y = (data_test["ori_test_s"] - 1).astype(np.int16)
        train_x, test_x = [x.reshape((-1, x.shape[-2], x.shape[-1])) for x in [train_x, test_x]]
        train_y, test_y = [s.reshape(-1) for s in [train_y, test_y]]

    train_x, test_x = [x.reshape((-1, x.shape[-2], x.shape[-1])) for x in [train_x, test_x]]

    fs = 1000 if Task == "ERP" else 250
    DataProcessor = preprocessing(fs=fs)
    processor = DataProcessor.EEGpipline

    train_x, test_x = [_process_in_chunks(x, processor) for x in [train_x, test_x]]
    tx, vx, ts, vs = train_test_split(train_x, train_y, test_size=0.2, random_state=seed, stratify=train_y)

    print("-----数据预处理完成-----")
    print(f"是否任务分类: {is_task}, 类别数量: {len(np.unique(train_y))}")
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{test_x.shape}")

    trainloader, validateloader, testloader = _prepare_loaders(
        tx,
        vx,
        test_x,
        ts,
        vs,
        test_y,
        batchsize,
        pin_memory,
    )
    del data_train, data_test, train_x, train_y, test_x, test_y, tx, vx, ts, vs
    gc.collect()
    return trainloader, validateloader, testloader


"""
Clibration#---------------------------------------------- Test
Rest (600, 65, 1000) (600,) -- 20 * 30 (subs * trials)
Transient (1791, 65, 1000) (1791,) -- 20 * (88~90)
Steady (740, 65, 1000) (740,) -- 20 * 37
P300 (299, 65, 1000) (299,) -- 20 * (15~14)
Motor (2400, 65, 1000) (2400,) -- 20 * 120
SSVEP_SA (240, 65, 1000) (240,) -- 20 * 12

Partial Enrollment#---------------------------------------------- Real Train
Rest (1200, 65, 1000) (1200,) -- 20 * 60 (subs * trials)
Transient (3590, 65, 1000) (3590,) -- 20 * (178~180)
Steady (1530, 65, 1000) (1530,) -- 20 * (75/105)
P300 (599, 65, 1000) (599,) -- 20 * (29~30)
Motor (4828, 65, 1000) (4828,) -- 20* (239~243 / 265)
SSVEP_SA (480, 65, 1000) (480,) -- 20 * 24
! 注意: 去除EasyCap后是64通道
"""
# Task = "Rest", "Transient", "Steady", "Motor"
def GetLoaderM3CV(seed, Task: str = "Rest", batchsize: int = 64, is_task: bool = True, pin_memory: bool = True):
    def load_data(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    if is_task:
        data_train = load_data(f"/mnt/data1/tyl/data/M3CV/Task/Session1_{Task}.pkl")
        data_test = load_data(f"/mnt/data1/tyl/data/M3CV/Task/Session2_{Task}.pkl")
        train_x = data_train["data"][:, :-1, :].astype(np.float32)
        test_x = data_test["data"][:, :-1, :].astype(np.float32)
        train_y = data_train["label"].astype(np.int16)
        test_y = data_test["label"].astype(np.int16)
    else:
        data_train = load_data(f"/mnt/data1/tyl/data/M3CV/Train/T_{Task}.pkl")
        data_test = load_data(f"/mnt/data1/tyl/data/M3CV/Test/{Task}.pkl")
        train_x = data_train["data"][:, :-1, :].astype(np.float32)
        test_x = data_test["data"][:, :-1, :].astype(np.float32)
        train_y = data_train["label"].astype(np.int16)
        test_y = data_test["label"].astype(np.int16)

    DataProcessor = preprocessing(fs=250)
    processor = DataProcessor.EEGpipline

    train_x, test_x = [_process_in_chunks(x, processor) for x in [train_x, test_x]]
    tx, vx, ts, vs = train_test_split(train_x, train_y, test_size=0.2, random_state=seed, stratify=train_y)

    print("-----数据预处理完成-----")
    print(f"是否任务分类: {is_task}, 类别数量: {len(np.unique(train_y))}")
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{test_x.shape}")

    trainloader, validateloader, testloader = _prepare_loaders(
        tx,
        vx,
        test_x,
        ts,
        vs,
        test_y,
        batchsize,
        pin_memory,
    )
    del data_train, data_test, train_x, train_y, test_x, test_y, tx, vx, ts, vs
    gc.collect()
    return trainloader, validateloader, testloader


def Load_Dataloader(
    seed,
    dataset,
    paradigm=None,
    batchsize: int = 64,
    is_task: bool = True,
    pin_memory: bool = True,
):
    openbmi_tasks = ["MI", "SSVEP", "ERP"]
    m3cv_tasks = ["Rest", "Transient", "Steady", "P300", "Motor", "SSVEP_SA"]
    dataset_aliases = {
        "14001": "001",
        "001": "001",
        "14004": "004",
        "004": "004",
        "LJ30": "LJ30",
        "LingJiu30": "LJ30",
        "OpenBMI": "OpenBMI",
        "M3CV": "M3CV",
    }

    set_seed(seed)

    if paradigm is None:
        if dataset in openbmi_tasks:
            paradigm = dataset
            dataset = "OpenBMI"
        elif dataset in m3cv_tasks:
            paradigm = dataset
            dataset = "M3CV"

    dataset_key = dataset_aliases.get(dataset)

    if dataset_key in {"001", "004"}:
        trainloader, valloader, testloader = GetLoader14xxx(
            seed,
            split=dataset_key,
            batchsize=batchsize,
            pin_memory=pin_memory,
        )
    elif dataset_key == "LJ30":
        trainloader, valloader, testloader = GetloaderLJ30(
            seed,
            batchsize=batchsize,
            pin_memory=pin_memory,
        )
    elif dataset_key == "OpenBMI" and paradigm in openbmi_tasks:
        trainloader, valloader, testloader = GetLoaderOpenBMI(
            seed,
            Task=paradigm,
            batchsize=batchsize,
            is_task=is_task,
            pin_memory=pin_memory,
        )
    elif dataset_key == "M3CV" and paradigm in m3cv_tasks:
        trainloader, valloader, testloader = GetLoaderM3CV(
            seed,
            Task=paradigm,
            batchsize=batchsize,
            is_task=is_task,
            pin_memory=pin_memory,
        )
    else:
        raise ValueError(f"Invalid dataset or paradigm name: dataset={dataset}, paradigm={paradigm}")

    return trainloader, valloader, testloader

import os
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
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
import scipy.io as scio
from dataset import ToDataLoader, set_seed
from preprocess import preprocessing


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


def _ensure_session_array(sessions):
    sessions = np.asarray(sessions)
    if sessions.ndim != 1:
        sessions = sessions.reshape(-1)
    if np.issubdtype(sessions.dtype, np.number):
        return sessions.astype(int)
    return np.array([int(s) for s in sessions], dtype=int)


def _extract_subjects(data_dict, allow_label_fallback=False):
    subject_keys = ["subject", "subjects", "sub", "subj", "sub_id", "subject_id"]
    for key in subject_keys:
        if key in data_dict:
            return np.asarray(data_dict[key]).reshape(-1)
    if allow_label_fallback and "label" in data_dict:
        return np.asarray(data_dict["label"]).reshape(-1)
    return None


def _split_by_subject_session(subjects, sessions, seed, forget_subject=None):
    set_seed(seed)
    subjects = np.asarray(subjects).reshape(-1)
    sessions = _ensure_session_array(sessions)
    unique_subjects = np.unique(subjects)
    if unique_subjects.size == 0:
        raise ValueError("No subjects found for MU split.")
    if forget_subject is None:
        forget_subject = np.random.choice(unique_subjects)
    else:
        if forget_subject not in unique_subjects:
            raise ValueError(
                f"forget_subject={forget_subject} is not present in available subjects: {unique_subjects}"
            )

    mask_subject = subjects == forget_subject
    mask_session1 = sessions == 1
    mask_session2 = sessions == 2

    forget_train_idx = np.where(mask_subject & mask_session1)[0]
    forget_test_idx = np.where(mask_subject & mask_session2)[0]
    remain_train_idx = np.where(~mask_subject & mask_session1)[0]
    remain_test_idx = np.where(~mask_subject & mask_session2)[0]
    train_idx = np.concatenate([forget_train_idx, remain_train_idx])

    return (
        forget_subject,
        remain_train_idx,
        forget_train_idx,
        remain_test_idx,
        forget_test_idx,
        train_idx,
    )


def _make_loader(data, labels, mode, batchsize, pin_memory, shuffle=None):
    data = np.expand_dims(data, axis=1)
    return ToDataLoader(
        data,
        labels,
        mode,
        batch_size=batchsize,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


def _build_mu_loaders(
    data,
    labels,
    subjects,
    sessions,
    seed,
    batchsize,
    pin_memory,
    forget_subject=None,
):
    (
        forget_subject,
        remain_train_idx,
        forget_train_idx,
        remain_test_idx,
        forget_test_idx,
        train_idx,
    ) = _split_by_subject_session(subjects, sessions, seed, forget_subject=forget_subject)

    train_loader = _make_loader(
        data[train_idx],
        labels[train_idx],
        mode="train",
        batchsize=batchsize,
        pin_memory=pin_memory,
        shuffle=True,
    )
    test_loader_remain = _make_loader(
        data[remain_test_idx],
        labels[remain_test_idx],
        mode="test",
        batchsize=batchsize,
        pin_memory=pin_memory,
        shuffle=False,
    )
    test_loader_forget = _make_loader(
        data[forget_test_idx],
        labels[forget_test_idx],
        mode="test",
        batchsize=batchsize,
        pin_memory=pin_memory,
        shuffle=False,
    )
    remain_train_loader = _make_loader(
        data[remain_train_idx],
        labels[remain_train_idx],
        mode="train",
        batchsize=batchsize,
        pin_memory=pin_memory,
        shuffle=True,
    )
    forget_train_loader = _make_loader(
        data[forget_train_idx],
        labels[forget_train_idx],
        mode="train",
        batchsize=batchsize,
        pin_memory=pin_memory,
        shuffle=True,
    )

    loaders = {
        "train_loader": train_loader,
        "test_loader_remain": test_loader_remain,
        "test_loader_forget": test_loader_forget,
        "remain_train_loader": remain_train_loader,
        "forget_train_loader": forget_train_loader,
        "forget_subject": forget_subject,
    }
    return loaders


def GetMULoader14xxx(
    seed,
    split: str = "001",
    is_task: bool = True,
    batchsize: int = 64,
    pin_memory: bool = True,
    forget_subject=None,
):
    data = scio.loadmat(f"/mnt/data1/tyl/UserID/dataset/mydata/ori_{split}.mat")
    data1, data2 = data["ori_train_x"], data["ori_test_x"]
    if is_task:
        label1, label2 = data["ori_train_y"], data["ori_test_y"]
    else:
        label1, label2 = data["ori_test_x"], data["ori_test_s"]
    label1, label2 = [label.squeeze() for label in [label1, label2]]

    subject1 = None
    subject2 = None
    for key in ["ori_train_s", "train_s", "subject_train", "subjects_train"]:
        if key in data:
            subject1 = np.asarray(data[key]).reshape(-1)
            break
    for key in ["ori_test_s", "test_s", "subject_test", "subjects_test"]:
        if key in data:
            subject2 = np.asarray(data[key]).reshape(-1)
            break

    if subject1 is None or subject2 is None:
        raise ValueError("Subject metadata is required for MU splitting but was not found in the .mat file.")

    DataProcessor = preprocessing(fs=250)
    data1, data2 = [DataProcessor.EEGpipline(x) for x in [data1, data2]]

    data_all = np.concatenate([data1, data2], axis=0)
    label_all = np.concatenate([label1, label2], axis=0)
    subject_all = np.concatenate([subject1, subject2], axis=0)
    session_all = np.concatenate([np.ones(len(label1)), np.ones(len(label2)) * 2], axis=0)

    print(f"MU split: dataset={split}, forget-subject selected from {len(np.unique(subject_all))} subjects")

    loaders = _build_mu_loaders(
        data_all,
        label_all,
        subject_all,
        session_all,
        seed,
        batchsize,
        pin_memory,
        forget_subject=forget_subject,
    )
    del data1, data2, label1, label2, data_all, label_all, subject_all, session_all
    gc.collect()
    return loaders


def GetMULoaderOpenBMI(
    seed,
    Task: str = "MI",
    batchsize: int = 64,
    is_task: bool = True,
    pin_memory: bool = True,
    forget_subject=None,
):
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
        train_subject = _extract_subjects(data_train, allow_label_fallback=False)
        test_subject = _extract_subjects(data_test, allow_label_fallback=False)
    else:
        data_train = load_data(f"/mnt/data1/tyl/data/OpenBMI/processed/{Task}/train.pkl")
        data_test = load_data(f"/mnt/data1/tyl/data/OpenBMI/processed/{Task}/test.pkl")
        train_x = data_train["ori_train_x"].astype(np.float32)
        test_x = data_test["ori_test_x"].astype(np.float32)
        train_y = (data_train["ori_train_s"] - 1).astype(np.int16)
        test_y = (data_test["ori_test_s"] - 1).astype(np.int16)
        train_x, test_x = [x.reshape((-1, x.shape[-2], x.shape[-1])) for x in [train_x, test_x]]
        train_y, test_y = [s.reshape(-1) for s in [train_y, test_y]]
        train_subject = _extract_subjects(data_train, allow_label_fallback=True)
        test_subject = _extract_subjects(data_test, allow_label_fallback=True)

    if train_subject is None or test_subject is None:
        raise ValueError("Subject metadata is required for MU splitting but was not found in OpenBMI data.")

    train_x, test_x = [x.reshape((-1, x.shape[-2], x.shape[-1])) for x in [train_x, test_x]]

    fs = 1000 if Task == "ERP" else 250
    DataProcessor = preprocessing(fs=fs)
    processor = DataProcessor.EEGpipline

    train_x, test_x = [_process_in_chunks(x, processor) for x in [train_x, test_x]]

    data_all = np.concatenate([train_x, test_x], axis=0)
    label_all = np.concatenate([train_y, test_y], axis=0)
    subject_all = np.concatenate([train_subject, test_subject], axis=0)
    session_all = np.concatenate([np.ones(len(train_y)), np.ones(len(test_y)) * 2], axis=0)

    print("-----MU数据预处理完成-----")
    print(f"是否任务分类: {is_task}, 类别数量: {len(np.unique(label_all))}")

    loaders = _build_mu_loaders(
        data_all,
        label_all,
        subject_all,
        session_all,
        seed,
        batchsize,
        pin_memory,
        forget_subject=forget_subject,
    )
    del data_train, data_test, train_x, train_y, test_x, test_y, data_all, label_all, subject_all, session_all
    gc.collect()
    return loaders


def GetMULoaderM3CV(
    seed,
    Task: str = "Rest",
    batchsize: int = 64,
    is_task: bool = True,
    pin_memory: bool = True,
    forget_subject=None,
):
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
        train_subject = _extract_subjects(data_train, allow_label_fallback=False)
        test_subject = _extract_subjects(data_test, allow_label_fallback=False)
    else:
        data_train = load_data(f"/mnt/data1/tyl/data/M3CV/Train/T_{Task}.pkl")
        data_test = load_data(f"/mnt/data1/tyl/data/M3CV/Test/{Task}.pkl")
        train_x = data_train["data"][:, :-1, :].astype(np.float32)
        test_x = data_test["data"][:, :-1, :].astype(np.float32)
        train_y = data_train["label"].astype(np.int16)
        test_y = data_test["label"].astype(np.int16)
        train_subject = _extract_subjects(data_train, allow_label_fallback=True)
        test_subject = _extract_subjects(data_test, allow_label_fallback=True)

    if train_subject is None or test_subject is None:
        raise ValueError("Subject metadata is required for MU splitting but was not found in M3CV data.")

    DataProcessor = preprocessing(fs=250)
    processor = DataProcessor.EEGpipline

    train_x, test_x = [_process_in_chunks(x, processor) for x in [train_x, test_x]]

    data_all = np.concatenate([train_x, test_x], axis=0)
    label_all = np.concatenate([train_y, test_y], axis=0)
    subject_all = np.concatenate([train_subject, test_subject], axis=0)
    session_all = np.concatenate([np.ones(len(train_y)), np.ones(len(test_y)) * 2], axis=0)

    print("-----MU数据预处理完成-----")
    print(f"是否任务分类: {is_task}, 类别数量: {len(np.unique(label_all))}")

    loaders = _build_mu_loaders(
        data_all,
        label_all,
        subject_all,
        session_all,
        seed,
        batchsize,
        pin_memory,
        forget_subject=forget_subject,
    )
    del data_train, data_test, train_x, train_y, test_x, test_y, data_all, label_all, subject_all, session_all
    gc.collect()
    return loaders


def Load_MU_Dataloader(
    seed,
    dataset,
    paradigm=None,
    batchsize: int = 64,
    is_task: bool = True,
    pin_memory: bool = True,
    forget_subject=None,
):
    openbmi_tasks = ["MI", "SSVEP", "ERP"]
    m3cv_tasks = ["Rest", "Transient", "Steady", "P300", "Motor", "SSVEP_SA"]
    dataset_aliases = {
        "14001": "001",
        "001": "001",
        "14004": "004",
        "004": "004",
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
        loaders = GetMULoader14xxx(
            seed,
            split=dataset_key,
            batchsize=batchsize,
            pin_memory=pin_memory,
            is_task=is_task,
            forget_subject=forget_subject,
        )
    elif dataset_key == "OpenBMI" and paradigm in openbmi_tasks:
        loaders = GetMULoaderOpenBMI(
            seed,
            Task=paradigm,
            batchsize=batchsize,
            is_task=is_task,
            pin_memory=pin_memory,
            forget_subject=forget_subject,
        )
    elif dataset_key == "M3CV" and paradigm in m3cv_tasks:
        loaders = GetMULoaderM3CV(
            seed,
            Task=paradigm,
            batchsize=batchsize,
            is_task=is_task,
            pin_memory=pin_memory,
            forget_subject=forget_subject,
        )
    else:
        raise ValueError(f"Invalid dataset or paradigm name: dataset={dataset}, paradigm={paradigm}")

    return loaders

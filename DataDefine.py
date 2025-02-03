import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from utils import print_log, NoneScaler
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split, Subset, ConcatDataset
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_datloader(args, mode = "train", infer_num = [-1]):
    """Generate dataloader"""
    last_num = args.data_after_num + 1
    if args.model_type in ["AE","AE_Conv"]:
        dataset = AE_Dataset(args)
    elif  args.model_type in ["DON", "EN_DON"]:
        dataset = DON_Dataset(args)
        last_num = 1
    else:
        dataset = CFD_Dataset(args)
    data_scaler = dataset.scaler if dataset.scaler else None

    if mode == "inference":
        inference_dataset = Subset(dataset, infer_num) # B, T, H, W
        try:
            orgs_data = dataset.orgs[infer_num]
        except:
            orgs_data = None
        infer_loader = DataLoader(inference_dataset,
                                num_workers=args.num_workers,
                                batch_size=args.per_device_valid_batch_size,
                                pin_memory=True,
                                shuffle = False,
                                drop_last=False)
        print_log(f"Length of inference_dataset: {len(inference_dataset)}")
        print_log(f"Shape of input_data: {inference_dataset[0][0].shape}")
        print_log(f"Shape of label_data: {inference_dataset[0][1].shape}")
        return infer_loader, data_scaler, dataset.x_mesh, dataset.y_mesh, orgs_data
    
    # Split dataset into training dataset, validation dataset and test_dataset
    indices = list(range(len(dataset)))

    if (args.test_ratio > 1) or (args.test_ratio < 0):
        print_log(f"Errot test_ratio!")
        raise EOFError

    dataset_rm_last = Subset(dataset, indices[:-last_num])
    last_data_dataset = Subset(dataset, indices[-last_num:])

    testlen = int(args.test_ratio * len(dataset)) - last_num 
    validlen = int(args.valid_ratio * len(dataset))
    trainlen = len(dataset_rm_last) - testlen - validlen

    if args.valid_ratio > 0:
        lengths = [trainlen, validlen, testlen]
        train_dataset, valid_dataset, test_dataset = random_split(dataset_rm_last, lengths)
        test_indices = test_dataset.indices + indices[-last_num:]
        test_dataset = ConcatDataset([test_dataset, last_data_dataset])
    else:
        lengths = [trainlen, testlen]
        train_dataset, test_dataset = random_split(dataset_rm_last, lengths)
        test_indices = test_dataset.indices + indices[-last_num:]
        test_dataset = ConcatDataset([test_dataset, last_data_dataset])
        valid_dataset = test_dataset

    try:
        orgs_data = dataset.orgs[test_indices]
    except:
        orgs_data = None

    if args.init:
        print_log(f"Length of all dataset: {len(dataset)}")
        print_log(f"Length of train_dataset: {len(train_dataset)}")
        print_log(f"Length of valid_dataset: {len(valid_dataset)}")
        print_log(f"Length of test_dataset: {len(test_dataset)}")
        print_log(f"Shape of input_data: {test_dataset[0][0].shape}")
        print_log(f"Shape of label_data: {test_dataset[0][1].shape}")

    # DataLoaders creation:
    if not args.dist:
        train_sampler = RandomSampler(train_dataset)
        vaild_sampler = SequentialSampler(valid_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        vaild_sampler = DistributedSampler(valid_dataset)

    train_loader = DataLoader(train_dataset,
                                sampler=train_sampler,
                                num_workers=args.num_workers,
                                drop_last=args.drop_last,
                                pin_memory=True,
                                batch_size=args.per_device_train_batch_size)
    vali_loader = DataLoader(valid_dataset,
                                sampler=vaild_sampler,
                                num_workers=args.num_workers,
                                drop_last=False,
                                pin_memory=True,
                                batch_size=args.per_device_valid_batch_size)
    test_loader = DataLoader(test_dataset,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                batch_size=args.per_device_valid_batch_size,
                                shuffle = False,
                                drop_last=False)
    return train_loader, vali_loader, test_loader, data_scaler, dataset.x_mesh, dataset.y_mesh, orgs_data


def get_lam_datloader(args, inputs, labels, scaler):
    dataset = lam_Dataset(inputs, labels, scaler)
    loader = DataLoader(dataset,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                batch_size=args.per_device_valid_batch_size,
                                shuffle = False,
                                drop_last=False)
    return loader


def get_scaler(args, num):
    if args.data_scaler[num] == "Standard":
        scaler = StandardScaler()
        scaler.mean_= args.data_mean[num]
        scaler.scale_ = args.data_std[num]
    elif args.data_scaler[num] == "MinMax":
        scaler= MinMaxScaler()
        custom_min = 0.0
        custom_max = 1.0
        scale = (custom_max - custom_min) / (args.data_max[num] - args.data_min[num])
        scaler.scale_ = scale
        scaler.min_= custom_min - args.data_min[num] * scale
    elif args.data_scaler[num] == "None":
        scaler = NoneScaler()
    elif args.data_scaler[num] == "BCT":
        scaler = PowerTransformer()
    else:
        print_log(f"Error data_scaler type: {args.data_scaler[num]}")
        raise EOFError
    return scaler



class CFD_Dataset(Dataset):
    ''' Dataset for loading and preprocessing the CFD data '''
    def __init__(self, args):
        # Read inputs
        self.height, self.width = args.data_shape
        self.data_after = args.data_after
        self.data_after_num = args.data_after_num
        self.data_num = args.data_num
        self.all = [0] + self.data_after

        mesh = np.load(args.mesh_path)
        data_matrix = np.load(args.data_path)[:self.data_num, self.all]
        self.y_mesh, self.x_mesh = mesh[0], mesh[1]
        inputs = data_matrix[:, 0]
        labels = data_matrix[:, 1:]
        
        # Data Standard
        self.scaler = []
        for i in range(len(args.data_scaler)):
            self.scaler.append(get_scaler(args, i))

        #inputs (num, H, W)
        inputs = inputs.reshape(-1, self.height * self.width)
        if args.data_scaler[0] == "BCT":
            self.scaler[0].fit(inputs)
        inputs = self.scaler[0].transform(inputs)

        #inputs (num, 1, H, W)
        self.inputs = inputs.reshape(self.data_num, 1, self.height, self.width)


        #labels (num, after, H, W)
        labels = labels.reshape(-1, self.height * self.width)
        if args.data_scaler[1] == "BCT":
            self.scaler[1].fit(labels)
        labels = self.scaler[1].transform(labels)

        self.labels = labels.reshape(self.data_num, self.data_after_num, self.height, self.width)


        del mesh
        del inputs, labels


    def __getitem__(self, index):
        # Convert Data into PyTorch tensors

        input = torch.Tensor(self.inputs[index])
        label = torch.Tensor(self.labels[index])
        return input, label


    def __len__(self):
        # Returns the size of the dataset
        return self.data_num


class AE_Dataset(Dataset):
    ''' Dataset for loading and preprocessing the AE data '''
    def __init__(self, args):
        # Read inputs
        self.height, self.width = args.data_shape
        self.data_after = args.data_after
        self.data_after_num = args.data_after_num
        self.data_num = args.data_num
        self.all = [0] + self.data_after
        mesh = np.load(args.mesh_path)
        data_matrix = np.load(args.data_path)[:self.data_num, self.all]
        self.y_mesh, self.x_mesh = mesh[0], mesh[1]

        # Data Standard
        self.scaler = [get_scaler(args, 0)]

        # Reshape data: (num, data_time, height, width) -> (num * data_time, height * width)
        label_data = data_matrix
        label_data = label_data.reshape(self.data_num * (self.data_after_num+1), -1)
        label_data = self.scaler[-1].transform(label_data)
        label_data = label_data.reshape(-1, 1, self.height, self.width)
        
        self.data_matrix = label_data # (num * data_time, height, width)

        del mesh
        del label_data, data_matrix


    def __getitem__(self, index):
        # Convert Data into PyTorch tensors
        inputs_labels = self.data_matrix[index]        #(height, width)
        inputs_labels = torch.Tensor(inputs_labels)
        return inputs_labels, inputs_labels


    def __len__(self):
        # Returns the size of the dataset
        return self.data_num * (self.data_after_num+1)


class DON_Dataset(Dataset):
    ''' Dataset for loading and preprocessing the DON or EN_DON data '''
    def __init__(self, args):
        # Read inputs
        self.input_shape = args.input_shape
        self.latent_dim = args.latent_dim
        self.data_after = args.data_after
        self.data_after_num = args.data_after_num
        self.data_num = args.data_num
        self.all = [0] + self.data_after
        inputs = np.load(args.input_path)[:self.data_num]
        lams = np.load(args.lam_path)[:self.data_num, [i-1 for i in self.data_after]]
        orgs = np.load(args.org_path)[:self.data_num, self.all]

        mesh = np.load(args.mesh_path)
        self.y_mesh, self.x_mesh = mesh[0], mesh[1]

        # Data Standard
        self.scaler = []
        for i in range(len(args.data_scaler)):
            self.scaler.append(get_scaler(args, i))

        #inputs (num, lam) or (num, H, W) 
        inputs = inputs.reshape(self.data_num, -1)
        if args.data_scaler[0] == "BCT":
            self.scaler[0].fit(inputs)
        self.inputs = self.scaler[0].transform(inputs)

        #inputs (num, 1, q, q) or (num, 1, H, W)
        self.inputs = self.inputs.reshape(self.data_num, 1, self.input_shape[0], self.input_shape[1])

        #labels (num, after, lam)
        lams = lams.reshape(self.data_num * self.data_after_num, self.latent_dim)
        if args.data_scaler[2] == "BCT":
            self.scaler[2].fit(lams)
        lams = self.scaler[2].transform(lams)
        self.labels = lams.reshape(self.data_num, self.data_after_num, self.latent_dim)

        #orgs (num, after+1, H, W)
        self.orgs = orgs

        del mesh
        del inputs, lams, orgs


    def __getitem__(self, index):
        # Convert Data into PyTorch tensors

        input = torch.Tensor(self.inputs[index])
        label = torch.Tensor(self.labels[index])
        return input, label


    def __len__(self):
        # Returns the size of the dataset
        return self.data_num


class lam_Dataset(Dataset):
    ''' Dataset for loading and preprocessing the lam data '''
    def __init__(self, inputs, labels, scaler):
        #input (num, after, lam)
        #label (num, after, H, W)
        self.lam = inputs.shape[2]
        self.num, self.after, self.H, self.W = labels.shape
        
        inputs = inputs.reshape(-1, self.lam)
        inputs = scaler[2].inverse_transform(inputs)
        self.inputs = inputs #input (num * after, lam)

        labels = labels.reshape(-1, self.H * self.W)
        labels = scaler[1].transform(labels)
        labels = labels.reshape(-1, self.H, self.W) #label (num * after, H, W)

        self.labels = labels


    def __getitem__(self, index):
        # Convert Data into PyTorch tensors

        input = torch.Tensor(self.inputs[index])
        label = torch.Tensor(self.labels[index])
        return input, label


    def __len__(self):
        # Returns the size of the dataset
        return self.num*self.after


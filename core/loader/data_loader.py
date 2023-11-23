import os
import nibabel as nib
from torch.utils.data import Dataset


class ATLAS2_Train_Dataset(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.n_classes = 2
        self.train_files = []
        self.train_files_labels = []

        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.nii.gz'):
                    if 'label' not in file:
                        self.train_files.append(os.path.join(root, file))
                    else:
                        self.train_files_labels.append(os.path.join(root, file))
        
        self.train_files = sorted(self.train_files)
        self.train_files_labels = sorted(self.train_files_labels)


    def __getitem__(self, index):
        file_name = self.train_files[index]
        file_name_label = self.train_files_labels[index]

        nii_img = nib.load(file_name)
        nii_img_label = nib.load(file_name_label)
        nii_data = nii_img.get_fdata()
        nii_data_label = nii_img_label.get_fdata()

        nii_data = nii_data.reshape(
            1, nii_data.shape[2], nii_data.shape[0], nii_data.shape[1]
        )
        nii_data_label = nii_data_label.reshape(
            1, nii_data_label.shape[2], nii_data_label.shape[0], nii_data_label.shape[1]
        )

        return nii_data, nii_data_label


    def __len__(self):
        return len(self.train_files)
    

    def get_class_weights(self):
        return [0.50192237, 130.54794746]
    

    def get_n_classes(self):
        return self.n_classes


class ATLAS2_Test_Dataset(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.n_classes = 2
        self.test_files = []
        
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.nii.gz'):
                    self.test_files.append(os.path.join(root, file))


    def __getitem__(self, index):
        pass


    def __len__(self):
        return len(self.test_files)

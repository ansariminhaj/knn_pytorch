import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

CUTOFF = 0.01

#Repetition of code (lines 74-146)
def assign_labels(data_set, lab_values, file_dict):
    labels = []
    voltages = []
    labels_raw = []

    excluded = []
    failed = []
    
    for mrn in tqdm(data_set.values):
        for key in lab_values[lab_values['mrn'] == mrn].index:
            try:
                label = float(lab_values.loc[key,'disease'])
    #             this_hr = get_hr(file_dict[key][0])
    #             if not math.isnan(this_hr): 
                this_ecg = file_dict[key][0]
                if label>=0.0 and np.max(this_ecg)<15000 and np.min(this_ecg)>-15000:# and key not in uuids_exclude.values:


                    voltages.append(file_dict[key][0])

                    if label < CUTOFF:
                        labels.append([0])
                    else:
                        labels.append([1])
                    labels_raw.append([label])
    #                 val_labels.append(this_hr)
                else:
                    excluded.append(key)
                    pass
            except:
                failed.append(key)
                pass

    return labels, voltages, labels_raw, failed, excluded
    
#Add filenames in parameters rather than in the code. Easier to locate and change (if filename changes) (30 and 49)
def load_data(lab_value, run_num = 0, file_dict=None, npy_file='mvp.npy', csv_file='cve_uuid_mrn.csv'):
    
    if file_dict is None:
        file_dict = np.load(npy_file,allow_pickle=True,encoding='latin1')
    #     file_dict = np.load('trop_cath_dict.npy',allow_pickle=True,encoding='latin1')
    #     file_dict = np.load('k_ser_all_dict.npy',allow_pickle=True,encoding='latin1')
    #     file_dict = np.load('potassium_serum_dict.npy',allow_pickle=True,encoding='latin1')
        file_dict = file_dict.item()

    train_voltages = []
    train_labels = []
    val_voltages = []
    val_labels = []
    test_voltages = []
    test_labels = []
    
    train_labels_raw = []
    val_labels_raw = []
    test_labels_raw = []

    train_dems = [] # [gender, age, weight, height, bsa]
    val_dems = []
    test_dems = []

    train_failed = []
    train_excluded = []
    test_failed = []
    test_excluded = []
    val_failed = []
    val_excluded = []
    
    lab_values = pd.read_csv(csv_file)

    mrns = lab_values['mrn'].drop_duplicates()
    lab_values = lab_values.set_index('UUID')
    mrn_train,mrn_not_train = train_test_split(mrns,test_size=config['test_size'], random_state= 4* (run_num+1))
    mrn_val, mrn_test = train_test_split(mrn_not_train,test_size=config['val_size'], random_state=4* (run_num+1))

    train_voltages, train_labels, train_labels_raw, train_failed, train_excluded = assign_labels(mrn_train, lab_values, file_dict)
    val_voltages, val_labels, val_labels_raw, val_failed, val_excluded = assign_labels(mrn_test, lab_values, file_dict)
    test_voltages, test_labels, test_labels_raw, test_failed, test_excluded = assign_labels(mrn_val, lab_values, file_dict)

    excluded = train_excluded + val_excluded + test_excluded
    failed = train_failed + val_failed + test_failed

    x_train = np.array(train_voltages)
    y_train = np.array(train_labels)

    x_val = np.array(val_voltages)
    y_val = np.array(val_labels)
#     y_val = (np.array(val_labels).reshape((-1,1))  - labels_min) / (labels_max - labels_min)
#     y_val = (np.array(val_labels).reshape((-1,1))  - labels_mean) / (labels_std)

#     x_test = np.swapaxes(test_voltages,1,2)
#     x_test = ( ( np.clip(test_voltages, global_min, global_max) - global_min ) / (global_max - global_min) ).reshape((-1,2500,12))
    x_test = np.array(test_voltages)
    y_test = np.array(test_labels)

    return x_train, x_val, x_test, y_train, y_val, y_test, np.array(train_dems), np.array(val_dems), np.array(test_dems), train_labels_raw, val_labels_raw, test_labels_raw


#Dataset class needed to create dataloaders. All transformations take place here
class ImageDataset(Dataset):
  def __init__(self, np_array_of_images, np_array_of_labels, transform):
    self.transform = transform
    self.images = np_array_of_images
    self.labels = np_array_of_labels

  def __len__(self):
    return len(self.np_array_of_images)
 
  def __getitem__(self,index):
    image=self.images.iloc[index]
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    image=self.transform(image)
    label=self.labels[index]
     
    sample = {'image': image,'label':label}
 
    return sample

def create_dataloaders(config):

    train_transforms =  transforms.Compose([
          transforms.ToPILImage(), #Transforms only works on PIL images (not numpy arrays)
          transforms.RandomResizedCrop(config['n_dim']), #Augment function in line 69 block 5
          transforms.ToTensor(),
        ])

    val_transforms = transforms.Compose([   
          transforms.ToPILImage(),
          transforms.Resize(config['n_dim']),
          transforms.ToTensor()
        ])

    x_train, x_val, x_test, y_train, y_val, y_test, np.array(train_dems), np.array(val_dems),
    np.array(test_dems), train_labels_raw,
    val_labels_raw, test_labels_raw =
    load_data(lab_value = config['lab_value'], run_num = config['run_num'], file_dict=config['file_dict'],
              csv_file = config['csv_file'], file2='cve_uuid_mrn.csv'):

    #calc weights not used in code.
    
    train_dataset=ImageDataset(x_train, y_train, train_transforms)
    test_dataset=ImageDataset(x_test, y_test, test_transforms)
    val_dataset=ImageDataset(x_val, y_val, val_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    return train_dataloader, test_dataloader, val_dataloader

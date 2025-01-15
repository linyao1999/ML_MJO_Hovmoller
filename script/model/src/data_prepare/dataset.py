# Contains classes and functions for loading and preprocessing data.
import yaml
from datetime import datetime, timedelta
import xarray as xr
import pandas as pd 
import numpy as np  
from torch.utils.data import Dataset, DataLoader
import torch    

class MapDataset(Dataset):
    def __init__(self, input_path, target_path, date_start, date_end, lead, lat_range=20, transform=None):
        """
        Args:
            input_path (str): Path to the input data file.
            target_path (str): Path to the target data file.
            date_start (str): Start date in 'YYYY-MM-DD' format.
            date_end (str): End date in 'YYYY-MM-DD' format.
            lead (int): Lead time in days for the target variable.
            lat_range (int): Latitude range to use for the input data.
            transform (callable, optional): Optional transform to apply to the data.
        """

        self.input = xr.open_dataarray(input_path).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range))

        # Define the target data shifted by the lead time
        date_start_date = datetime.strptime(date_start, "%Y-%m-%d")
        date_end_date = datetime.strptime(date_end, "%Y-%m-%d")
        target_start_date = date_start_date + timedelta(days=lead)
        target_end_date = date_end_date + timedelta(days=lead)

        date_end_date = pd.Timestamp(self.input.time[-1].values).to_pydatetime()
        # redefine the end date if target_end_date is greater than the last date in the dataset
        if target_end_date > date_end_date:
            target_end_date = date_end_date
            date_end_date = target_end_date - timedelta(days=lead)

        # Load input data
        self.input = xr.open_dataarray(input_path).sel(time=slice(date_start, date_end_date.strftime("%Y-%m-%d")), lat=slice(lat_range, -lat_range))
        
        self.target = xr.open_dataarray(target_path).sel(time=slice(target_start_date.strftime("%Y-%m-%d"),
                                                                    target_end_date.strftime("%Y-%m-%d")))
        
        print(f"Date start: {self.input.time[0].values}, Date end: {self.input.time[-1].values}")
        
        # Ensure input and target lengths match
        if len(self.input.time) != len(self.target.time):
            raise ValueError("Input and target dimensions do not match.")
        
        self.transform = transform

        self.has_logged = False # Flag to track if the shape of the input and target data has been logged

    def __len__(self):
        return len(self.input.time)

    def __getitem__(self, idx):
        # Get data and target for the given index
        input_shape = self.input.shape
        target_shape = self.target.shape

        input = self.input.isel(time=idx).values
        target = self.target.isel(time=idx).values

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        # Convert input and target to PyTorch tensors
        input = torch.tensor(input, dtype=torch.float32)
        input = input.view(-1, input_shape[1], input_shape[2])  # Add channel dimension
        target = torch.tensor(target, dtype=torch.float32)
        # target = target.view(-1, target_shape[1])  # Add channel dimension

        # Print input and output shapes for the first call
        if not self.has_logged:
            print(f"Input shape: {input.shape}, Target shape: {target.shape}")
            input_time = self.input.time[idx].values
            target_time = self.target.time[idx].values
            print(f"Input time: {input_time}, Target time: {target_time}")
            self.has_logged = True  # Set the flag to True after logging

        return input, target

class HovDataset(Dataset):
    def __init__(self, input_path, target_path, date_start, date_end, lead, memory_last, 
                 lat_range=10, transform=None):
        """
        Args:
            input_path (str): Path to the input data file.
            target_path (str): Path to the target data file.
            date_start (str): Start date in 'YYYY-MM-DD' format.
            date_end (str): End date in 'YYYY-MM-DD' format.
            lead (int): Lead time in days for the target variable.
            memory_last (int): Last time steps to use for the input data.
            lat_range (int): Latitude range to use for average.
            transform (callable, optional): Optional transform to apply to the data.
        """

        # Load input data [time, lon]
        self.input = xr.open_dataarray(input_path).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range)).mean(dim='lat')


        # Define the target data shifted by the lead time
        date_start_date = datetime.strptime(date_start, "%Y-%m-%d") + timedelta(days=int(memory_last))
        date_end_date = datetime.strptime(date_end, "%Y-%m-%d")
        target_start_date = date_start_date + timedelta(days=lead)
        target_end_date = date_end_date + timedelta(days=lead)

        date_end_date = pd.Timestamp(self.input.time[-1].values).to_pydatetime()
        # redefine the end date if target_end_date is greater than the last date in the dataset
        if target_end_date > date_end_date:
            target_end_date = date_end_date
            date_end_date = target_end_date - timedelta(days=lead)

        # Load input data
        input_wmem_values = []

        # memory_list = np.asarray(memory_list).astype(int)
        memory_list = np.arange(0, memory_last+1)

        for mem in memory_list[::-1]:
            date_wmem_start = date_start_date - timedelta(days=int(mem))
            date_wmem_end = date_end_date - timedelta(days=int(mem))

            input_wmem_values.append(
                self.input.sel(time=slice(date_wmem_start.strftime("%Y-%m-%d"),date_wmem_end.strftime("%Y-%m-%d"))).values
            )

        input_wmem_values = np.stack(input_wmem_values, axis=1)
        time = self.input.time.sel(time=slice(date_start_date.strftime("%Y-%m-%d"), date_end_date.strftime("%Y-%m-%d")))
        self.input = xr.DataArray(input_wmem_values, coords=[time, memory_list, self.input.lon], dims=['time', 'memory', 'lon'])

        self.target = xr.open_dataarray(target_path).sel(time=slice(target_start_date.strftime("%Y-%m-%d"),
                                                                    target_end_date.strftime("%Y-%m-%d")))
        
        print(f"Date start: {self.input.time[0].values}, Date end: {self.input.time[-1].values}")
        
        # Ensure input and target lengths match
        if len(self.input.time) != len(self.target.time):
            raise ValueError("Input and target dimensions do not match.")
        
        self.transform = transform

        self.has_logged = False # Flag to track if the shape of the input and target data has been logged

    def __len__(self):
        return len(self.input.time)

    def __getitem__(self, idx):
        # Get data and target for the given index
        input_shape = self.input.shape
        target_shape = self.target.shape

        input = self.input.isel(time=idx).values
        target = self.target.isel(time=idx).values

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        # Convert input and target to PyTorch tensors
        input = torch.tensor(input, dtype=torch.float32)
        input = input.view(-1, input_shape[1], input_shape[2])  # Add channel dimension
        target = torch.tensor(target, dtype=torch.float32)
        # target = target.view(-1, target_shape[1])  # Add channel dimension

        # Print input and output shapes for the first call
        if not self.has_logged:
            print(f"Input shape: {input.shape}, Target shape: {target.shape}")
            input_time = self.input.time[idx].values
            target_time = self.target.time[idx].values
            print(f"Input time: {input_time}, Target time: {target_time}")
            self.has_logged = True  # Set the flag to True after logging

        return input, target

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # print the config
    # print(yaml.dump(config))

    return config

def load_train_data(config, dataset_type="map"):
    # Load training and validation data
    if dataset_type == "map":
        train_data = MapDataset(config["data"]["input_path"], config["data"]["target_path"],
                                config["data"]["train_start"], config["data"]["train_end"], 
                                config["data"]["lead"], config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)
   
    elif dataset_type == "hov":
        train_data = HovDataset(config["data"]["input_path"], config["data"]["target_path"],
                                config["data"]["train_start"], config["data"]["train_end"], 
                                config["data"]["lead"], config["data"]["memory_last"], 
                                config["data"]["lat_range"], config["data"]["transform"])

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)

    return train_loader

def load_val_data(config, dataset_type="map"):
    # Load training and validation data
    if dataset_type == "map":
        val_data = MapDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["val_start"], config["data"]["val_end"], 
                              config["data"]["lead"], config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        val_loader = DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False)
    elif dataset_type == "hov": 
        val_data = HovDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["val_start"], config["data"]["val_end"], 
                              config["data"]["lead"], config["data"]["memory_last"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        val_loader = DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False)

    return val_loader

# clarify the time dimension
def get_time_dimension(input_path, date_start, date_end, lead, mem=0):
    input_da = xr.open_dataarray(input_path).sel(time=slice(date_start, date_end))

    date_start_date = datetime.strptime(date_start, "%Y-%m-%d") + timedelta(days=int(mem))
    date_end_date = datetime.strptime(date_end, "%Y-%m-%d")
    target_end_date = date_end_date + timedelta(days=lead)
    date_end_date = pd.Timestamp(input_da.time[-1].values).to_pydatetime()
    # redefine the end date if target_end_date is greater than the last date in the dataset
    if target_end_date > date_end_date:
        target_end_date = date_end_date
        date_end_date = target_end_date - timedelta(days=lead)

    return input_da.sel(time=slice(date_start_date.strftime("%Y-%m-%d"), date_end_date.strftime("%Y-%m-%d"))).time


# Contains classes and functions for loading and preprocessing data.
import yaml
from datetime import datetime, timedelta
import xarray as xr
import pandas as pd 
import numpy as np  
from torch.utils.data import Dataset, DataLoader
import torch    
from scipy.signal import windows  
# ======================================================================
# Maps 
# ======================================================================
class MapsDataset(Dataset):
    def __init__(self, input_path, target_path, date_start, date_end, lead, lat_range=20, transform=None):
        """
        Args:
            input_path (list of str): List of paths to input data files (one per variable).
            target_path (str): Path to the target data file.
            date_start (str): Start date in 'YYYY-MM-DD' format.
            date_end (str): End date in 'YYYY-MM-DD' format.
            lead (int): Lead time in days for the target variable.
            lat_range (int): Latitude range to use for the input data.
            transform (callable, optional): Optional transform to apply to the data.
        """
        assert isinstance(input_path, (list, tuple)), "input_path should be a list/tuple of file paths"

        self.n_var = len(input_path)
        self.transform = transform
        self.has_logged = False

        # Load and preprocess each variable, stack into shape [time, n_var, lat, lon]
        inputs = []
        for ipath in input_path:
            da = xr.open_dataarray(ipath).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range))
            # Normalize per variable using the training period (hardcoded here, change as needed)
            da_train = xr.open_dataarray(ipath).sel(time=slice('1979-01-01', '2001-12-31'), lat=slice(lat_range, -lat_range))
            da_mean = da_train.mean().values
            da_std = da_train.std().values
            inputs.append(((da - da_mean) / da_std).expand_dims('variable'))
        # Stack on new 'variable' dimension
        self.input = xr.concat(inputs, dim='variable')  # [variable, time, lat, lon]
        self.input = self.input.transpose('time', 'variable', 'lat', 'lon')  # [time, variable, lat, lon]

        # Target time handling
        date_start_dt = datetime.strptime(date_start, "%Y-%m-%d")
        date_end_dt = datetime.strptime(date_end, "%Y-%m-%d")
        target_start_date = date_start_dt + timedelta(days=lead)
        target_end_date = date_end_dt + timedelta(days=lead)
        last_input_date = pd.Timestamp(self.input.time[-1].values).to_pydatetime()
        if target_end_date > last_input_date:
            target_end_date = last_input_date
            date_end_dt = target_end_date - timedelta(days=lead)
            # Re-slice input so that input and target times match
            self.input = self.input.sel(time=slice(date_start, date_end_dt.strftime("%Y-%m-%d")))

        # Target data
        # Prepare target for all leads
        output_allleads = []
        for le in range(0, lead+1):
            target_start = date_start_dt + timedelta(days=le)
            target_end = date_end_dt + timedelta(days=le)
            target_da = xr.open_dataarray(target_path).sel(
                time=slice(target_start.strftime("%Y-%m-%d"), target_end.strftime("%Y-%m-%d"))
            )
            output_allleads.append(target_da.values)
        # [lead+1, time, n_modes]
        output_allleads = np.stack(output_allleads, axis=0)  # shape: [lead+1, time, n_modes]
        output_allleads = np.transpose(output_allleads, (1, 0, 2))  # [time, lead+1, n_modes]
        self.target = output_allleads

        # self.target = xr.open_dataarray(target_path).sel(
        #     time=slice(target_start_date.strftime("%Y-%m-%d"), target_end_date.strftime("%Y-%m-%d"))
        # )

        # Ensure lengths match
        if len(self.input.time) != len(self.target):
            raise ValueError(f"Input and target dimensions do not match: {len(self.input.time)} vs {len(self.target)}")

    def __len__(self):
        return len(self.input.time)

    def __getitem__(self, idx):
        # Input shape: [time, n_var, lat, lon]
        input = self.input.isel(time=idx).values   # shape: [n_var, lat, lon]
        if input.ndim == 2:  # [lat, lon], only one variable, add channel dimension
            input = np.expand_dims(input, axis=0)  # Now [1, lat, lon]

        target = self.target[idx].flatten()  # shape: [lead+1, n_modes]

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        # Print shape once for debug
        if not self.has_logged:
            print(f"Input shape: {input.shape}, Target shape: {target.shape}")
            input_time = self.input.time[idx].values
            print(f"Input time: {input_time}")
            self.has_logged = True

        return input, target

class MapsSymDataset(Dataset):
    def __init__(self, input_path, target_path, date_start, date_end, lead, lat_range=20, transform=None):
        """
        Args:
            input_path (list of str): List of paths to input data files (one per variable).
            target_path (str): Path to the target data file.
            date_start (str): Start date in 'YYYY-MM-DD' format.
            date_end (str): End date in 'YYYY-MM-DD' format.
            lead (int): Lead time in days for the target variable.
            lat_range (int): Latitude range to use for the input data.
            transform (callable, optional): Optional transform to apply to the data.
        """
        assert isinstance(input_path, (list, tuple)), "input_path should be a list/tuple of file paths"

        self.n_var = len(input_path)
        self.transform = transform
        self.has_logged = False

        # Load and preprocess each variable, stack into shape [time, n_var, lat, lon]
        inputs = []
        for ipath in input_path:
            da = xr.open_dataarray(ipath).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range))
            # Normalize per variable using the training period (hardcoded here, change as needed)
            da_train = xr.open_dataarray(ipath).sel(time=slice('1979-01-01', '2001-12-31'), lat=slice(lat_range, -lat_range))
            da_mean = da_train.mean().values
            da_std = da_train.std().values
            data = (da - da_mean) / da_std  # [time, lat, lon]
            # extract symmetric component
            data_sym = 0.5 * (data + data.sel(lat=-data.lat).values)

            inputs.append(data_sym.expand_dims('variable'))
        # Stack on new 'variable' dimension
        self.input = xr.concat(inputs, dim='variable')  # [variable, time, lat, lon]
        self.input = self.input.transpose('time', 'variable', 'lat', 'lon')  # [time, variable, lat, lon]

        # Target time handling
        date_start_dt = datetime.strptime(date_start, "%Y-%m-%d")
        date_end_dt = datetime.strptime(date_end, "%Y-%m-%d")
        target_start_date = date_start_dt + timedelta(days=lead)
        target_end_date = date_end_dt + timedelta(days=lead)
        last_input_date = pd.Timestamp(self.input.time[-1].values).to_pydatetime()
        if target_end_date > last_input_date:
            target_end_date = last_input_date
            date_end_dt = target_end_date - timedelta(days=lead)
            # Re-slice input so that input and target times match
            self.input = self.input.sel(time=slice(date_start, date_end_dt.strftime("%Y-%m-%d")))

        # Target data
        # Prepare target for all leads
        output_allleads = []
        for le in range(0, lead+1):
            target_start = date_start_dt + timedelta(days=le)
            target_end = date_end_dt + timedelta(days=le)
            target_da = xr.open_dataarray(target_path).sel(
                time=slice(target_start.strftime("%Y-%m-%d"), target_end.strftime("%Y-%m-%d"))
            )
            output_allleads.append(target_da.values)
        # [lead+1, time, n_modes]
        output_allleads = np.stack(output_allleads, axis=0)  # shape: [lead+1, time, n_modes]
        output_allleads = np.transpose(output_allleads, (1, 0, 2))  # [time, lead+1, n_modes]
        self.target = output_allleads

        # self.target = xr.open_dataarray(target_path).sel(
        #     time=slice(target_start_date.strftime("%Y-%m-%d"), target_end_date.strftime("%Y-%m-%d"))
        # )

        # Ensure lengths match
        if len(self.input.time) != len(self.target):
            raise ValueError(f"Input and target dimensions do not match: {len(self.input.time)} vs {len(self.target)}")

    def __len__(self):
        return len(self.input.time)

    def __getitem__(self, idx):
        # Input shape: [time, n_var, lat, lon]
        input = self.input.isel(time=idx).values   # shape: [n_var, lat, lon]
        if input.ndim == 2:  # [lat, lon], only one variable, add channel dimension
            input = np.expand_dims(input, axis=0)  # Now [1, lat, lon]

        target = self.target[idx].flatten()  # shape: [lead+1, n_modes]

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        # Print shape once for debug
        if not self.has_logged:
            print(f"Input shape: {input.shape}, Target shape: {target.shape}")
            input_time = self.input.time[idx].values
            print(f"Input time: {input_time}")
            self.has_logged = True

        return input, target

class MapsAsymDataset(Dataset):
    def __init__(self, input_path, target_path, date_start, date_end, lead, lat_range=20, transform=None):
        """
        Args:
            input_path (list of str): List of paths to input data files (one per variable).
            target_path (str): Path to the target data file.
            date_start (str): Start date in 'YYYY-MM-DD' format.
            date_end (str): End date in 'YYYY-MM-DD' format.
            lead (int): Lead time in days for the target variable.
            lat_range (int): Latitude range to use for the input data.
            transform (callable, optional): Optional transform to apply to the data.
        """
        assert isinstance(input_path, (list, tuple)), "input_path should be a list/tuple of file paths"

        self.n_var = len(input_path)
        self.transform = transform
        self.has_logged = False

        # Load and preprocess each variable, stack into shape [time, n_var, lat, lon]
        inputs = []
        for ipath in input_path:
            da = xr.open_dataarray(ipath).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range))
            # Normalize per variable using the training period (hardcoded here, change as needed)
            da_train = xr.open_dataarray(ipath).sel(time=slice('1979-01-01', '2001-12-31'), lat=slice(lat_range, -lat_range))
            da_mean = da_train.mean().values
            da_std = da_train.std().values
            data = (da - da_mean) / da_std  # [time, lat, lon]
            # extract antisymmetric component
            data_asym = 0.5 * (data - data.sel(lat=-data.lat).values)
            inputs.append(data_asym.expand_dims('variable'))
        # Stack on new 'variable' dimension
        self.input = xr.concat(inputs, dim='variable')  # [variable, time, lat, lon]
        self.input = self.input.transpose('time', 'variable', 'lat', 'lon')  # [time, variable, lat, lon]

        # Target time handling
        date_start_dt = datetime.strptime(date_start, "%Y-%m-%d")
        date_end_dt = datetime.strptime(date_end, "%Y-%m-%d")
        target_start_date = date_start_dt + timedelta(days=lead)
        target_end_date = date_end_dt + timedelta(days=lead)
        last_input_date = pd.Timestamp(self.input.time[-1].values).to_pydatetime()
        if target_end_date > last_input_date:
            target_end_date = last_input_date
            date_end_dt = target_end_date - timedelta(days=lead)
            # Re-slice input so that input and target times match
            self.input = self.input.sel(time=slice(date_start, date_end_dt.strftime("%Y-%m-%d")))

        # Target data
        # Prepare target for all leads
        output_allleads = []
        for le in range(0, lead+1):
            target_start = date_start_dt + timedelta(days=le)
            target_end = date_end_dt + timedelta(days=le)
            target_da = xr.open_dataarray(target_path).sel(
                time=slice(target_start.strftime("%Y-%m-%d"), target_end.strftime("%Y-%m-%d"))
            )
            output_allleads.append(target_da.values)
        # [lead+1, time, n_modes]
        output_allleads = np.stack(output_allleads, axis=0)  # shape: [lead+1, time, n_modes]
        output_allleads = np.transpose(output_allleads, (1, 0, 2))  # [time, lead+1, n_modes]
        self.target = output_allleads

        # self.target = xr.open_dataarray(target_path).sel(
        #     time=slice(target_start_date.strftime("%Y-%m-%d"), target_end_date.strftime("%Y-%m-%d"))
        # )

        # Ensure lengths match
        if len(self.input.time) != len(self.target):
            raise ValueError(f"Input and target dimensions do not match: {len(self.input.time)} vs {len(self.target)}")

    def __len__(self):
        return len(self.input.time)

    def __getitem__(self, idx):
        # Input shape: [time, n_var, lat, lon]
        input = self.input.isel(time=idx).values   # shape: [n_var, lat, lon]
        if input.ndim == 2:  # [lat, lon], only one variable, add channel dimension
            input = np.expand_dims(input, axis=0)  # Now [1, lat, lon]

        target = self.target[idx].flatten()  # shape: [lead+1, n_modes]

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        # Print shape once for debug
        if not self.has_logged:
            print(f"Input shape: {input.shape}, Target shape: {target.shape}")
            input_time = self.input.time[idx].values
            print(f"Input time: {input_time}")
            self.has_logged = True

        return input, target

# ======================================================================
# LatavgDataset
# ======================================================================
class LatavgDataset(Dataset):
    def __init__(self, input_path, target_path, date_start, date_end, lead, lat_range=20, transform=None):
        """
        Args:
            input_path (list of str): List of paths to input data files (one per variable).
            target_path (str): Path to the target data file.
            date_start (str): Start date in 'YYYY-MM-DD' format.
            date_end (str): End date in 'YYYY-MM-DD' format.
            lead (int): Lead time in days for the target variable.
            lat_range (int): Latitude range to use for the input data.
            transform (callable, optional): Optional transform to apply to the data.
        """
        assert isinstance(input_path, (list, tuple)), "input_path should be a list/tuple of file paths"

        self.n_var = len(input_path)
        self.transform = transform
        self.has_logged = False

        # Load and preprocess each variable, stack into shape [time, n_var, lat, lon]
        inputs = []
        for ipath in input_path:
            da = xr.open_dataarray(ipath).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range)).mean(dim='lat')
            # Normalize per variable using the training period (hardcoded here, change as needed)
            da_train = xr.open_dataarray(ipath).sel(time=slice('1979-01-01', '2001-12-31'), lat=slice(lat_range, -lat_range)).mean(dim='lat')
            da_mean = da_train.mean().values
            da_std = da_train.std().values
            data = (da - da_mean) / da_std  # [time, lon]

            inputs.append(data.expand_dims('variable'))
        # Stack on new 'variable' dimension
        self.input = xr.concat(inputs, dim='variable')  # [variable, time, lon]
        self.input = self.input.transpose('time', 'variable', 'lon')  # [time, variable, lon]

        # Target time handling
        date_start_dt = datetime.strptime(date_start, "%Y-%m-%d")
        date_end_dt = datetime.strptime(date_end, "%Y-%m-%d")
        target_start_date = date_start_dt + timedelta(days=lead)
        target_end_date = date_end_dt + timedelta(days=lead)
        last_input_date = pd.Timestamp(self.input.time[-1].values).to_pydatetime()
        if target_end_date > last_input_date:
            target_end_date = last_input_date
            date_end_dt = target_end_date - timedelta(days=lead)
            # Re-slice input so that input and target times match
            self.input = self.input.sel(time=slice(date_start, date_end_dt.strftime("%Y-%m-%d")))

        # Target data
        # Prepare target for all leads
        output_allleads = []
        for le in range(0, lead+1):
            target_start = date_start_dt + timedelta(days=le)
            target_end = date_end_dt + timedelta(days=le)
            target_da = xr.open_dataarray(target_path).sel(
                time=slice(target_start.strftime("%Y-%m-%d"), target_end.strftime("%Y-%m-%d"))
            )
            output_allleads.append(target_da.values)
        # [lead+1, time, n_modes]
        output_allleads = np.stack(output_allleads, axis=0)  # shape: [lead+1, time, n_modes]
        output_allleads = np.transpose(output_allleads, (1, 0, 2))  # [time, lead+1, n_modes]
        self.target = output_allleads

        # self.target = xr.open_dataarray(target_path).sel(
        #     time=slice(target_start_date.strftime("%Y-%m-%d"), target_end_date.strftime("%Y-%m-%d"))
        # )

        # Ensure lengths match
        if len(self.input.time) != len(self.target):
            raise ValueError(f"Input and target dimensions do not match: {len(self.input.time)} vs {len(self.target)}")

    def __len__(self):
        return len(self.input.time)

    def __getitem__(self, idx):
        # Input shape: [time, n_var, lon]
        input = self.input.isel(time=idx).values   # shape: [n_var, lon]
        if input.ndim == 1:  # [lon], only one variable, add channel dimension
            input = np.expand_dims(input, axis=(0, 1))  # Now [1, 1, lon]
        elif input.ndim == 2:  # [n_var, lon], add channel dimension
            input = np.expand_dims(input, axis=1)  # Now [n_var, 1, lon]

        target = self.target[idx].flatten()  # shape: [lead+1, n_modes]

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        # Print shape once for debug
        if not self.has_logged:
            print(f"Input shape: {input.shape}, Target shape: {target.shape}")
            input_time = self.input.time[idx].values
            print(f"Input time: {input_time}")
            self.has_logged = True

        return input, target

class LatavgTimeDataset(Dataset):
    def __init__(self, input_path, target_path, date_start, date_end, lead, memory_last, lat_range=20, transform=None):
        """
        Args:
            input_path (list of str): List of paths to input data files (one per variable).
            target_path (str): Path to the target data file.
            date_start (str): Start date in 'YYYY-MM-DD' format.
            date_end (str): End date in 'YYYY-MM-DD' format.
            lead (int): Lead time in days for the target variable.
            memory_last (int): Last time steps to use for the input data.
            lat_range (int): Latitude range to use for the input data.
            transform (callable, optional): Optional transform to apply to the data.
        """
        assert isinstance(input_path, (list, tuple)), "input_path should be a list/tuple of file paths"

        self.n_var = len(input_path)
        self.transform = transform
        self.has_logged = False

        # Load and preprocess each variable, stack into shape [time, n_var, lat, lon]
        inputs = []
        for ipath in input_path:
            da = xr.open_dataarray(ipath).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range)).mean(dim='lat')
            # Normalize per variable using the training period (hardcoded here, change as needed)
            da_train = xr.open_dataarray(ipath).sel(time=slice('1979-01-01', '2001-12-31'), lat=slice(lat_range, -lat_range)).mean(dim='lat')
            da_mean = da_train.mean().values
            da_std = da_train.std().values
            data = (da - da_mean) / da_std  # [time, lon]

            inputs.append(data.expand_dims('variable'))
        # Stack on new 'variable' dimension
        self.input = xr.concat(inputs, dim='variable')  # [variable, time, lon]
        self.input = self.input.expand_dims({'memory': range(memory_last+1)})  # [memory, variable, time, lon]
        self.input = self.input.copy()
        self.input = self.input.transpose('time', 'variable', 'memory', 'lon')  # [time, variable, memory, lon]

        # Target time handling
        date_start_dt = datetime.strptime(date_start, "%Y-%m-%d")
        date_end_dt = datetime.strptime(date_end, "%Y-%m-%d")
        target_start_date = date_start_dt + timedelta(days=lead)
        target_end_date = date_end_dt + timedelta(days=lead)
        last_input_date = pd.Timestamp(self.input.time[-1].values).to_pydatetime()
        if target_end_date > last_input_date:
            target_end_date = last_input_date
            date_end_dt = target_end_date - timedelta(days=lead)
            # Re-slice input so that input and target times match
            self.input = self.input.sel(time=slice(date_start, date_end_dt.strftime("%Y-%m-%d")))

        # Target data
        # Prepare target for all leads
        output_allleads = []
        for le in range(0, lead+1):
            target_start = date_start_dt + timedelta(days=le)
            target_end = date_end_dt + timedelta(days=le)
            target_da = xr.open_dataarray(target_path).sel(
                time=slice(target_start.strftime("%Y-%m-%d"), target_end.strftime("%Y-%m-%d"))
            )
            output_allleads.append(target_da.values)
        # [lead+1, time, n_modes]
        output_allleads = np.stack(output_allleads, axis=0)  # shape: [lead+1, time, n_modes]
        output_allleads = np.transpose(output_allleads, (1, 0, 2))  # [time, lead+1, n_modes]
        self.target = output_allleads

        # self.target = xr.open_dataarray(target_path).sel(
        #     time=slice(target_start_date.strftime("%Y-%m-%d"), target_end_date.strftime("%Y-%m-%d"))
        # )

        # Ensure lengths match
        if len(self.input.time) != len(self.target):
            raise ValueError(f"Input and target dimensions do not match: {len(self.input.time)} vs {len(self.target)}")

    def __len__(self):
        return len(self.input.time)

    def __getitem__(self, idx):
        # Input shape: [time, n_var, memory, lon]
        input = self.input.isel(time=idx).values   # shape: [n_var, memory, lon]
        if input.ndim == 2:  # [memory, lon], only one variable, add channel dimension
            input = np.expand_dims(input, axis=0)  # Now [1, memory, lon]

        target = self.target[idx].flatten()  # shape: [lead+1, n_modes]

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        # Print shape once for debug
        if not self.has_logged:
            print(f"Input shape: {input.shape}, Target shape: {target.shape}")
            input_time = self.input.time[idx].values
            print(f"Input time: {input_time}")
            self.has_logged = True

        return input, target

# ======================================================================
# Hovmoller Dataset
# ======================================================================
class HovDataset(Dataset):
    # This class is used to load symmetric and asymmetric hovmoller data
    # to predict ROMI at all leads
    def __init__(self, input_path, target_path, date_start, date_end, lead, memory_last, 
                 lat_range=20, transform=None):
        """
        create the symmetric and antisymmetric hovmoller diagrams for the input data using OLR
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
        # self.input = xr.open_dataarray(input_path).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range)).mean(dim='lat')
        inputs = []
        for ipath in input_path:
            da = xr.open_dataarray(ipath).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range)).mean(dim='lat')
            # normalize per variable using the training period (hardcoded here, change as needed)
            da_train = xr.open_dataarray(ipath).sel(time=slice('1979-01-01', '2001-12-31'), lat=slice(lat_range, -lat_range)).mean(dim='lat')
            da_mean = da_train.mean().values
            da_std = da_train.std().values
            data = (da - da_mean) / da_std  # [time, lon]
            inputs.append(data.expand_dims('variable'))

        # Stack on new 'variable' dimension
        self.input = xr.concat(inputs, dim='variable')  # [variable, time, lon]
        self.input = self.input.transpose('time', 'variable', 'lon')  # [time, variable, lon]

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

        time = self.input.time.sel(time=slice(date_start_date.strftime("%Y-%m-%d"), date_end_date.strftime("%Y-%m-%d"))).values

        # Load input data
        input_wmem_values = []

        # memory_list = np.asarray(memory_list).astype(int)
        memory_list = np.arange(0, memory_last+1)

        for mem in memory_list[::-1]:
            date_wmem_start = date_start_date - timedelta(days=int(mem))
            date_wmem_end = date_end_date - timedelta(days=int(mem))

            data = self.input.sel(time=slice(date_wmem_start.strftime("%Y-%m-%d"),date_wmem_end.strftime("%Y-%m-%d"))).values
            input_wmem_values.append(
                data
            )

        input_wmem_values = np.stack(input_wmem_values, axis=2)
        print(f"Input shape: {input_wmem_values.shape}")

        # self.input = xr.DataArray(input_wmem_values, coords=[time,self.input.variable.values, memory_list, self.input.lon.values], dims=['time','variable', 'memory', 'lon'])
        self.input = xr.DataArray(
            input_wmem_values,  # shape: [time, variable, memory, lon]
            dims=['time', 'variable', 'memory', 'lon'],
            coords={
                'time': time,
                'variable': np.arange(len(input_path)),  # [0, 1, ..., n_var-1]
                'memory': memory_list,
                'lon': self.input.lon.values
            }
     )
        output_allleads = []
        for le in range(0, lead+1):
            target_start = date_start_date + timedelta(days=le)
            target_end = date_end_date + timedelta(days=le)
            target = xr.open_dataarray(target_path).sel(time=slice(target_start.strftime("%Y-%m-%d"),
                                                                    target_end.strftime("%Y-%m-%d")))
            output_allleads.append(target.values)

        output_allleads = np.stack(output_allleads, axis=1)
        self.target = xr.DataArray(output_allleads, coords=[time, np.arange(0, lead+1), np.arange(1,3)], dims=['time', 'lead', 'mode'])

        # self.target = xr.open_dataarray(target_path).sel(time=slice(target_start_date.strftime("%Y-%m-%d"),
        #                                                             target_end_date.strftime("%Y-%m-%d")))
        
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
        target = self.target.isel(time=idx).values.flatten()

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        # Convert input and target to PyTorch tensors
        input = torch.tensor(input, dtype=torch.float32)
        input = input.view(-1, input_shape[-2], input_shape[-1])  # Add channel dimension
        target = torch.tensor(target, dtype=torch.float32)

        # Print input and output shapes for the first call
        if not self.has_logged:
            print(f"Input shape: {input.shape}, Target shape: {target.shape}")
            input_time = self.input.time[idx].values
            target_time = self.target.time[idx].values
            print(f"Input time: {input_time}, Target time: {target_time}")
            self.has_logged = True  # Set the flag to True after logging

        return input, target

class HovSMDataset(Dataset):
    # This class is used to load symmetric and asymmetric hovmoller data
    # to predict ROMI at all leads
    def __init__(self, input_path, target_path, date_start, date_end, lead, memory_last, 
                 lat_range=10, window_len=11, residual=False, transform=None):
        """
        create the symmetric and antisymmetric hovmoller diagrams for the input data using OLR; 
        apply running averages over segments
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
        print(type(residual))
        # Load input data [time, lon]
        # self.input = xr.open_dataarray(input_path).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range)).mean(dim='lat')
        inputs = []
        for ipath in input_path:
            da = xr.open_dataarray(ipath).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range)).mean(dim='lat')
            # normalize per variable using the training period (hardcoded here, change as needed)
            da_train = xr.open_dataarray(ipath).sel(time=slice('1979-01-01', '2001-12-31'), lat=slice(lat_range, -lat_range)).mean(dim='lat')
            da_mean = da_train.mean().values
            da_std = da_train.std().values
            data = (da - da_mean) / da_std  # [time, lon]
            inputs.append(data.expand_dims('variable'))

        # Stack on new 'variable' dimension
        # self.input = xr.concat(inputs, dim='variable')  # [variable, time, lon]
        # self.input = self.input.transpose('time', 'variable', 'lon')  # [time, variable, lon]
        input_concat = xr.concat(inputs, dim='variable')  # [variable, time, lon]
        input_concat = input_concat.transpose('time', 'variable', 'lon')  # [time, variable, lon]

        # Define the target data shifted by the lead time
        date_start_date = datetime.strptime(date_start, "%Y-%m-%d") + timedelta(days=int(memory_last))
        date_end_date = datetime.strptime(date_end, "%Y-%m-%d")
        target_start_date = date_start_date + timedelta(days=lead)
        target_end_date = date_end_date + timedelta(days=lead)

        date_end_date = pd.Timestamp(input_concat.time[-1].values).to_pydatetime()
        # redefine the end date if target_end_date is greater than the last date in the dataset
        if target_end_date > date_end_date:
            target_end_date = date_end_date
            date_end_date = target_end_date - timedelta(days=lead)

        time = input_concat.time.sel(time=slice(date_start_date.strftime("%Y-%m-%d"), date_end_date.strftime("%Y-%m-%d"))).values

        # Load input data
        input_wmem_values = []

        # memory_list = np.asarray(memory_list).astype(int)
        memory_list = np.arange(0, memory_last+1)

        for mem in memory_list[::-1]:
            date_wmem_start = date_start_date - timedelta(days=int(mem))
            date_wmem_end = date_end_date - timedelta(days=int(mem))

            data = input_concat.sel(time=slice(date_wmem_start.strftime("%Y-%m-%d"),date_wmem_end.strftime("%Y-%m-%d"))).values
            input_wmem_values.append(
                data
            )

        input_wmem_values = np.stack(input_wmem_values, axis=2)
        print(f"Input shape: {input_wmem_values.shape}")

        # self.input = xr.DataArray(input_wmem_values, coords=[time,self.input.variable.values, memory_list, self.input.lon.values], dims=['time','variable', 'memory', 'lon'])
        input_concat = xr.DataArray(
            input_wmem_values,  # shape: [time, variable, memory, lon]
            dims=['time', 'variable', 'memory', 'lon'],
            coords={
                'time': time,
                'variable': np.arange(len(input_path)),  # [0, 1, ..., n_var-1]
                'memory': memory_list,
                'lon': input_concat.lon.values
            }
     )
        # apply running averages
        taper_weights = xr.DataArray(
            windows.hann(window_len),
            dims=['window']
        )

        taper_weights = taper_weights / taper_weights.sum()

        rolling_windows = input_concat.rolling(
            memory=window_len,
            center=True,
            min_periods=1,
        ).construct('window')

        weighted_sum = (rolling_windows * taper_weights).sum(dim='window', skipna=True)
        valid_weights = taper_weights.where(rolling_windows.notnull())
        weights_sum = valid_weights.sum(dim='window', skipna=True)

        if residual:
            print(str(residual))
            self.input = input_concat - (weighted_sum / weights_sum).values
        else:
            print(str(residual))
            self.input = (weighted_sum / weights_sum)

        output_allleads = []
        for le in range(0, lead+1):
            target_start = date_start_date + timedelta(days=le)
            target_end = date_end_date + timedelta(days=le)
            target = xr.open_dataarray(target_path).sel(time=slice(target_start.strftime("%Y-%m-%d"),
                                                                    target_end.strftime("%Y-%m-%d")))
            output_allleads.append(target.values)

        output_allleads = np.stack(output_allleads, axis=1)
        self.target = xr.DataArray(output_allleads, coords=[time, np.arange(0, lead+1), np.arange(1,3)], dims=['time', 'lead', 'mode'])

        # self.target = xr.open_dataarray(target_path).sel(time=slice(target_start_date.strftime("%Y-%m-%d"),
        #                                                             target_end_date.strftime("%Y-%m-%d")))
        
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
        target = self.target.isel(time=idx).values.flatten()

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        # Convert input and target to PyTorch tensors
        input = torch.tensor(input, dtype=torch.float32)
        input = input.view(-1, input_shape[-2], input_shape[-1])  # Add channel dimension
        target = torch.tensor(target, dtype=torch.float32)

        # Print input and output shapes for the first call
        if not self.has_logged:
            print(f"Input shape: {input.shape}, Target shape: {target.shape}")
            input_time = self.input.time[idx].values
            target_time = self.target.time[idx].values
            print(f"Input time: {input_time}, Target time: {target_time}")
            self.has_logged = True  # Set the flag to True after logging

        return input, target

class HovTwoDataset(Dataset):
    # This class is used to load symmetric and asymmetric hovmoller data
    # to predict ROMI at all leads
    def __init__(self, input_path, target_path, date_start, date_end, lead, memory_last, 
                 lat_range=20, transform=None):
        """
        create the symmetric and antisymmetric hovmoller diagrams for the input data using OLR
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
        # self.input = xr.open_dataarray(input_path).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range)).mean(dim='lat')
        inputs = []
        for ipath in input_path:
            da = xr.open_dataarray(ipath).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range))
            # normalize per variable using the training period (hardcoded here, change as needed)
            da_train = xr.open_dataarray(ipath).sel(time=slice('1979-01-01', '2001-12-31'), lat=slice(lat_range, -lat_range))
            da_mean = da_train.mean().values
            da_std = da_train.std().values
            data = (da - da_mean) / da_std  # [time, lat, lon]
            # extract symmetric component
            data_sym = 0.5 * (data + data.sel(lat=-data.lat).values).sel(lat=slice(lat_range,0)).mean(dim='lat')
            inputs.append(data_sym.expand_dims('variable'))
            # extract antisymmetric component
            data_asym = 0.5 * (data - data.sel(lat=-data.lat).values).sel(lat=slice(lat_range,0)).mean(dim='lat')
            inputs.append(data_asym.expand_dims('variable'))
        # Stack on new 'variable' dimension
        self.input = xr.concat(inputs, dim='variable')  # [variable, time, lon]
        self.input = self.input.transpose('time', 'variable', 'lon')

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

        time = self.input.time.sel(time=slice(date_start_date.strftime("%Y-%m-%d"), date_end_date.strftime("%Y-%m-%d"))).values

        # Load input data
        input_wmem_values = []

        # memory_list = np.asarray(memory_list).astype(int)
        memory_list = np.arange(0, memory_last+1)

        for mem in memory_list[::-1]:
            date_wmem_start = date_start_date - timedelta(days=int(mem))
            date_wmem_end = date_end_date - timedelta(days=int(mem))

            data = self.input.sel(time=slice(date_wmem_start.strftime("%Y-%m-%d"),date_wmem_end.strftime("%Y-%m-%d"))).values
            input_wmem_values.append(
                data
            )

        input_wmem_values = np.stack(input_wmem_values, axis=2)
        print(f"Input shape: {input_wmem_values.shape}")

        # self.input = xr.DataArray(input_wmem_values, coords=[time,self.input.variable.values, memory_list, self.input.lon.values], dims=['time','variable', 'memory', 'lon'])
        self.input = xr.DataArray(
            input_wmem_values,  # shape: [time, variable, memory, lon]
            dims=['time', 'variable', 'memory', 'lon'],
            coords={
                'time': time,
                'variable': np.arange(len(input_path) * 2),  # [0, 1, ..., n_var-1]
                'memory': memory_list,
                'lon': self.input.lon.values
            }
     )
        output_allleads = []
        for le in range(0, lead+1):
            target_start = date_start_date + timedelta(days=le)
            target_end = date_end_date + timedelta(days=le)
            target = xr.open_dataarray(target_path).sel(time=slice(target_start.strftime("%Y-%m-%d"),
                                                                    target_end.strftime("%Y-%m-%d")))
            output_allleads.append(target.values)

        output_allleads = np.stack(output_allleads, axis=1)
        self.target = xr.DataArray(output_allleads, coords=[time, np.arange(0, lead+1), np.arange(1,3)], dims=['time', 'lead', 'mode'])

        # self.target = xr.open_dataarray(target_path).sel(time=slice(target_start_date.strftime("%Y-%m-%d"),
        #                                                             target_end_date.strftime("%Y-%m-%d")))
        
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
        target = self.target.isel(time=idx).values.flatten()

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        # Convert input and target to PyTorch tensors
        input = torch.tensor(input, dtype=torch.float32)
        input = input.view(-1, input_shape[-2], input_shape[-1])  # Add channel dimension
        target = torch.tensor(target, dtype=torch.float32)

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
        train_data = MapsDataset(config["data"]["input_path"], config["data"]["target_path"],
                                config["data"]["train_start"], config["data"]["train_end"], 
                                config["data"]["lead"], config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)

    elif dataset_type == "map_sym":
        train_data = MapsSymDataset(config["data"]["input_path"], config["data"]["target_path"],
                                config["data"]["train_start"], config["data"]["train_end"], 
                                config["data"]["lead"], config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)

    elif dataset_type == "map_asym":
        train_data = MapsAsymDataset(config["data"]["input_path"], config["data"]["target_path"],
                                config["data"]["train_start"], config["data"]["train_end"], 
                                config["data"]["lead"], config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)

    elif dataset_type == "latavg":
        train_data = LatavgDataset(config["data"]["input_path"], config["data"]["target_path"],
                                config["data"]["train_start"], config["data"]["train_end"], 
                                config["data"]["lead"], config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)

    elif dataset_type == "latavg_time":
        train_data = LatavgTimeDataset(config["data"]["input_path"], config["data"]["target_path"],
                                config["data"]["train_start"], config["data"]["train_end"], 
                                config["data"]["lead"], config["data"]["memory_last"], 
                                config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)

    elif dataset_type == "hov":
        train_data = HovDataset(config["data"]["input_path"], config["data"]["target_path"],
                                config["data"]["train_start"], config["data"]["train_end"], 
                                config["data"]["lead"], config["data"]["memory_last"], 
                                config["data"]["lat_range"], config["data"]["transform"])

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)

    elif dataset_type == "hov_sm":
        train_data = HovSMDataset(config["data"]["input_path"], config["data"]["target_path"],
                                config["data"]["train_start"], config["data"]["train_end"], 
                                config["data"]["lead"], config["data"]["memory_last"], 
                                config["data"]["lat_range"], config["data"]["window_len"], config["data"]["residual"], config["data"]["transform"])

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)

    elif dataset_type == "hov_two":
        train_data = HovTwoDataset(config["data"]["input_path"], config["data"]["target_path"],
                                config["data"]["train_start"], config["data"]["train_end"], 
                                config["data"]["lead"], config["data"]["memory_last"], 
                                config["data"]["lat_range"], config["data"]["transform"])

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)

    return train_loader

def load_val_data(config, dataset_type="map"):
    # Load training and validation data
    if dataset_type == "map":
        val_data = MapsDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["val_start"], config["data"]["val_end"], 
                              config["data"]["lead"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        val_loader = DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False)

    elif dataset_type == "map_sym":
        val_data = MapsSymDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["val_start"], config["data"]["val_end"], 
                              config["data"]["lead"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        val_loader = DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False)
    
    elif dataset_type == "map_asym":
        val_data = MapsAsymDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["val_start"], config["data"]["val_end"], 
                              config["data"]["lead"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        val_loader = DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False)
    
    elif dataset_type == "latavg":
        val_data = LatavgDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["val_start"], config["data"]["val_end"], 
                              config["data"]["lead"], config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        val_loader = DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False)
    
    elif dataset_type == "latavg_time":
        val_data = LatavgTimeDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["val_start"], config["data"]["val_end"], 
                              config["data"]["lead"], config["data"]["memory_last"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        val_loader = DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False)
    
    elif dataset_type == "hov": 
        val_data = HovDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["val_start"], config["data"]["val_end"], 
                              config["data"]["lead"], config["data"]["memory_last"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        val_loader = DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False)

    elif dataset_type == "hov_sm": 
        val_data = HovSMDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["val_start"], config["data"]["val_end"], 
                              config["data"]["lead"], config["data"]["memory_last"], 
                              config["data"]["lat_range"], config["data"]["window_len"], config["data"]["residual"], config["data"]["transform"])
        # Create data loaders
        val_loader = DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False)

    elif dataset_type == "hov_two":
        val_data = HovTwoDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["val_start"], config["data"]["val_end"], 
                              config["data"]["lead"], config["data"]["memory_last"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        val_loader = DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False)

    return val_loader

def load_test_data(config, dataset_type="map"):
    # Load training and test data
    if dataset_type == "map":
        test_data = MapsDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["test_start"], config["data"]["test_end"], 
                              config["data"]["lead"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        test_loader = DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False)

    elif dataset_type == "map_sym":
        test_data = MapsSymDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["test_start"], config["data"]["test_end"], 
                              config["data"]["lead"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        test_loader = DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False)
    elif dataset_type == "map_asym":
        test_data = MapsAsymDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["test_start"], config["data"]["test_end"], 
                              config["data"]["lead"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        test_loader = DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False)
    elif dataset_type == "latavg":
        test_data = LatavgDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["test_start"], config["data"]["test_end"], 
                              config["data"]["lead"], config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        test_loader = DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False)
    elif dataset_type == "latavg_time":
        test_data = LatavgTimeDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["test_start"], config["data"]["test_end"], 
                              config["data"]["lead"], config["data"]["memory_last"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        test_loader = DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False)

    elif dataset_type == "hov": 
        test_data = HovDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["test_start"], config["data"]["test_end"], 
                              config["data"]["lead"], config["data"]["memory_last"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        test_loader = DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False)

    elif dataset_type == "hov_sm": 
        test_data = HovSMDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["test_start"], config["data"]["test_end"], 
                              config["data"]["lead"], config["data"]["memory_last"], 
                              config["data"]["lat_range"], config["data"]["window_len"], config["data"]["residual"], config["data"]["transform"])
        # Create data loaders
        test_loader = DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False)

    elif dataset_type == "hov_two":
        test_data = HovTwoDataset(config["data"]["input_path"], config["data"]["target_path"],
                              config["data"]["test_start"], config["data"]["test_end"], 
                              config["data"]["lead"], config["data"]["memory_last"], 
                              config["data"]["lat_range"], config["data"]["transform"])
        # Create data loaders
        test_loader = DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False)
 
    return test_loader

# clarify the time dimension
def get_time_dimension(input_path, date_start, date_end, lead, mem=0):
    input_da = xr.open_dataarray(input_path[0]).sel(time=slice(date_start, date_end))

    date_start_date = datetime.strptime(date_start, "%Y-%m-%d") + timedelta(days=int(mem))
    date_end_date = datetime.strptime(date_end, "%Y-%m-%d")
    target_end_date = date_end_date + timedelta(days=lead)
    date_end_date = pd.Timestamp(input_da.time[-1].values).to_pydatetime()
    # redefine the end date if target_end_date is greater than the last date in the dataset
    if target_end_date > date_end_date:
        target_end_date = date_end_date
        date_end_date = target_end_date - timedelta(days=lead)

    return input_da.sel(time=slice(date_start_date.strftime("%Y-%m-%d"), date_end_date.strftime("%Y-%m-%d"))).time


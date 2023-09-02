# Modeling Irregular Time Series with Continuous Recurrent Units (CRUs)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import multiprocessing
from concurrent import futures

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import os

from lib.frenchpiezo_preprocessing import download_and_process_frenchpiezo
from lib.physionet_preprocessing import download_and_process_physionet
from lib.ushcn_preprocessing import download_and_process_ushcn
from lib.pendulum_generation import generate_pendulums
from lib.fdb_preparation import prepare_fdb


# new code component 
def load_data(args):
    file_path = f'data/{args.dataset}/'

    # Pendulum 
    if args.dataset == 'pendulum':

        if args.task == 'interpolation':
            if not os.path.exists(os.path.join(file_path, f'pend_interpolation_ir{args.impute_rate}.npz')):
                print(f'Generating pendulum trajectories and saving to {file_path} ...')
                generate_pendulums(file_path, task=args.task, impute_rate=args.impute_rate)

            train = Pendulum_interpolation(file_path=file_path, name=f'pend_interpolation_ir{args.impute_rate}.npz',
                                           mode='train', sample_rate=args.sample_rate,
                                           random_state=args.data_random_seed)
            valid = Pendulum_interpolation(file_path=file_path, name=f'pend_interpolation_ir{args.impute_rate}.npz',
                                           mode='valid', sample_rate=args.sample_rate,
                                           random_state=args.data_random_seed)

        elif args.task == 'regression':
            if not os.path.exists(os.path.join(file_path, 'pend_regression.npz')):
                print(f'Generating pendulum trajectories and saving to {file_path} ...')
                generate_pendulums(file_path, task=args.task)

            train = Pendulum_regression(file_path=file_path, name='pend_regression.npz',
                                        mode='train', sample_rate=args.sample_rate, random_state=args.data_random_seed)
            valid = Pendulum_regression(file_path=file_path, name='pend_regression.npz',
                                        mode='valid', sample_rate=args.sample_rate, random_state=args.data_random_seed)
        else:
            raise Exception('Task not available for Pendulum data')
        collate_fn = None

    # USHCN
    elif args.dataset == 'ushcn':
        if not os.path.exists(os.path.join(file_path, 'pivot_train_valid_1990_1993_thr4_normalize.csv')):
            print(f'Downloading USHCN data and saving to {file_path} ...')
            download_and_process_ushcn(file_path)


        if args.task in ('interpolation', 'extrapolation'):
            train = USHCN(file_path=file_path, name='pivot_train_valid_1990_1993_thr4_normalize.csv',
                          unobserved_rate=args.unobserved_rate,
                          impute_rate=args.impute_rate, sample_rate=args.sample_rate)
            valid = USHCN(file_path=file_path, name='pivot_test_1990_1993_thr4_normalize.csv',
                          unobserved_rate=args.unobserved_rate,
                          impute_rate=args.impute_rate, sample_rate=args.sample_rate)

        collate_fn = None

    elif args.dataset == 'ushcn_forecast':
        if not os.path.exists(os.path.join(file_path, 'pivot_train_valid_1990_1993_thr4_normalize.csv')):
            print(f'Downloading USHCN data and saving to {file_path} ...')
            download_and_process_ushcn(file_path)

        train = USHCNForecast(file_path=file_path, name='pivot_train_valid_1990_1993_thr4_normalize.csv',
                              unobserved_rate=args.unobserved_rate,
                              impute_rate=args.impute_rate, sample_rate=args.sample_rate, num_past=args.num_past,
                              num_future=args.num_future)
        valid = USHCNForecast(file_path=file_path, name='pivot_test_1990_1993_thr4_normalize.csv',
                              unobserved_rate=args.unobserved_rate,
                              impute_rate=args.impute_rate, sample_rate=args.sample_rate, num_past=args.num_past,
                              num_future=args.num_future)

        collate_fn = None

    # Physionet
    elif args.dataset == 'physionet':
        if not os.path.exists(os.path.join(file_path, 'norm_train_valid.pt')):
            print(f'Downloading Physionet data and saving to {file_path} ...')
            download_and_process_physionet(file_path)

        train = Physionet(file_path=file_path, name='norm_train_valid.pt')
        valid = Physionet(file_path=file_path, name='norm_test.pt')
        collate_fn = collate_fn_physionet

    # FrenchPiezo
    elif args.dataset == 'frenchpiezo':
        if not os.path.exists(os.path.join(file_path, 'cleaned_train.csv')):
            print(f"Downloading FrenchPiezo dataset and saving to {file_path} ...")
            download_and_process_frenchpiezo(file_path)

        train = FrenchPiezo(file_path=file_path, name='cleaned_train_valid.csv',
                            unobserved_rate=args.unobserved_rate,
                            impute_rate=args.impute_rate, sample_rate=args.sample_rate, num_past=args.num_past,
                            num_future=args.num_future)
        valid = FrenchPiezo(file_path=file_path, name='cleaned_test.csv',
                            unobserved_rate=args.unobserved_rate,
                            impute_rate=args.impute_rate, sample_rate=args.sample_rate, num_past=args.num_past,
                            num_future=args.num_future)
        collate_fn = None

    elif args.dataset == 'fdb':
        if not os.path.exists(os.path.join(file_path, 'processed', 'x_train.csv')):
            print("Preparing FDB dataset...")
            prepare_fdb(file_path, args.timestamp_freq)

        train = FDB(file_path=file_path, mode='train', sample_rate=args.sample_rate)
        valid = FDB(file_path=file_path, mode='valid', sample_rate=args.sample_rate)

        collate_fn = None

    train_dl = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=args.num_workers, pin_memory=args.pin_memory)
    valid_dl = DataLoader(valid, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=args.num_workers, pin_memory=args.pin_memory)

    return train_dl, valid_dl


# new code component 
class Pendulum_interpolation(Dataset):
    def __init__(self, file_path, name, mode, sample_rate=0.5, random_state=0):

        data = dict(np.load(os.path.join(file_path, name)))
        train_obs, train_targets, train_time_points, train_obs_valid, \
            test_obs, test_targets, test_time_points, test_obs_valid = subsample(
            data, sample_rate=sample_rate, imagepred=True, random_state=random_state)

        if mode == 'train':
            self.obs = train_obs
            self.targets = train_targets
            self.obs_valid = train_obs_valid
            self.time_points = train_time_points

        else:
            self.obs = test_obs
            self.targets = test_targets
            self.obs_valid = test_obs_valid
            self.time_points = test_time_points

        self.obs = np.ascontiguousarray(
            np.transpose(self.obs, [0, 1, 4, 2, 3])) / 255.0
        self.targets = np.ascontiguousarray(
            np.transpose(self.targets, [0, 1, 4, 2, 3])) / 255.0
        self.obs_valid = np.squeeze(self.obs_valid, axis=2)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.obs[idx, ...].astype(np.float64))
        targets = torch.from_numpy(self.targets[idx, ...].astype(np.float64))
        obs_valid = torch.from_numpy(self.obs_valid[idx, ...])
        time_points = torch.from_numpy(self.time_points[idx, ...])
        mask_truth = torch.ones_like(targets)
        return obs, targets, obs_valid, time_points, mask_truth


# new code component 
class Pendulum_regression(Dataset):
    def __init__(self, file_path, name, mode, sample_rate=0.5, random_state=0):

        data = dict(np.load(os.path.join(file_path, name)))
        train_obs, train_targets, test_obs, test_targets, train_time_points, \
            test_time_points = subsample(
            data, sample_rate=sample_rate, random_state=random_state)

        if mode == 'train':
            self.obs = train_obs
            self.targets = train_targets
            self.time_points = train_time_points
        else:
            self.obs = test_obs
            self.targets = test_targets
            self.time_points = test_time_points

        self.obs = np.ascontiguousarray(
            np.transpose(self.obs, [0, 1, 4, 2, 3])) / 255.0

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.obs[idx, ...].astype(np.float64))
        targets = torch.from_numpy(self.targets[idx, ...].astype(np.float64))
        time_points = torch.from_numpy(self.time_points[idx, ...])
        obs_valid = torch.ones_like(time_points, dtype=torch.bool)
        return obs, targets, time_points, obs_valid


class FDB(Dataset):
    def __init__(self, file_path, mode, impute_rate=None, sample_rate=0.5,
                 columns=range(0, 48), unobserved_rate=None):
        file_path = os.path.join(file_path, 'processed')
        self.params = list(columns)
        self.sample_rate = sample_rate
        self.impute_rate = impute_rate
        self.unobserved_rate = unobserved_rate
        if mode == 'train':
            self.x = pd.read_csv(os.path.join(file_path, 'x_train.csv'), low_memory=False)
            self.y = pd.read_csv(os.path.join(file_path, 'y_train.csv'), low_memory=False)
            self.time = pd.read_csv(os.path.join(file_path, 'time_train.csv'), low_memory=False)
        elif mode == 'valid':
            self.x = pd.read_csv(os.path.join(file_path, 'x_test.csv'), low_memory=False)
            self.y = pd.read_csv(os.path.join(file_path, 'y_test.csv'), low_memory=False)
            self.time = pd.read_csv(os.path.join(file_path, 'time_test.csv'), low_memory=False)

        self.label_columns = self.x.columns[self.x.columns.isin([str(i) for i in self.params])]
        self.num_features = 1
        self.set_ = mode

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        obs = torch.from_numpy(np.expand_dims(self.x.loc[idx].values, 1))
        # get only last value of the last num_future
        truth = torch.from_numpy(np.expand_dims(self.y.loc[idx].values, 1))
        # valori dei timestamp, originariamente servivano indici, magari funziona lo stesso
        time_points = torch.from_numpy(self.time.loc[idx].values)
        # forse struttura interna necessaria per il modello
        obs_valid = torch.ones_like(time_points, dtype=torch.bool)

        return torch.nan_to_num(obs), torch.nan_to_num(truth), torch.nan_to_num(time_points), obs_valid


class USHCNForecast(Dataset):

    def __init__(self, file_path, name, num_past, num_future, discard_window_threshold: float = 0.5, impute_rate=None,
                 sample_rate=0.5, columns=(0, 1, 2, 3, 4, 5, 6, 7),
                 unobserved_rate=None, year_range=4):
        self.params = ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'LONGITUDE', 'LATITUDE', 'ELEVATION']
        self.sample_rate = sample_rate
        self.impute_rate = impute_rate
        self.unobserved_rate = unobserved_rate
        self.year_range = year_range
        self.data = pd.read_csv(file_path + name).sort_values(['UNIQUE_ID', 'TIME_STAMP']).set_index('UNIQUE_ID')
        self.label_columns = self.data.columns[self.data.columns.isin([str(i) for i in columns])]
        self.num_past = num_past
        self.num_future = num_future
        self.window_size = self.num_past + self.num_future  # x, y
        self.discard_window_threshold = discard_window_threshold
        self.features = ['0', '1', '2', '3', '4', 'TIME_STAMP']
        self.numeric_features = ['0', '1', '2', '3', '4']
        self.num_features = len(self.features)
        self.num_numeric_features = len(self.numeric_features)
        self.past_features = self.num_past * self.num_features
        self.future_features = self.num_future * self.num_features
        self.set_ = 'train' if 'train' in name else 'test'
        self.save_file = os.path.join(file_path, f'{self.set_}_windows.npy')
        if os.path.exists(self.save_file):
            self.data_windows = np.load(self.save_file, allow_pickle=True)
        else:
            self.data_windows = self.generate_windows()
            np.save(self.save_file, self.data_windows)

    def parallel_generate_windows(self, data_df, station_id):
        tmp_window = list()
        data_windows = {station_id: list()}

        for _, row in data_df.iterrows():
            tmp_window.append(row[self.features].values)

            if len(tmp_window) == self.window_size:
                # get last num_future lists of values, but consider only the numeric features (not time)
                future_data = tmp_window[-self.num_future:]
                prcp_data = [future[0] for future in future_data]
                # check there are less than 50% NaNs in the future values, else scrap the window
                # sum nans for each array, then compute the total sum of the window,
                # divided by the total number of features in the window (5 per array)
                if pd.isna(prcp_data).sum(axis=0) / self.num_future < self.discard_window_threshold:
                    data_windows[station_id].append(tmp_window.copy())
                    # remove window with most NaNs
                    # most_empty_data_idx = np.argmax([pd.isna(future).sum(axis=0) for future in future_data])
                    # neg_idx_in_window = -(self.num_future - most_empty_data_idx)

                tmp_window.pop(0)  # more efficient than slicing tmp_window[1:]

        return data_windows

    def generate_windows(self):
        data_windows = dict()

        stations_ids = self.data.index.unique()

        for station_id in stations_ids:
            data_windows[station_id] = list()

        with futures.ProcessPoolExecutor() as executor:
            futures_ = list()
            for station_id in stations_ids:
                kwargs = {
                    'data_df': self.data.loc[station_id],
                    'station_id': station_id
                }

                futures_.append(executor.submit(self.parallel_generate_windows, **kwargs))
            done, not_done = futures.wait(futures_, return_when=futures.FIRST_EXCEPTION)

            futures_exceptions = [future.exception() for future in done]
            failed_futures = sum(map(lambda exception_: True if exception_ is not None else False,
                                     futures_exceptions))

            if failed_futures > 0:
                print("Could not create all time windows. Thrown exceptions: ")

                for exception in futures_exceptions:
                    print(exception)

                raise RuntimeError(f"Could not created windows, {failed_futures} processes failed.")

            if failed_futures == 0:
                print("Time windows created successfully.")

                # merge all the dictionaries
                data_windows = {key: value for future_ in done for key, value in future_.result().items()}

        print("Concatenating data...")
        # np.array(list(shared_data_windows.items()), dtype=np.float64, ndmin=4)
        # maybe np.concatenate?
        # data_windows = np.stack([np.array(data_window, dtype=np.float64) for data_window in data_windows.values()])
        data_windows = np.concatenate(
            [np.array(data_window, dtype=np.float64) for data_window in data_windows.values()])

        return data_windows

    def __len__(self):
        return len(self.data_windows)

    def __getitem__(self, idx):
        data_window = self.data_windows[idx]
        obs = torch.from_numpy(data_window[:self.num_past, :self.num_numeric_features])
        # obs = torch.from_numpy(data_window[:self.num_past, :3])
        # 0 on y-axis to forecast precipitations (PRCP)
        truth = torch.from_numpy(np.array([[data_window[-self.num_future:-1, 0].sum()]]))
        # valori dei timestamp, originariamente servivano indici, magari funziona lo stesso
        time_points = torch.from_numpy(data_window[:self.num_past, -1])
        # forse struttura interna necessaria per il modello
        obs_valid = torch.ones_like(time_points, dtype=torch.bool)

        return torch.nan_to_num(obs), torch.nan_to_num(truth), torch.nan_to_num(time_points), obs_valid


class FrenchPiezo(Dataset):

    def __init__(self, file_path, name, num_past, num_future, impute_rate=None,
                 sample_rate=0.5, columns=('p', 'e', 'tp'),
                 unobserved_rate=None):
        self.params = list(columns) + ['unique_id', 'time']
        self.sample_rate = sample_rate
        self.impute_rate = impute_rate
        self.unobserved_rate = unobserved_rate
        # sorting index (bss) and then time values
        self.data = pd.read_csv(file_path + name, low_memory=False).sort_values(by=['unique_id', 'time']).set_index('unique_id')
        self.label_columns = self.data.columns[self.data.columns.isin([str(i) for i in self.params])]
        self.num_past = num_past
        self.num_future = num_future
        self.window_size = self.num_past + self.num_future  # x, y
        self.features = ['p', 'e', 'tp', 'time']
        self.numeric_features = ['p', 'e', 'tp']
        self.num_features = len(self.features)
        self.num_numeric_features = len(self.numeric_features)
        self.past_features = self.num_past * self.num_features
        self.future_features = self.num_future * self.num_features
        self.set_ = 'train' if 'train' in name else 'test'
        self.save_file = os.path.join(file_path, f'{self.set_}_windows.npy')
        if os.path.exists(self.save_file):
            self.data_windows = np.load(self.save_file, allow_pickle=True)
        else:
            self.data_windows = self.generate_windows()
            np.save(self.save_file, self.data_windows)

    def parallel_generate_windows(self, data_df, station_id):
        tmp_window = list()
        data_windows = {station_id: list()}

        for _, row in data_df.iterrows():
            tmp_window.append(row[self.features].values)

            if len(tmp_window) == self.window_size:
                # get last num_future lists of values, but consider only the numeric features (not time)
                future_data = tmp_window[-self.num_future:]
                last_future_p = future_data[-1][0]
                # check there are less than 50% NaNs in the future values, else scrap the window
                # sum nans for each array, then compute the total sum of the window,
                # divided by the total number of features in the window (5 per array)
                if not pd.isna(last_future_p):
                    data_windows[station_id].append(tmp_window.copy())
                    # remove window with most NaNs
                    # most_empty_data_idx = np.argmax([pd.isna(future).sum(axis=0) for future in future_data])
                    # neg_idx_in_window = -(self.num_future - most_empty_data_idx)

                tmp_window.pop(0)  # more efficient than slicing tmp_window[1:]

        return data_windows

    def generate_windows(self):
        data_windows = dict()

        sensors_ids = self.data.index.unique()

        for sensor_id in sensors_ids:
            data_windows[sensor_id] = list()

        with futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures_ = list()
            for sensor_id in sensors_ids:
                kwargs = {
                    'data_df': self.data.loc[sensor_id],
                    'station_id': sensor_id
                }

                futures_.append(executor.submit(self.parallel_generate_windows, **kwargs))
            done, not_done = futures.wait(futures_, return_when=futures.FIRST_EXCEPTION)

            futures_exceptions = [future.exception() for future in done]
            failed_futures = sum(map(lambda exception_: True if exception_ is not None else False,
                                     futures_exceptions))

            if failed_futures > 0:
                print("Could not create all time windows. Thrown exceptions: ")

                for exception in futures_exceptions:
                    print(exception)

                raise RuntimeError(f"Could not created windows, {failed_futures} processes failed.")

            if failed_futures == 0:
                print("Time windows created successfully.")

                # merge all the dictionaries
                data_windows = {key: value for future_ in done for key, value in future_.result().items()}

        print("Concatenating data...")
        # np.array(list(shared_data_windows.items()), dtype=np.float64, ndmin=4)
        # maybe np.concatenate?
        # data_windows = np.stack([np.array(data_window, dtype=np.float64) for data_window in data_windows.values()])
        np_data_windows = [np.array(data_window, dtype=np.float64) for data_window in data_windows.values()
                           if data_window]
        data_windows = np.concatenate(np_data_windows)

        return data_windows

    def __len__(self):
        return len(self.data_windows)

    def __getitem__(self, idx):
        data_window = self.data_windows[idx]
        obs = torch.from_numpy(data_window[:self.num_past, :self.num_numeric_features])
        # get only last value of the last num_future
        truth = torch.from_numpy(np.array([[data_window[-1, 0]]]))
        # valori dei timestamp, originariamente servivano indici, magari funziona lo stesso
        time_points = torch.from_numpy(data_window[:self.num_past, -1])
        # forse struttura interna necessaria per il modello
        obs_valid = torch.ones_like(time_points, dtype=torch.bool)

        return torch.nan_to_num(obs), torch.nan_to_num(truth), torch.nan_to_num(time_points), obs_valid


# new code component
class USHCN(Dataset):
    params = ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']

    def __init__(self, file_path, name, impute_rate=None, sample_rate=0.5, columns=[0, 1, 2, 3, 4],
                 unobserved_rate=None, year_range=4):
        self.sample_rate = sample_rate
        self.impute_rate = impute_rate
        self.unobserved_rate = unobserved_rate
        self.year_range = year_range
        self.data = pd.read_csv(
            file_path + name).sort_values(['UNIQUE_ID', 'TIME_STAMP']).set_index('UNIQUE_ID')
        self.label_columns = self.data.columns[self.data.columns.isin(
            [str(i) for i in columns])]

    def __len__(self):
        return self.data.index.nunique()

    def subsample_time_points(self, sample, n_total_time_points, n_sample_time_points, seed=0):
        rng = np.random.RandomState(seed)

        choice = np.sort(rng.choice(n_total_time_points,
                                    n_sample_time_points, replace=False))
        return sample.loc[choice]

    def subsample_features(self, n_features, n_sample_time_points, seed=0):
        rng = np.random.RandomState(seed)

        # no subsampling
        if self.unobserved_rate is None:
            unobserved_mask = np.full(
                (n_features, n_sample_time_points), False, dtype=bool)

        # subsample such that it is equally probable that 1, 2, 3,.. features are missing per time point
        if self.unobserved_rate == 'stratified':
            unobserved_mask = create_unobserved_mask(
                n_features, n_sample_time_points)

        # subsample features based on overall rate (most time points will have 1, 2 features missing, few will have more missing)
        elif isinstance(self.unobserved_rate, float) or isinstance(self.unobserved_rate, int):
            assert 0 <= self.unobserved_rate < 1, 'Unobserved rate must be between 0 and 1.'
            unobserved_mask = ~ (
                    rng.rand(n_sample_time_points, n_features) > self.unobserved_rate)

        else:
            raise Exception('Unobserved mode unknown')
        return unobserved_mask

    def get_data_based_on_impute_rate(self, sample, unobserved_mask, n_features, n_sample_time_points):

        # task is not imputation (i.e. extrapolation or one-step-ahead prediction)
        if self.impute_rate is None:
            sample[self.label_columns] = np.where(
                unobserved_mask, np.nan, sample[self.label_columns])
            obs = torch.tensor(sample.loc[:, self.label_columns].values)
            targets = obs.clone()
            # valid if we have at least one dim observed
            obs_valid = np.sum(unobserved_mask, axis=-1) < n_features

        # impute missing time step
        elif isinstance(self.impute_rate, float):
            assert 0 <= self.impute_rate < 1, 'Imputation rate must be between 0 and 1.'
            sample[self.label_columns] = np.where(
                unobserved_mask, np.nan, sample[self.label_columns])
            obs = torch.tensor(sample.loc[:, self.label_columns].values)
            targets = obs.clone()
            # remove time steps that have to be imputed
            obs_valid = torch.rand(n_sample_time_points) >= self.impute_rate
            obs_valid[:10] = True
            obs[~obs_valid] = np.nan

        else:
            raise Exception('Impute mode unknown')

        time_points = torch.tensor(sample.loc[:, 'TIME_STAMP'].values)
        mask_targets = 1 * ~targets.isnan()
        mask_obs = ~ unobserved_mask

        return torch.nan_to_num(obs), torch.nan_to_num(targets), obs_valid, time_points, mask_targets, mask_obs

    def __getitem__(self, idx):
        sample = self.data.loc[idx, :].reset_index(drop=True)
        n_total_time_points = len(sample)
        n_sample_time_points = int(365 * self.year_range * self.sample_rate)
        n_features = len(self.label_columns)

        # subsample time points to increase irregularity
        sample = self.subsample_time_points(
            sample, n_total_time_points, n_sample_time_points)

        # subsample features to increase partial observability
        unobserved_mask = self.subsample_features(
            n_features, n_sample_time_points)

        # create masks and target based on if/what kind of imputation
        obs, targets, obs_valid, time_points, mask_targets, mask_obs = \
            self.get_data_based_on_impute_rate(
                sample, unobserved_mask, n_features, n_sample_time_points)

        return obs, targets, obs_valid, time_points, mask_targets, mask_obs


# new code component 
class Physionet(Dataset):
    def __init__(self, file_path, name):
        self.data = torch.load(os.path.join(
            file_path, name), map_location='cpu')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# new code component 
def collate_fn_physionet(batch):
    obs = [obs for patient_id, time_points, obs, mask, label in batch]
    time_points = [time_points for patient_id,
    time_points, obs, mask, label in batch]
    mask = [mask for patient_id, time_points, obs, mask, label in batch]

    obs = pad_sequence(obs, batch_first=True).to(
        device='cpu', dtype=torch.double)
    time_points = pad_sequence(time_points, batch_first=True).to(
        device='cpu', dtype=torch.double)
    mask_obs = pad_sequence(mask, batch_first=True).to(device='cpu')
    targets = obs.clone()
    mask_targets = mask_obs.clone()

    # create obs_valid mask such that update step will be skipped on padded time points
    obs_valid = ~torch.all(mask_obs == 0, dim=-1)

    return obs, targets, obs_valid, time_points, mask_targets, mask_obs


# new code component 
def subsample(data, sample_rate, imagepred=False, random_state=0):
    train_obs, train_targets, test_obs, test_targets = data["train_obs"], \
        data["train_targets"], data["test_obs"], data["test_targets"]
    seq_length = train_obs.shape[1]
    train_time_points = []
    test_time_points = []
    n = int(sample_rate * seq_length)

    if imagepred:
        train_obs_valid = data["train_obs_valid"]
        test_obs_valid = data["test_obs_valid"]
        data_components = train_obs, train_targets, test_obs, test_targets, train_obs_valid, test_obs_valid
        train_obs_sub, train_targets_sub, test_obs_sub, test_targets_sub, train_obs_valid_sub, test_obs_valid_sub = [
            np.zeros_like(x[:, :n, ...]) for x in data_components]
    else:
        data_components = train_obs, train_targets, test_obs, test_targets
        train_obs_sub, train_targets_sub, test_obs_sub, test_targets_sub = [
            np.zeros_like(x[:, :n, ...]) for x in data_components]

    for i in range(train_obs.shape[0]):
        rng_train = np.random.default_rng(random_state + i + train_obs.shape[0])
        choice = np.sort(rng_train.choice(seq_length, n, replace=False))
        train_time_points.append(choice)
        train_obs_sub[i, ...], train_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [train_obs, train_targets]]
        if imagepred:
            train_obs_valid_sub[i, ...] = train_obs_valid[i, choice, ...]

    for i in range(test_obs.shape[0]):
        rng_test = np.random.default_rng(random_state + i)
        choice = np.sort(rng_test.choice(seq_length, n, replace=False))
        test_time_points.append(choice)
        test_obs_sub[i, ...], test_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [test_obs, test_targets]]
        if imagepred:
            test_obs_valid_sub[i, ...] = test_obs_valid[i, choice, ...]

    train_time_points, test_time_points = np.stack(
        train_time_points, 0), np.stack(test_time_points, 0)

    if imagepred:
        return train_obs_sub, train_targets_sub, train_time_points, train_obs_valid_sub, test_obs_sub, test_targets_sub, test_time_points, test_obs_valid_sub
    else:
        return train_obs_sub, train_targets_sub, test_obs_sub, test_targets_sub, train_time_points, test_time_points


# new code component
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# new code component 
def discretize_data(obs, targets, time_points, obs_valid, n_bins=10, take_always_closest=True):
    N = obs.shape[0]
    T_max = time_points.max()
    bin_length = T_max / n_bins
    obs_valid = np.squeeze(obs_valid)

    # define the bins
    _, bin_edges = np.histogram(time_points, bins=n_bins)

    # get the center of each bin
    bin_length = bin_edges[1] - bin_edges[0]
    bin_center = bin_edges + bin_length / 2

    # get the timepoint, obs etc that is closest to the bin center
    tp_all = []
    obs_valid_all = []
    obs_all = np.zeros((N, n_bins, 24, 24, 1), dtype='uint8')
    targets_all = np.zeros((N, n_bins, 24, 24, 1), dtype='uint8')
    for i in range(N):
        tp_list = []
        obs_valid_list = []
        for j in range(n_bins):
            sample_tp = time_points[i, :]
            center = bin_center[j]
            idx = find_nearest(sample_tp, center)
            if (bin_edges[j] <= sample_tp[idx] <= bin_edges[j + 1]) or take_always_closest:
                tp_list.append(sample_tp[idx])
                obs_valid_list.append(obs_valid[i, idx])
                obs_all[i, j, ...] = obs[i, idx, ...]
                targets_all[i, j, ...] = targets[i, idx, ...]
            else:
                tp_list.append(np.nan)
                obs_valid_list.append(False)
                obs_all[i, j, ...] = 0
                targets_all[i, j, ...] = 0

        tp_all.append(tp_list)
        obs_valid_all.append(obs_valid_list)

    return obs_all, targets_all, np.array(tp_all), np.array(obs_valid_all)


# new code component 
def create_unobserved_mask(n_col, T, seed=0):
    # subsamples features (used to experiment with partial observability on USHCN)
    rng = np.random.RandomState(seed)
    mask = []
    for i in range(T):
        mask_t = np.full(n_col, False, dtype=bool)
        n_unobserved_dimensions = rng.choice(
            n_col, 1, p=[0.6, 0.1, 0.1, 0.1, 0.1])
        unobserved_dimensions = rng.choice(
            n_col, n_unobserved_dimensions, replace=False)
        mask_t[unobserved_dimensions] = True
        mask.append(mask_t)
    return np.array(mask)


# new code component
def align_output_and_target(output_mean, output_var, targets, mask_targets):
    # removes last time point of output and first time point of target for one-step-ahead prediction
    output_mean = output_mean[:, :-1, ...]
    output_var = output_var[:, :-1, ...]
    targets = targets[:, 1:, ...]
    mask_targets = mask_targets[:, 1:, ...]
    return output_mean, output_var, targets, mask_targets


# new code component
def adjust_obs_for_extrapolation(obs, obs_valid, obs_times=None, cut_time=None):
    obs_valid_extrap = obs_valid.clone()
    obs_extrap = obs.clone()

    # zero out last half of observation (used for USHCN)
    if cut_time is None:
        n_observed_time_points = obs.shape[1] // 2
        obs_valid_extrap[:, n_observed_time_points:, ...] = False
        obs_extrap[:, n_observed_time_points:, ...] = 0

    # zero out observations at > cut_time (used for Physionet)
    else:
        mask_before_cut_time = obs_times < cut_time
        obs_valid_extrap *= mask_before_cut_time
        obs_extrap = torch.where(obs_valid_extrap[:, :, None], obs_extrap, 0.)

    return obs_extrap, obs_valid_extrap

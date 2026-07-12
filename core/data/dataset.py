# coding: utf-8
 
#
# updated: Mar. 25, 2022
# Filled non-existing raw features with non-zero after encoded from encoders

"""
Data pre-processing
##########################
"""
import os
import logging

import pandas as pd


# import lmdb


class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = logging.getLogger("nexusrec")

        # data path & files
        self.dataset_name = config["dataset"]
        self.dataset_path = os.path.abspath(config["data_path"] + self.dataset_name)

        # dataframe
        self.uid_field = self.config["USER_ID_FIELD"]
        self.iid_field = self.config["ITEM_ID_FIELD"]
        self.splitting_label = self.config["inter_splitting_label"]

        # load rating file from data path (skip if DataFrame is provided directly)
        if df is not None:
            self.df = df
        else:
            # Only check file existence when loading from disk
            file_path = os.path.join(self.dataset_path, self.config["interaction_file"])
            if not os.path.isfile(file_path):
                raise ValueError("File {} not exist".format(file_path))
            self.load_inter_graph(config["interaction_file"])

        # Check for empty DataFrame
        if len(self.df) == 0:
            raise ValueError(
                f"Empty interaction dataframe for dataset at {self.dataset_path}. "
                "Please ensure the interaction file contains valid data."
            )

        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1
        self.inter_num = len(self.df)

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        # read_csv(usecols=cols) already raises ValueError if any required column
        # is missing, so the loaded columns are always a subset of cols — no
        # post-load presence check needed.
        self.df = pd.read_csv(
            inter_file, usecols=cols, sep=self.config["field_separator"]
        )

    def split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)  # no use again
            dfs.append(temp_df)
        if self.config["filter_out_cold_start_users"]:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    @classmethod
    def from_dataframe(cls, config, df, item_num=None, user_num=None):
        """Create a RecDataset from an existing DataFrame without disk I/O.

        Use this instead of ``RecDataset(config, df)`` when the DataFrame is
        already in memory and the interaction CSV should not be re-read.
        """
        obj = object.__new__(cls)
        obj.config = config
        obj.logger = logging.getLogger("nexusrec")
        obj.dataset_name = config["dataset"]
        obj.dataset_path = os.path.abspath(config["data_path"] + obj.dataset_name)
        obj.uid_field = config["USER_ID_FIELD"]
        obj.iid_field = config["ITEM_ID_FIELD"]
        obj.splitting_label = config["inter_splitting_label"]
        obj.df = df
        obj.item_num = item_num if item_num is not None else (
            int(max(df[obj.iid_field].values)) + 1 if len(df) > 0 else 0
        )
        obj.user_num = user_num if user_num is not None else (
            int(max(df[obj.uid_field].values)) + 1 if len(df) > 0 else 0
        )
        obj.inter_num = len(df)
        return obj

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_df``, and all the other attributes the same.

        Args:
            new_df (pandas.DataFrame): The new interaction feature need to be updated.

        Returns:
            :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
        """
        return RecDataset.from_dataframe(
            self.config, new_df,
            item_num=self.item_num, user_num=self.user_num,
        )

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace."""
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info, avg = [], []

        inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])

        tmp_user_num, tmp_item_num = len(uni_u), len(uni_i)
        avg_actions_of_users = inter_num / tmp_user_num if tmp_user_num else 0
        avg_actions_of_items = inter_num / tmp_item_num if tmp_item_num else 0

        info.append(f"#Users: {tmp_user_num},")
        info.append(f"#Items: {tmp_item_num},")
        info.append(f"#Inters: {inter_num},")
        if tmp_user_num and tmp_item_num:
            sparsity = 1 - inter_num / tmp_user_num / tmp_item_num
            info.append(f"Sparsity: {sparsity * 100:.4f}%,")

        avg.append(f"#Avg User Actions: {avg_actions_of_users:.4f},")
        avg.append(f"#Avg Item Actions: {avg_actions_of_items:.4f},")

        info.extend(avg)

        return " ".join(info)

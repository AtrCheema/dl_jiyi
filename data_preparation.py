
import pandas as pd
import os


from utils import do_plot

# TODO check for outliers in data


class DATA(object):
    """
    `attributes`
    -----------
     df:        dataframe consisting of input data
     all_outs:  list consisting of names of all possible outputs
     otus:     list consisting of names of all OTUs
     args:     list consisting of names of all possible ARGs
     prosp_ins: list consisting of names of all possible inputs. This is excludes all_outs.
     rain:      list consisting of names of differnet rainfall data
     misc_in
     env:      list consisting of names of environmental data

    `methods`
    -----------
    plot_input: plots all the input and output data.
    """

    def __init__(self, freq='30min', verbosity=1):

        self.freq = freq
        self.data_dir = os.path.join(os.getcwd(), 'data')
        self.verbosity = verbosity

    @property
    def args(self):
        return ['Total_args', 'tetx', 'sul1', 'blaTEM', 'aac']

    @property
    def otus(self):
        if hasattr(self, 'columns'):
            return [otu for otu in self.columns if 'otu' in otu]
        else:
            raise ValueError("use get_df method first to get data")

    @property
    def all_outs(self):
        return self.otus + self.args

    @property
    def prosp_ins(self):
        if hasattr(self, 'columns'):
            return [col for col in self.columns if col not in self.all_outs]
        else:
            raise ValueError("use get_df method first to get data")

    @property
    def rains(self):
        return [pcp for pcp in self.prosp_ins if 'pcp' in pcp]  # rainfall data

    @property
    def misc_in(self):
        return ['ecoli', '16s', 'inti1']

    @property
    def evn(self):
        return [d for d in self.prosp_ins if d not in self.rains + self.misc_in]  # environmental data

    def get_df(self):
        fname = 'all_data_' + self.freq + '.xlsx'
        fpath = os.path.join(self.data_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_excel(fpath)
            index_col = [c for c in df.columns if 'Date_Time' in c][0]
            df.index = df[index_col]
            df.pop(index_col)
            if self.verbosity > 0:
                print('file with {} freq is available as {}'.format(self.freq, fpath))
        else:
            # 20180601 - 201909-30, (5856,3)
            wat_df = self.load_wat_data()
            if 'Date_Time' in wat_df.columns:
                wat_df.pop('Date_Time')

            # 201806 - 20190906,  (1176, 8)
            env_df = self.load_env_data()
            if 'Date_Time' in env_df.columns:
                env_df.pop('Date_Time')

            # (295, 12)
            obs_df = self.load_obs_data()

            df = pd.concat([wat_df, env_df, obs_df],
                           axis=1, join_axes=[env_df.index])

            df.to_excel(fpath)

        setattr(self, 'columns', list(df.columns))

        return df

    def plot_data(self, df):
        obs_logy = False
        for out in self.all_outs:
            if out in self.args:
                obs_logy = True
            idx = 0
            for in_type in [self.rains, self.misc_in, self.env]:
                plt_df = pd.DataFrame(index=df.index)
                for _in in in_type:
                    plt_df[_in] = df[_in]
                plt_df[out] = df[out]

                plt_df = remove_chunk('20180630', '20190516', plt_df)

                do_plot(plt_df, plt_df.columns, save_name='results/' + out + '_' + str(idx), obs_logy=obs_logy)
                idx += 1
        return

    def load_obs_data(self, desired_output=None, sheets=None):
        """ gwangali site data at 1 hour frequency, contains all data, input and output ts"""
        if desired_output is None:
            desired_output = ['ecoli', '16s', 'inti1', 'Total_args', 'tetx_coppml', 'sul1_coppml', 'blaTEM_coppml',
                              'aac_coppml', 'Total_otus', 'otu_5575', 'otu_273', 'otu_94']
        columns = ['pcp1', 'pcp3', 'pcp6', 'pcp12', 'tide', 'W_temp', 'sal', 'wind_sp',
                   'wind_dir', 'atm_temp', 'atm_p', 'mslp_hpa', 'rel_hum',  'ecoli', '16s', 'inti1', 'Total_args',
                   'tetx_coppml', 'sul1_coppml', 'blaTEM_coppml', 'aac_coppml', 'Total_otus', 'otu_5575',
                   'otu_273', 'otu_94']

        if sheets is None:
            sheets = ['201806', '201905', '201908_1', '201908_2', '201909']

        fpath = os.path.join(self.data_dir, 'Time_series_data.xlsx')
        haupt_df = pd.DataFrame()

        for sheet in sheets:
            df = pd.read_excel(fpath, sheet_name=sheet)
            date = df['date'].astype(str)
            time = df['time'].astype(str)
            idx1 = date + ' ' + time
            if sheet == '201909':
                yearfirst = True
                dayfirst = False
            else:
                yearfirst = False
                dayfirst = True
            idx = pd.to_datetime(idx1, yearfirst=yearfirst, dayfirst=dayfirst)
            df.index = idx
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError
            df.index.freq = pd.infer_freq(df.index)
            if 'date' in df.columns:
                df.pop('date')
            if 'time' in df.columns:
                df.pop('time')
            if 'Sample No.' in df.columns:
                df.pop('Sample No.')
            df.columns = columns

            haupt_df = pd.concat([haupt_df, df])
        return haupt_df[desired_output]

    def load_wat_data(self, desired_output=None):
        """1 minute data at a site 10 km away from Gwangali"""

        desired_file = os.path.join(self.data_dir, 'wat_data_' + self.freq + '.csv')
        if os.path.exists(desired_file):
            if self.verbosity > 0:
                print('file with {} freq is available as {}'.format(self.freq, desired_file))
            df = pd.read_csv(desired_file)
            df.index = pd.to_datetime(df['Date_Time'])
            df.pop('Date_Time')
            return df
        else:
            if desired_output is None:
                desired_output = ['tide_cm', 'wat_temp_c', 'sal_psu']

            _d_dir = os.path.join(os.getcwd(), 'data\\busan')
            _files = [f for f in os.listdir(_d_dir) if f.endswith('txt')]
            haupt_df = pd.DataFrame()
            for fname in _files:
                fpath = os.path.join(_d_dir, fname)
                _df = pd.read_csv(fpath, comment='#', sep='\t', na_values='-')
                idx = pd.to_datetime(_df['Date_Time'])
                _df.index = idx
                _df.index.freq = pd.infer_freq(_df.index)
                if not isinstance(_df.index, pd.DatetimeIndex):
                    raise ValueError
                if not isinstance(_df.index.freqstr, str):
                    raise ValueError
                if self.verbosity > 0:
                    print(fpath, _df.index.freq)

                ts = pd.DataFrame()
                for col in desired_output:  # _df.columns:
                    to_resample = pd.DataFrame(_df[col])
                    _ts = down_sample(to_resample, col, self.freq, idx=None, verbosity=self.verbosity)
                    ts = pd.concat([ts, _ts], axis=1, sort=False)

                haupt_df = pd.concat([haupt_df, ts])
            final_df = haupt_df[desired_output]
            final_df.to_csv(desired_file)
            return final_df

    def load_env_data(self, desired_output=None):
        """ loads 1 minute data from a site located 10 km from Gwangali.
        The data is from two sites. For each data, the number of nans are comapred and data from that site is accepted
        which contains lower number of nans."""

        desired_file = os.path.join(self.data_dir, 'env_data_' + self.freq + '.csv')
        if os.path.exists(desired_file):
            df = pd.read_csv(desired_file)
            index_col = [c for c in df.columns if 'Date_Time' in c][0]
            df.index = pd.to_datetime(df[index_col])
            df.pop(index_col)
            if self.verbosity > 0:
                print('file with {} freq is available as {}'.format(self.freq, desired_file))
            return df
        else:
            if desired_output is None:
                desired_output = ['air_temp_c', 'pcp_mm', 'wind_dir_deg', 'wind_speed_mps',
                                  'air_p_hpa', 'mslp_hpa', 'rel_hum']
            cols = ["Point_No", "Point", "Date_Time",
                    "air_temp_c", "pcp_mm", "wind_dir_deg", "wind_speed_mps",
                    "air_p_hpa", "mslp_hpa", "rel_hum"]

            d_dir = os.path.join(self.data_dir, 'AWS_data')
            files = [f for f in os.listdir(d_dir) if f.endswith('txt')]
            haupt_df = pd.DataFrame()

            for fname in files:
                fpath = os.path.join(d_dir, fname)
                file_df = pd.read_csv(fpath, sep='\t')

                col_df = pd.DataFrame()
                col_df_ds = None
                for col in cols:  # for each columns in file
                    col_df1 = pd.DataFrame()
                    col_df2 = pd.DataFrame()
                    #  each file contains samples from two sites whose columns have suffix 1 and 2
                    for site in ['1', '2']:
                        _idx = file_df['Date_Time' + site]
                        _col = col + site
                        _df1 = pd.DataFrame(file_df[_col])
                        _df1 = assign_freq(_df1, _idx, fname + ' ' + col, force_freq='1min', verbosity=self.verbosity-1,
                                           print_only=False)
                        if site == '1':
                            col_df1 = pd.concat([col_df1, _df1], axis=1, sort=False)
                        else:
                            col_df2 = pd.concat([col_df2, _df1], axis=1, sort=False)
                    nans_1 = int(col_df1.isna().sum())
                    nans_2 = int(col_df2.isna().sum())
                    if nans_1 > nans_2:
                        col_df2.columns = [col]
                        col_df2 = assign_freq(col_df2, _idx, fname + ' ' + col, force_freq='1min',
                                              verbosity=self.verbosity)
                        col_df = pd.concat([col_df, col_df2], axis=1, sort=False)
                        if self.verbosity > 1:
                            print('for {}, {} is chosen which had {} nans while {} had {} nans'.format(col, 2, nans_2,
                                                                                                       1, nans_1))
                    else:
                        col_df1.columns = [col]
                        col_df1 = assign_freq(col_df1, _idx, fname + ' ' + col, force_freq='1min',
                                              verbosity=self.verbosity-1)
                        col_df = pd.concat([col_df, col_df1], axis=1, sort=False)
                        if self.verbosity > 1:
                            print('for {}, {} is chosen which had {} nans while {} had {} nans'
                                  .format(col, 1, nans_1, 2, nans_2))

                    col_df_ds = down_sample(col_df, col, self.freq, _idx, self.verbosity, fname=fname)

                # here only printing, not forcing it. If index does not have frequency yet, then we are doomed
                col_df_mit_freq = assign_freq(col_df_ds, file=fname, verbosity=self.verbosity, print_only=True)

                haupt_df = pd.concat([haupt_df, col_df_mit_freq])

            final_df = haupt_df[desired_output]
            final_df.to_csv(desired_file)
            return final_df


def assign_freq(df, index=None, file=None, force_freq=None,  verbosity=1, print_only=False):
    if not print_only:
        if index is None:
            idx = pd.to_datetime(df['Date_Time'])
        else:
            idx = index
        df.index = idx
    df.index.freq = pd.infer_freq(df.index)
    if df.index.freq is None:
        if force_freq is not None:
            df.index.freq = force_freq
    if verbosity > 1:
        print('in file {} frequency is {}'.format(file, df.index.freq))
    return df


def down_sample(data_frame, data_name, desired_freq, idx,  verbosity=1, fname=None):

    if idx is not None:
        data_frame.index = pd.to_datetime(idx)
        data_frame.index.freq = pd.infer_freq(data_frame.index)
    elif 'Date_Time' in data_frame.columns:
        data_frame.index = data_frame['Date_Time']

    if verbosity > 1:
        print('dataframe to downsample has {} shape and {} columns'.format(data_frame.shape, list(data_frame.columns)))

    if not isinstance(data_frame.index, pd.DatetimeIndex):
        raise TypeError("index of data_frame must be of Datetime")

    out_freq = desired_freq
    data_frame = data_frame.copy()
    old_freq = data_frame.index.freq
    if old_freq is None:
        raise TypeError("Index of datafrmae {} to downsample has no initial frequency in file {}"
                        .format(data_name, fname))

    if verbosity > 0:
        print('downsampling {} data from {} min to {}'.format(data_name, old_freq, out_freq))
    # e.g. from hourly to daily
    if data_name in ['air_temp_c', 'rel_hum', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa',
                     'tide_cm', 'wat_temp_c', 'sal_psu']:
        return data_frame.resample(out_freq).mean()
    elif data_name in ['pcp_mm', 'ss_gpl', 'solar_rad']:
        return data_frame.resample(out_freq).sum()
    else:
        if verbosity > 0:
            print('not resampling ', data_name)
        return data_frame


def load_1min_gwangali_sitewise():
    """ loads 1 minute data from 2 sites close to gwangali. This does not contain wat temp, tide and salinity"""
    _d_dir = os.path.join(os.getcwd(), 'data\\AWS_data\\site_wise')
    _files = [f for f in os.listdir(_d_dir) if f.endswith('txt')]
    a_files, b_files = [], []
    for f in _files:
        if f.split('.')[0].endswith("_a"):
            a_files.append(f)
        elif f.split('.')[0].endswith('_b'):
            b_files.append(f)

    haupt_df = pd.DataFrame()
    for af, bf in zip(a_files, b_files):
        _f = os.path.join(_d_dir, af)
        _df = pd.read_csv(_f)
        _df.index = pd.to_datetime(_df['Date_Time1'])
        _df.index.freq = pd.infer_freq(_df.index)
        if _df.index.freq is None:
            _f = os.path.join(_d_dir, bf)
            _df = pd.read_csv(_f)
            _df.index = pd.to_datetime(_df['Date_Time2'])
            _df.index.freq = pd.infer_freq(_df.index)
            print(_df.index.freq, ' taken from ', bf)
        print(_df.index.freq)
        haupt_df = pd.concat([haupt_df, _df])

    return haupt_df


# function to remove a chunk of rows from dataframe
def remove_chunk(_st, _en, _df):
    st_indx = _df.index[0]
    en_indx = _df.index[-1]
    dfs = _df[st_indx: _st]
    dfe = _df[_en: en_indx]
    df_new = pd.concat([dfs, dfe])
    return df_new


# if __name__ == "__main__":
#     data = DATA()
    # data.plot_data()

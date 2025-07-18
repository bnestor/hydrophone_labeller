"""

"""

import torch
import torchaudio
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

import os
import glob
import warnings
import time
# from copy import deepcopy



class FileDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, processed_outputs,  receptive_field=16384):
        self.audio_files = audio_files
        self.processed_outputs = processed_outputs
        self.receptive_field = receptive_field

        self.spectrogrammer = torchaudio.transforms.Spectrogram()

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        filename, start_time = self.audio_files[idx]
        try:
            data, sr = torchaudio.load(filename)
        except:
            time.sleep(1)
            print('failed',filename)
            data, sr = torchaudio.load(filename)
        data = torchaudio.functional.resample(data, sr, 32000)

        try:
            data = torchaudio.functional.highpass_biquad(data, 32000,  1000,)
        except:
            print('failed',filename)
            print(data.shape)
            raise

        if data.shape[0]==2:
            # check if one side is all nans?
            if torch.sum(data[0])==0:
                data=data[1]
            elif torch.sum(data[1])==0:
                print("it was all 0")
                raise
                data=data[0]
            else:
                data=data.mean(0, keepdim=True)

        # spec = self.spectrogrammer(data)

        # height, width = spec.shape

        # # get only the appropriate width corresponding to start_time
        # convert_data_index_to_spec_index = lambda x: int(x*32000/data.shape[1]*width)
        # start_index = convert_data_index_to_spec_index(start_time)
        # end_index = convert_data_index_to_spec_index(start_time+15)

        # end_index= min(end_index, width)

        # spec = spec[:, start_index:end_index]

        # return {'spec':spec, 'idx':idx}

        # check if any data is nan
        if torch.isnan(data).any():
            raise Exception(filename)




        if len(data.shape)==1:
            if self.mono:
                data =data.view(1, -1)
            else:
                data = data.view(2, -1)

        # data_orig = deepcopy(data)


        if self.receptive_field is not None:
            try:data.shape[1]
            except:
                print()
                print(data.shape)
                print()
                data.shape[1]
            if data.shape[1]>self.receptive_field:
                data = torch.split(data, self.receptive_field, dim=-1)
                try:
                    attention_mask  = torch.split(attention_mask, self.receptive_field, dim=-1)
                except:
                    pass
            elif len(data)<self.receptive_field:
                print()
                print(data.shape)
                print(len(data), self.receptive_field)
                print('failed on ', filename)
                raise NotImplementedError
        # drop the last of the data if it is not the same shape as the rest.
        if isinstance(data, tuple):
            if len(data)>1:
                if all(data[i].shape==data[0].shape for i in range(len(data))):
                    # they are all the same shape
                    data = torch.cat([d.unsqueeze(0) for d in data], dim=0)
                else:
                    # drop the last batch
                    data = data[:len(data)-1]
                    data = torch.cat([d.unsqueeze(0) for d in data], dim=0)
                    # print([d.shape for d in data])
        else:
            print(data.shape)
            raise


        # check if any data is nan
        if torch.isnan(data).any():
            raise Exception(filename)
            

        # print('1',data.shape)
        data = self.spectrogrammer(data) # ... freq, time
        if len(data.shape)>4:
            data = data.squeeze(1)



        data_norm = data.mean(dim=2, keepdim=True)
        # replace 0s in the mean with 1s
        data_norm[data_norm==0]=1
        data = data / data_norm

        if len(data.shape)==3:
            data = data.squeeze(-1).unsqueeze(1).expand(-1,3,-1,-1)
        else:
            data = data.squeeze(-1).expand(-1,3,-1,-1)
        # print('3',data.shape)





       
        packet = {'data': data.float(),
                  'item':torch.tensor(idx),
                  'start_time':torch.tensor(start_time)}

        return packet
    




        

        

    

def prepare_data(audio_files,  processed_outputs, start_segments=None, start_time_column_name='start_time',filename_column_name='filename'):
    """
    Args:
        audio_files (str or list): **required**. The directory where the audio files are stored.
        start_segments (str, optional): A csv with the filename and the start time for each clip. If this is not provided, the first 15 seconds are used
        processed_outputs (str): **required**. The directory where the processed audio files will be saved
    """


    # resolve all of the files
    # collect the data
    if isinstance(audio_files, str):
        if '*' in audio_files:
            filenames = glob.glob(audio_files, recursive=True)
        else:
            assert os.path.isfile(audio_files), print(audio_files, 'does not exist')
            filenames = [audio_files]
    elif isinstance(audio_files, list): # or isinstance(audio_files, listconfig.ListConfig):
        filenames=[]
        for f in audio_files:
            if '*' in f:
                print('globbing', f)
                filenames.extend(glob.glob(f, recursive=True))
            else:
                assert os.path.isfile(f), print(f, 'does not exist')
                filenames.append(f)

    audio_files = filenames


    assert len(audio_files) > 0, print('No audio files found')

    if start_segments is not None:
        start_segments = pl.read_csv(start_segments)
        # apply os.path.basename to the filename column
        start_segments = start_segments.with_columns(pl.col(filename_column_name).map_elements(lambda x: os.path.basename(x), return_dtype=pl.String).alias(filename_column_name))
        # get a dict of filename:segment_start
        start_segments = dict(zip(start_segments.select(filename_column_name).to_dict()[filename_column_name], start_segments.select(start_time_column_name).to_dict()[start_time_column_name]))

    audio_files_with_start_time = []
    for filename in audio_files:
        if start_segments is None:
            audio_files_with_start_time.append((filename, 0))
        elif os.path.basename(filename) in start_segments.keys():
            audio_files_with_start_time.append((filename, start_segments[os.path.basename(filename)]))
        else:
            audio_files_with_start_time.append((filename, 0))

        
    dataset = FileDataset(audio_files_with_start_time, processed_outputs)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3)

    for batch in dataloader:

        # batch = {k:v.float().to(device) for k, v in batch.items()}
        item = int(batch['item'].cpu().data.numpy())


        # optionally use best_segment
        end_index=None
        fname = filenames[item]
        
        start_index = batch['start_time']
        print(start_index)
        # print(len(start_index))
        if len(start_index)>0:
            start_index=start_index[0]*32000
            if start_index is not None:
                if not(np.isnan(start_index)):
                    spec = batch['data'].squeeze().cpu().data.numpy()
                    # the files are 5 minutes long
                    end_index = int(((start_index/32000) +15) /300 *len(spec) )
                    start_index = int((start_index/32000) /300 *len(spec) )
                    # print(start_index, end_index)
                    


        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)

        spec = batch['data'].squeeze().cpu().data.numpy()
        if end_index is not None:
            spec = np.concatenate([spec[i,0].squeeze() for i in range(start_index, end_index)], axis=-1)
        else:
            spec = np.concatenate([spec[i,0].squeeze() for i in range(len(spec))], axis=-1)
        #im = ax.imshow(librosa.power_to_db(spec), origin="lower", aspect='auto')
        im = ax.imshow(power_to_db(spec), origin="lower", aspect='auto')


        
        ax.set_title(f"{os.path.basename(filenames[item])}") #, lat:{lat} lon:{lon}, depth:{depth}")
        ax.set_ylabel('Frequency (kHz)')
        ax.set_xlabel('Time (s)')

        start_time = batch['start_time']


        #xticks
        spec.shape[1]/15

        xticks = np.arange(0, spec.shape[1], int(spec.shape[1]/15))
        xlabels = [int(x) for x in range(int(start_time), int(start_time)+len(xticks))]
        ax.set_xticks(xticks, labels=xlabels)

        #yticks
        # 201 bins, ranging from 0 to 32000/2, we want every 1000th freq
        # interval =1000/(32000/2) *201

        yticks = np.arange(0, spec.shape[0], int(spec.shape[0]/16))
        ylabels = [int(x/1000) for x in range(0, int(32000/2)+1, 1000)]
        ax.set_yticks(yticks, labels=ylabels)
        

        os.makedirs(processed_outputs, exist_ok=True)
        print(os.path.join(processed_outputs, os.path.basename(filenames[item]).replace('.flac',f'_{int(start_time)}.png')))
        plt.savefig(os.path.join(processed_outputs, os.path.basename(filenames[item]).replace('.flac',f'_{int(start_time)}.png')))

        plt.close()
        plt.clf()


        output_name =  os.path.join(processed_outputs, os.path.basename(filenames[item]).replace('.flac',f'_{int(start_time)}.mp3').replace('.wav',f'_{int(start_time)}.mp3'))

        if os.path.exists(output_name):
            continue


        # os.system(f'ffmpeg -ss {int(start_time)} -y -i {filenames[item]} -t 15 -af "volume={gain}dB" {output_name}')
        
        # command works but the library had an error
        filename = filenames[item].replace(' ','\\ ')
        output_name = output_name.replace(' ','\\ ')
        os.system(f'ffmpeg -ss {int(start_time)} -y -i {filename} -t 15 -af loudnorm {output_name}')

        






def power_to_db(S,*,ref = 1.0, amin: float = 1e-10, top_db: float = 80.0,):
    S = np.asarray(S)
    if amin <= 0:
        raise Exception("amin must be strictly positive")
    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = np.abs(S)
    else:
        magnitude = S
    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec




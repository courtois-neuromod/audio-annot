import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
import torchaudio
from torchaudio import functional as F
from scipy import signal

import librosa

def get_fig_indices(Df, indices, time_axe):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("<b>{}</b>".format(indices[0]), "<b>{}</b>".format(indices[1]), "<b>{}</b>".format(indices[2])), vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x = time_axe, y = Df[indices[0]], line = dict(color='black'), opacity=0.5, mode='markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x = time_axe, y = Df[indices[1]], line = dict(color='blue'), opacity=0.5, mode='markers'), row=2, col=1)
    fig.add_trace(go.Scatter(x = time_axe, y = Df[indices[2]], line = dict(color='green'), opacity=0.5, mode='markers'), row=3, col=1)
    
   
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, coloraxis_colorbar=dict(
                        title="<b>Probabilité</b>",
                        #titleside='right',
                        thicknessmode="pixels", thickness=30,
                        lenmode="pixels", len=400,
                        yanchor="bottom", y=0.0,
                        xanchor="right", x=1.1
                        ),yaxis = {'fixedrange': True})
    fig.update_yaxes({'fixedrange': True}, row=2, col=1)
    fig.update_yaxes({'fixedrange': True}, row=3, col=1)
    
    return(fig)



def get_sample_fig(file, path, transpose, mode, save, fminmax = None, cminmax = None, dB = 'Log', shift = 0):

    if cminmax == None : cminmax = (-100, -30)

    x, sr = torchaudio.load(os.path.join(path, file + '.flac'), format='flac')
    x_numpy = x[0,:].numpy()
    N = len(x_numpy)


    #### Save to tmp assets
    #if fminmax is not None:
    #    b, a = signal.butter(2, fminmax ,fs=sr, btype='band')
    #    x_numpy = signal.filtfilt(b, a, x_numpy)


    #### transpose
    if shift != 0:
        x_numpy_shifted = librosa.effects.pitch_shift(x_numpy, sr, n_steps=int(shift), bins_per_octave=1)
    else: x_numpy_shifted = x_numpy
    
    x_shifted = torch.FloatTensor(x_numpy_shifted.copy()).view(1, len(x_numpy_shifted))
    torchaudio.save( save , x_shifted, sr, format='flac')
    x = torch.FloatTensor(x_numpy.copy()).view(1, len(x_numpy))

    

    #### Time frequency representation

    if mode == 'STFT':
        f, t, z = signal.stft(x_numpy, fs = sr, nfft=2048, nperseg=2048)
    elif mode == 'MFCC':
        transform = torchaudio.transforms.MelSpectrogram(sr, n_fft=1024, win_length=512, n_mels=256)
        freq = F.melscale_fbanks(transform.n_fft // 2 + 1, transform.mel_scale.f_min, transform.mel_scale.f_max, transform.mel_scale.n_mels, transform.mel_scale.sample_rate, None, 'htk')
        # calculate mel freq bins
        m_min = F.functional._hz_to_mel(transform.mel_scale.f_min, mel_scale='htk')
        m_max = F.functional._hz_to_mel(transform.mel_scale.f_max, mel_scale='htk')

        m_pts = torch.linspace(m_min, m_max, transform.mel_scale.n_mels + 2)
        f = F.functional._mel_to_hz(m_pts, mel_scale='htk').numpy()
        z = transform(x)
        t = np.linspace(0, int(len(x_numpy)/sr), int(z.size()[-1]))
        z = z.numpy()[0,...]
        


    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("<b>Signal</b>", f"<b>{mode}</b>"), vertical_spacing=0.04)
    fig.add_trace(go.Scattergl(x = np.arange(N)/sr, y = x_numpy, line = dict(color='black'), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Heatmap(x = t, y = f, z = 10*np.log10(np.abs(z+1e-3)**2/np.max(np.abs(z+1e-3)**2)), zmin = cminmax[0], zmax = cminmax[1]), row=2, col=1)

    if dB == 'Log':
        fig.update_layout(xaxis2 = dict(title = 'Time (s)'), yaxis2 = dict(type="log", title = 'Frequency (Hz)'))
    else:
        fig.update_layout(yaxis2 = dict(title = 'Frequency (Hz)'), xaxis2 = dict(title = 'Time (s)'))


    return fig




if __name__ == '__main__':
    from datetime import datetime
    import argparse
    import pandas as pd
    import os


    parser = argparse.ArgumentParser(description='Script to display sound files recorded by Audiomoth ')
    parser.add_argument('--save_path', default='/Users/nicolas/Desktop/EAVT/example/metadata/', type=str, help='Path to save meta data')
    args = parser.parse_args()

    # Load datas
    Df = pd.read_csv(os.path.join(args.save_path, 'indices.csv'))
    indices = Df.columns[3:]
    time_axe = [datetime.strptime(ii,"%Y%m%d_%H%M%S") for ii in Df['datetime']]

    fig = get_fig_indices(Df, ('anthropophony', "geophony", "biophony"), time_axe)
    fig.show()
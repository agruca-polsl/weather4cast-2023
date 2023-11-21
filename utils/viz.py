# Weather4cast 2023 Starter Kit
#
# The files from this repository make up the Weather4cast 2023 Starter Kit.
# 
# It builds on and extends the Weather4cast 2022 Starter Kit, the
# original copyright and GPL license notices for which are included
# below.
#
# In line with the provisions of that license, all changes and
# additional code are also released under the GNU General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.

# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
#
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np



def channels_2_time(w, seq_time_bins, n_channels, height, width):
    """ unroll time and channels """
    w = np.reshape(w, (seq_time_bins, n_channels, height, width)) 
    return w

def compute_stats(x):
    ma, mi, me = x.max(), x.min(), x.mean()
    t = f"x in [{mi:.2f}, {ma:.2f}], mean: {me:.2f}"
    return t

def save_pdf(figs, path):
    
    #pp = PdfPages(f'{path}_{datetime.today().strftime("%Y-%m-%d-%H%M%S")}.pdf')
    pp = PdfPages(f'{path}.pdf')

    for fig in figs:   
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    pp.close()
    
def plot_row(fig, gs, row, x, phase, seq_type, text, vmax=0.01, vmin=0, color='b'):
    for i in range(len(x)):
        ax = fig.add_subplot(gs[row, i])
        t = compute_stats(x[i])
        im = ax.imshow(x[i], vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if(seq_type == "in"):
            plt.title(f'input sequence #{text[i]}: {x[i].shape}\n{t}', size=20, ha="center", color=color)
        elif (phase == "test") and (seq_type == "out_test"):  
            plt.title(f'taget seqence #{text[i]}: {x[i].shape}\n{t}', size=20, ha="center", color=color)
        elif (phase == "test") and (seq_type == "out_pred"):  
            plt.title(f'predicted seqence #{text[i]}: {x[i].shape}\n{t}', size=20, ha="center", color=color)
        else:
            plt.title(f'predicted sequence #{4*(row-1)+i}\n{t}', size=20, ha="center", color=color)
        i += 1
    

def plot_in_target_pred(imgs_in, texts_in,
                        imgs_ta, texts_ta,
                        imgs_pr, texts_pr,
                        phase,
                        vmax=0.01, vmin=0, 
                        title=''):
    """
    plot a grid of mages each with its text
    vmax=0.01 ~1.28 mm/h
    """
    ncols = len(imgs_in)
    if(phase == "test"):
        nrows = 1 + int(2*(len(imgs_ta)/ncols)) # todo: might crash by 1, should be fixed
    else:
        nrows = 1 + int((len(imgs_ta)/ncols)) # todo: might crash by 1, should be fixed
    # print(nrows)
    fig = plt.figure(figsize=(32, 10*nrows))
    gs = GridSpec(nrows=nrows, ncols=ncols)
    gs.update(hspace=0, wspace=0)

    # plot input images - blue
    row = 0
    x = imgs_in
    text = texts_in
    plot_row(fig, gs, row, x, phase, "in", text, vmax=vmax, vmin=vmin)
    
    for j in range(0, len(imgs_ta), ncols):
        if(phase == "test"):
            # ground truth - green 
            row += 1
            x = imgs_ta[j:j+ncols]
            text = texts_ta[j:j+ncols]
            plot_row(fig, gs, row, x, phase, "out_test", text, vmax=vmax, vmin=vmin, color='g')
        
        # prdiction - red
        row += 1
        x = imgs_pr[j:j+ncols]
        text = texts_pr[j:j+ncols]
        plot_row(fig, gs, row, x, phase, "out_pred", text, vmax=vmax, vmin=vmin, color='r')
    
    plt.subplots_adjust(top=0.97)
    plt.suptitle(title, fontweight='bold', size=40)
    return fig
            

def plot_sequence(x, y, y_hat, texts_in, texts_ta, params, phase,
                  time_collapsed=True, n=32,  vmax=0.01, vmin=0, channel=0, title=''):
    """
    plot a grid of mages each with its text
    vmax=0.01 ~1.28 mm/h
    """
    
    # time to channels 
    if time_collapsed:
        #x_im = channels_2_time(x, params['len_seq_in'], params['num_input_variables'], params['spatial_dim'], params['spatial_dim'])
        if(phase == "test"):
            y_im = channels_2_time(y, params['len_seq_predict'], params['out_channels'], params['size_target_center'], params['size_target_center'])
        yhat_im = channels_2_time(y_hat, params['len_seq_predict'], params['out_channels'], params['size_target_center'], params['size_target_center'])
    else:
        y_im = y
        yhat_im = y_hat

    # prepare sequences to be ploted
    imgs_in = x[channel]
    imgs_pr = yhat_im[:n, 0] #predicted
    if(phase == "test"):
        imgs_ta = y_im[:n, 0]   #gorund truth
    else:
        imgs_ta = imgs_pr #dummy

    texts_in = texts_in[:n]
    texts_ta = texts_ta[:n]
    texts_pr = texts_ta
    
    
    fig = plot_in_target_pred(imgs_in, texts_in,
                              imgs_ta, texts_ta,
                              imgs_pr, texts_pr,
                              phase,
                              vmax=vmax, vmin=vmin, 
                              title=title)

    return fig
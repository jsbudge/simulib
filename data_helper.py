import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import moviepy.editor as mpy
import io
from PIL import Image

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'


def plotly_fig2array(fig):
    # convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


class AnimationManager(object):
    zmin = 100
    zmax = 200

    def __init__(self):
        # Initialize the figure we want
        figure = {'data': [], 'layout': {}, 'frames': []}

        # Set the basic menu stuff
        figure['layout']['updatemenus'] = [dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None]),
                     {"args": [[None], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                      "label": "Pause",
                      "method": "animate"}
                     ])]

        # Add a slider
        sliders_dict = {
            'active': 0, 'yanchor': 'top', 'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Time: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 0},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': []
        }

        self._fig = figure
        self._slider = sliders_dict

    def addFrame(self, data, name=None, title=None):
        frame_name = name if name is not None else f'frame_{len(self._fig["frames"])}'
        title = title if title is not None else frame_name
        self._fig['frames'].append(go.Frame(data=data, name=frame_name, traces=list(range(len(data))),
                                               layout={'title': title}))
        self._slider['steps'].append({'args': [
            [frame_name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
                           'transition': {'duration': 0}}
        ],
            'label': frame_name,
            'method': 'animate'})

    def setRanges(self, xrngs, yrngs, zrngs):
        self._fig['layout']['scene'] = dict(
            xaxis=dict(range=xrngs, autorange=False),
            yaxis=dict(range=yrngs, autorange=False),
            zaxis=dict(range=zrngs, autorange=False),
            aspectratio=dict(x=1, y=1, z=1))

    def show(self):
        self._fig['layout']['sliders'] = [self._slider]
        self._fig['data'] = self._fig['frames'][0].data
        fig = go.Figure(data=self._fig['data'], layout=go.Layout(**self._fig['layout']), frames=self._fig['frames'])
        fig.show()
        return True

    def saveAsGIF(self, gif_fnme):
        def make_frame(t):
            return plotly_fig2array(go.Figure(data=self._fig['frames'][int(t)].data))
        anim = mpy.VideoClip(make_frame, duration=len(self._fig['frames']))
        anim.write_gif(gif_fnme, fps=1)
        return True


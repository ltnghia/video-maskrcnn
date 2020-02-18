import visdom
import torch


class VisdomVisualizer(object):

    def __init__(self):
        self.viz = visdom.Visdom()
        self.opts = None
        self.type = None

    def create_vis_plot(self, _xlabel, _ylabel, _title, _legend=None, type='line'):
        assert type in ['line', 'dot']
        self.type = type
        self.opts = dict(xlabel=_xlabel,
                         ylabel=_ylabel,
                         title=_title,
                         legend=_legend,
                         markers=False,
                         )
        if self.type == 'line':
            return self.viz.line(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros(1).cpu(),
                opts=self.opts
            )
        else:
            return self.viz.scatter(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros(1).cpu(),
                opts=self.opts
            )

    def update_vis_plot(self, iteration=0, loss=0, window='', name='', update_type=None, epoch_size=1):
        if self.type == 'line':
            return self.viz.line(
                X=torch.ones(1).cpu() * iteration,
                Y=torch.Tensor([loss]).unsqueeze(0).cpu() / epoch_size,
                win=window,
                name=name,
                update=update_type,
                opts=self.opts
            )
        else:
            return self.viz.scatter(
                X=torch.ones(1).cpu() * iteration,
                Y=torch.Tensor([loss]).unsqueeze(0).cpu() / epoch_size,
                win=window,
                name=name,
                update=update_type,
                opts=self.opts
            )



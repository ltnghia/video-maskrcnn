from .detection_checkpoint import DetectronCheckpointer
from .utils import Converter


REMOVED_KEYS = ['cls_score.bias', 'cls_score.weight',
                'pred_score.bias', 'pred_score.weight',
                'bbox_pred.bias', 'bbox_pred.weight',
                'mask_fcn_logits.bias', 'mask_fcn_logits.weight']


def remove_keys_in_model(d, listofkeys=REMOVED_KEYS):
    r = dict(d)
    for key in listofkeys:
        if key in r.keys() and 'rpn' not in key:
            print('key: {} is removed'.format(key))
            r.pop(key)
    return r


class ConvertedCheckpointer(DetectronCheckpointer):
    def __init__(self, cfg, model, save_dir=""):
        super(ConvertedCheckpointer, self).__init__(cfg, model, save_dir=save_dir)

    def convert(self, f=None, use_gn_fpn=False, use_gn_head_box=False, use_gn_head_mask=False, keys=REMOVED_KEYS):
        checkpoint = super(ConvertedCheckpointer, self)._load_file(f)

        checkpoint['model'] = remove_keys_in_model(checkpoint['model'], keys)
        checkpoint['model'] = Converter.convert_fpn_from_gn_detectron(checkpoint['model'], use_gn=use_gn_fpn)
        checkpoint['model'] = Converter.convert_head_box_from_gn_detectron(checkpoint['model'], use_gn=use_gn_head_box)
        checkpoint['model'] = Converter.convert_head_mask_from_gn_detectron(checkpoint['model'], use_gn=use_gn_head_mask)

        # for k, v in sorted(checkpoint['model'].items()):
        #     print(k)

        self._load_model(checkpoint)

        if 'optimizer' in checkpoint:
            checkpoint.pop('optimizer', None)
        if 'scheduler' in checkpoint:
            checkpoint.pop('scheduler', None)
        if 'iteration' in checkpoint:
            checkpoint.pop('iteration', None)

        return checkpoint




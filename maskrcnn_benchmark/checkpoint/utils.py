class Converter(object):
    def __init__(self):
        super(Converter, self).__init__()

    @staticmethod
    def convert_fpn_from_gn_detectron(d, use_gn=False):
        if not use_gn:
            return d
        d2 = {}
        keys = sorted(list(d.keys()))
        for i, key in enumerate(keys):
            if 'fpn' in key:
                if key.endswith('.weight'):
                    d2[key.replace('weight', '0.weight')] = d[key]
                if key.endswith('gn.bias'):
                    d2[key.replace('gn.bias', '1.bias')] = d[key]
                if key.endswith('gn.s'):
                    d2[key.replace('gn.s', '1.weight')] = d[key]
        d.update(d2)
        return d

    @staticmethod
    def convert_head_box_from_gn_detectron(d, use_gn=False):
        if not use_gn:
            return d
        d2 = {}
        keys = sorted(list(d.keys()))
        for key in keys:
            if 'xconvs' in key:
                k = key
                if k.endswith('.0.weight'):
                    d2[k.replace('.0.weight', '1.0.weight')] = d[key]
                    del d[key]
                elif k.endswith('.1.weight'):
                    d2[k.replace('.1.weight', '1.1.weight')] = d[key]
                    del d[key]
                elif k.endswith('.1.bias'):
                    d2[k.replace('.1.bias', '1.1.bias')] = d[key]
                    del d[key]
                elif k.endswith('.3.weight'):
                    d2[k.replace('.3.weight', '2.0.weight')] = d[key]
                    del d[key]
                elif k.endswith('.4.weight'):
                    d2[k.replace('.4.weight', '2.1.weight')] = d[key]
                    del d[key]
                elif k.endswith('.4.bias'):
                    d2[k.replace('.4.bias', '2.1.bias')] = d[key]
                    del d[key]
                elif k.endswith('.6.weight'):
                    d2[k.replace('.6.weight', '3.0.weight')] = d[key]
                    del d[key]
                elif k.endswith('.7.weight'):
                    d2[k.replace('.7.weight', '3.1.weight')] = d[key]
                    del d[key]
                elif k.endswith('.7.bias'):
                    d2[k.replace('.7.bias', '3.1.bias')] = d[key]
                    del d[key]
                elif k.endswith('.9.weight'):
                    d2[k.replace('.9.weight', '4.0.weight')] = d[key]
                    del d[key]
                elif k.endswith('.10.weight'):
                    d2[k.replace('.10.weight', '4.1.weight')] = d[key]
                    del d[key]
                elif k.endswith('.10.bias'):
                    d2[k.replace('.10.bias', '4.1.bias')] = d[key]
                    del d[key]
            elif 'head.bn' in key and 'rpn' not in key:
                k = key.replace('head.bn', 'xconvs')
                if k.endswith('weight'):
                    d2[k.replace('weight', '1.weight')] = d[key]
                    del d[key]
                elif k.endswith('bias'):
                    d2[k.replace('bias', '1.bias')] = d[key]
                    del d[key]
                elif k.endswith('.s'):
                    d2[k.replace('.s', '.1.weight')] = d[key]
                    del d[key]
            elif 'head.conv' in key and 'rpn' not in key:
                k = key.replace('head.conv', 'xconvs')
                if k.endswith('gn.s'):
                    d2[k.replace('gn.s', '1.weight')] = d[key]
                    del d[key]
                elif k.endswith('gn.bias'):
                    d2[k.replace('gn.bias', '1.bias')] = d[key]
                    del d[key]
                elif k.endswith('weight'):
                    d2[k.replace('weight', '0.weight')] = d[key]
                    del d[key]
                elif k.endswith('bias'):
                    d2[k.replace('bias', '0.bias')] = d[key]
                    del d[key]
            # elif 'fc6' in key or 'fc7' in key:
            #     k = key
            #     if k.endswith('weight'):
            #         d2[k.replace('weight', '0.weight')] = d[key]
            #         del d[key]
            #     elif k.endswith('bias'):
            #         d2[k.replace('bias', '0.bias')] = d[key]
            #         del d[key]
        d.update(d2)
        return d

    @staticmethod
    def convert_head_mask_from_gn_detectron(d, use_gn=False):
        if not use_gn:
            return d
        d2 = {}
        keys = sorted(list(d.keys()))
        for key in keys:
            if '.mask.fcn' in key and 'rpn' not in key:
                k = key.replace('.mask.fcn', 'mask_fcn')
                if k.endswith('weight'):
                    d2[k.replace('weight', '0.weight')] = d[key]
                    del d[key]
                elif k.endswith('gn.bias'):
                    d2[k.replace('gn.bias', '1.bias')] = d[key]
                    del d[key]
                elif k.endswith('gn.s'):
                    d2[k.replace('gn.s', '1.weight')] = d[key]
                    del d[key]
        d.update(d2)
        return d

    @staticmethod
    def copy_weight_from_head_box(d):
        d2 = {}
        keys = sorted(list(d.keys()))

        convert_mask_feature_extractor = True
        convert_keypoint_feature_extractor = True
        convert_prediction_feature_extractor = True
        convert_classification_feature_extractor = True
        convert_segmentation_feature_extractor = True

        for key in keys:
            if 'mask.feature_extractor' in key:
                convert_mask_feature_extractor = False
            if 'keypoint.feature_extractor' in key:
                convert_keypoint_feature_extractor = False
            if 'prediction.feature_extractor' in key:
                convert_prediction_feature_extractor = False
            if 'classification.feature_extractor' in key:
                convert_classification_feature_extractor = False
            if 'segmentation.feature_extractor' in key:
                convert_segmentation_feature_extractor = False

        for key in keys:
            if 'box.feature_extractor' in key:
                if convert_mask_feature_extractor:
                    if "feature_extractor.xconvs" in key:
                        k = key.replace('roi_heads.box.feature_extractor.xconvs', 'roi_heads.mask.feature_extractor.mask_fcn')
                        d2[k] = d[key]
                    if "feature_extractor.fc" in key:
                        k = key.replace('roi_heads.box.feature_extractor.fc', 'roi_heads.mask.feature_extractor.mask_fc')
                        d2[k] = d[key]

                if convert_keypoint_feature_extractor:
                    if "feature_extractor.xconvs" in key:
                        k = key.replace('roi_heads.box.feature_extractor.xconvs', 'roi_heads.keypoint.feature_extractor.kp_fcn')
                        d2[k] = d[key]
                    if "feature_extractor.fc" in key:
                        k = key.replace('roi_heads.box.feature_extractor.fc', 'roi_heads.keypoint.feature_extractor.kp_fc')
                        d2[k] = d[key]

                if convert_prediction_feature_extractor:
                    if "feature_extractor.xconvs" in key:
                        k = key.replace('roi_heads.box.feature_extractor.xconvs', 'roi_heads.prediction.feature_extractor.pred_fcn')
                        d2[k] = d[key]
                    if "feature_extractor.fc" in key:
                        k = key.replace('roi_heads.box.feature_extractor.fc', 'roi_heads.prediction.feature_extractor.pred_fc')
                        d2[k] = d[key]

                if convert_classification_feature_extractor:
                    if "feature_extractor.xconvs" in key:
                        k = key.replace('roi_heads.box.feature_extractor.xconvs', 'decoder_heads.classification.feature_extractor.decoder_cls_fcn')
                        d2[k] = d[key]
                    if "feature_extractor.fc" in key:
                        k = key.replace('roi_heads.box.feature_extractor.fc', 'decoder_heads.classification.feature_extractor.decoder_cls_fc')
                        d2[k] = d[key]

                if convert_segmentation_feature_extractor:
                    if "feature_extractor.xconvs" in key:
                        k = key.replace('roi_heads.box.feature_extractor.xconvs', 'decoder_heads.segmentation.feature_extractor.decoder_seg_fcn')
                        d2[k] = d[key]
                    if "feature_extractor.fc" in key:
                        k = key.replace('roi_heads.box.feature_extractor.fc', 'decoder_heads.segmentation.feature_extractor.decoder_seg_fc')
                        d2[k] = d[key]
        d.update(d2)
        return d

    @staticmethod
    def convert_head_box_from_gn_detectron_old_format(d, use_gn=False):
        if not use_gn:
            return d
        d2 = {}
        keys = sorted(list(d.keys()))
        for key in keys:
            if 'head.bn' in key and 'rpn' not in key:
                k = key.replace('head.bn', 'xconvs.')
                if k.endswith('1.bias'):
                    d2[k.replace('1.bias', '1.bias')] = d[key]
                    del d[key]
                if k.endswith('1.weight'):
                    d2[k.replace('1.weight', '1.weight')] = d[key]
                    del d[key]
                if k.endswith('2.bias'):
                    d2[k.replace('2.bias', '4.bias')] = d[key]
                    del d[key]
                if k.endswith('2.weight'):
                    d2[k.replace('2.weight', '4.weight')] = d[key]
                    del d[key]
                if k.endswith('3.bias'):
                    d2[k.replace('3.bias', '7.bias')] = d[key]
                    del d[key]
                if k.endswith('3.weight'):
                    d2[k.replace('3.weight', '7.weight')] = d[key]
                    del d[key]
                if k.endswith('4.bias'):
                    d2[k.replace('4.bias', '10.bias')] = d[key]
                    del d[key]
                if k.endswith('4.weight'):
                    d2[k.replace('4.weight', '10.weight')] = d[key]
                    del d[key]
            elif 'head.conv' in key and 'rpn' not in key:
                k = key.replace('head.conv', 'xconvs.')
                if k.endswith('1.weight'):
                    d2[k.replace('1.weight', '0.weight')] = d[key]
                    del d[key]
                if k.endswith('1.gn.bias'):
                    d2[k.replace('1.gn.bias', '1.bias')] = d[key]
                    del d[key]
                if k.endswith('1.gn.s'):
                    d2[k.replace('1.gn.s', '1.weight')] = d[key]
                    del d[key]
                if k.endswith('2.weight'):
                    d2[k.replace('2.weight', '3.weight')] = d[key]
                    del d[key]
                if k.endswith('2.gn.bias'):
                    d2[k.replace('2.gn.bias', '4.bias')] = d[key]
                    del d[key]
                if k.endswith('2.gn.s'):
                    d2[k.replace('2.gn.s', '4.weight')] = d[key]
                    del d[key]
                if k.endswith('3.weight'):
                    d2[k.replace('3.weight', '6.weight')] = d[key]
                    del d[key]
                if k.endswith('3.gn.bias'):
                    d2[k.replace('3.gn.bias', '7.bias')] = d[key]
                    del d[key]
                if k.endswith('3.gn.s'):
                    d2[k.replace('3.gn.s', '7.weight')] = d[key]
                    del d[key]
                if k.endswith('4.weight'):
                    d2[k.replace('4.weight', '9.weight')] = d[key]
                    del d[key]
                if k.endswith('4.gn.bias'):
                    d2[k.replace('4.gn.bias', '10.bias')] = d[key]
                    del d[key]
                if k.endswith('4.gn.s'):
                    d2[k.replace('4.gn.s', '10.weight')] = d[key]
                    del d[key]
        d.update(d2)
        return d

    @staticmethod
    def convert_from_version1_to_version2(d):
        d2 = {}
        keys = sorted(list(d.keys()))
        for key in keys:
            if '.attention.' in key and '.regional_attention.attention.' not in key and 'decoder_heads' not in key:
                k = key.replace('.attention.', '.regional_attention.attention.')
                d2[k] = d[key]
                del d[key]
            elif '.attention_' in key and '.regional_attention.attention.' not in key and 'decoder_heads' not in key:
                k = key.replace('.attention_', '.regional_attention.attention.attention_')
                d2[k] = d[key]
                del d[key]
        d.update(d2)
        return d


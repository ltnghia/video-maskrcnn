import numpy as np
import datetime
import time
from collections import defaultdict
from pycocotools import mask as maskUtils
import copy
from pycocotools.cocoeval import COCOeval, Params


class ExtendedCOCOeval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super(ExtendedCOCOeval, self).__init__(cocoGt, cocoDt, iouType)
        self.params = ExtendedParams(iouType=iouType)  # parameters
        if cocoGt:
            self.params.imgIds = sorted(cocoGt.getImgIds())

            self.params.Categories = cocoGt.getCats(category_type='categories')
            self.params.catIds = cocoGt.getCatIds(category_type='categories')
            if self.params.catIds:
                self.params.catIds = sorted(self.params.catIds)
            else:
                self.params.useCats = 0

            self.params.SecondCategories = cocoGt.getCats(category_type='second_categories')
            self.params.secondCatIds = cocoGt.getCatIds(category_type='second_categories')
            if self.params.secondCatIds:
                self.params.secondCatIds = sorted(self.params.secondCatIds)
            else:
                self.params.useSecondCats = 0

            self.params.ThirdCategories = cocoGt.getCats(category_type='third_categories')
            self.params.thirdCatIds = cocoGt.getCatIds(category_type='third_categories')
            if self.params.thirdCatIds:
                self.params.thirdCatIds = sorted(self.params.thirdCatIds)
            else:
                self.params.useThirdCats = 0

    def evaluate(self, category_type='category_id'):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        if p.useSecondCats:
            p.secondCatIds = list(np.unique(p.secondCatIds))
        if p.useThirdCats:
            p.thirdCatIds = list(np.unique(p.thirdCatIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare(category_type)
        # loop through images, area range, max detection number
        if category_type == 'category_id':
            catIds = p.catIds if p.useCats else [-1]
        elif category_type == 'second_category_id':
            catIds = p.secondCatIds if p.useSecondCats else [-1]
        elif category_type == 'third_category_id':
            catIds = p.thirdCatIds if p.useThirdCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId, category_type) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet, category_type)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def _prepare(self, category_type='category_id'):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats and category_type == 'category_id':
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds, category_type=category_type))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds, category_type=category_type))
        elif p.useSecondCats and category_type == 'second_category_id':
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.secondCatIds, category_type=category_type))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.secondCatIds, category_type=category_type))
        elif p.useThirdCats and category_type == 'third_category_id':
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.thirdCatIds, category_type=category_type))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.thirdCatIds, category_type=category_type))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt[category_type]].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt[category_type]].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def computeIoU(self, imgId, catId, category_type='category_id'):
        p = self.params

        if category_type == 'category_id':
            if p.useCats:
                gt = self._gts[imgId, catId]
                dt = self._dts[imgId, catId]
            else:
                gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
                dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        elif category_type == 'second_category_id':
            if p.useSecondCats:
                gt = self._gts[imgId, catId]
                dt = self._dts[imgId, catId]
            else:
                gt = [_ for cId in p.secondCatIds for _ in self._gts[imgId, cId]]
                dt = [_ for cId in p.secondCatIds for _ in self._dts[imgId, cId]]
        elif category_type == 'third_category_id':
            if p.useThirdCats:
                gt = self._gts[imgId, catId]
                dt = self._dts[imgId, catId]
            else:
                gt = [_ for cId in p.thirdCatIds for _ in self._gts[imgId, cId]]
                dt = [_ for cId in p.thirdCatIds for _ in self._dts[imgId, cId]]

        if len(gt) == 0 and len(dt) == 0:
            return []

        if category_type == 'category_id':
            inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        elif category_type == 'second_category_id':
            inds = np.argsort([-d['second_score'] for d in dt], kind='mergesort')
        elif category_type == 'third_category_id':
            inds = np.argsort([-d['third_score'] for d in dt], kind='mergesort')

        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet, category_type='category_id'):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params

        if category_type == 'category_id':
            if p.useCats:
                gt = self._gts[imgId, catId]
                dt = self._dts[imgId, catId]
            else:
                gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
                dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        elif category_type == 'second_category_id':
            if p.useSecondCats:
                gt = self._gts[imgId, catId]
                dt = self._dts[imgId, catId]
            else:
                gt = [_ for cId in p.secondCatIds for _ in self._gts[imgId, cId]]
                dt = [_ for cId in p.secondCatIds for _ in self._dts[imgId, cId]]
        elif category_type == 'third_category_id':
            if p.useThirdCats:
                gt = self._gts[imgId, catId]
                dt = self._dts[imgId, catId]
            else:
                gt = [_ for cId in p.thirdCatIds for _ in self._gts[imgId, cId]]
                dt = [_ for cId in p.thirdCatIds for _ in self._dts[imgId, cId]]

        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]

        if category_type == 'category_id':
            dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        elif category_type == 'second_category_id':
            dtind = np.argsort([-d['second_score'] for d in dt], kind='mergesort')
        elif category_type == 'third_category_id':
            dtind = np.argsort([-d['third_score'] for d in dt], kind='mergesort')

        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p=None, category_type='category_id'):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params

        if category_type == 'category_id':
            p.catIds = p.catIds if p.useCats == 1 else [-1]
            K = len(p.catIds) if p.useCats else 1
        elif category_type == 'second_category_id':
            p.secondCatIds = p.secondCatIds if p.useSecondCats == 1 else [-1]
            K = len(p.secondCatIds) if p.useSecondCats else 1
        elif category_type == 'third_category_id':
            p.thirdCatIds = p.thirdCatIds if p.useThirdCats == 1 else [-1]
            K = len(p.thirdCatIds) if p.useThirdCats else 1

        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        if category_type == 'category_id':
            catIds = _pe.catIds if _pe.useCats else [-1]
        elif category_type == 'second_category_id':
            catIds = _pe.secondCatIds if _pe.useSecondCats else [-1]
        elif category_type == 'third_category_id':
            catIds = _pe.thirdCatIds if _pe.useThirdCats else [-1]

        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        if category_type == 'category_id':
            k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        elif category_type == 'second_category_id':
            k_list = [n for n, k in enumerate(p.secondCatIds)  if k in setK]
        elif category_type == 'third_category_id':
            k_list = [n for n, k in enumerate(p.thirdCatIds)  if k in setK]

        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def summarize(self, category_type='category_id'):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100, category_type='category_id'):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])

                # cacluate AP(average precision) for each category
                num_classes = s.shape[2]
                if ap == 1:
                    print('')
                    cat_name = 'noname'
                    for i in range(0, num_classes):
                        if category_type == 'category_id':
                            cat_id = p.catIds[i]
                            for cat in p.Categories:
                                if cat_id == cat['id']:
                                    cat_name = cat['name']
                                    break
                        elif category_type == 'second_category_id':
                            cat_id = p.secondCatIds[i]
                            for cat in p.SecondCategories:
                                if cat_id == cat['id']:
                                    cat_name = cat['name']
                                    break
                        elif category_type == 'third_category_id':
                            cat_id = p.thirdCatIds[i]
                            for cat in p.ThirdCategories:
                                if cat_id == cat['id']:
                                    cat_name = cat['name']
                                    break

                        print('Category: {:<20} (ID {:3d}) - mAP : {}'.format(cat_name, cat_id, np.mean(s[:, :, i, :])))

            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets(category_type='category_id'):
            stats = np.zeros((12,))
            stats[0] = _summarize(1, category_type=category_type)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], category_type=category_type)
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2], category_type=category_type)
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2], category_type=category_type)
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2], category_type=category_type)
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2], category_type=category_type)
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0], category_type=category_type)
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1], category_type=category_type)
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2], category_type=category_type)
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2], category_type=category_type)
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2], category_type=category_type)
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2], category_type=category_type)
            return stats

        def _summarizeKps(category_type='category_id'):
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20, category_type=category_type)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5, category_type=category_type)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75, category_type=category_type)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium', category_type=category_type)
            stats[4] = _summarize(1, maxDets=20, areaRng='large', category_type=category_type)
            stats[5] = _summarize(0, maxDets=20, category_type=category_type)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5, category_type=category_type)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75, category_type=category_type)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium', category_type=category_type)
            stats[9] = _summarize(0, maxDets=20, areaRng='large', category_type=category_type)
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize(category_type=category_type)

    def __str__(self, category_type='category_id'):
        self.summarize(category_type=category_type)


class ExtendedParams(Params):
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        super(ExtendedParams, self).setDetParams()
        self.secondCatIds = []
        self.useSecondCats = 1
        self.thirdCatIds = []
        self.useThirdCats = 1

    def setKpParams(self):
        super(ExtendedParams, self).setKpParams()
        self.secondCatIds = []
        self.useSecondCats = 1
        self.thirdCatIds = []
        self.useThirdCats = 1

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
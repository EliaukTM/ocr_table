# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.

from .parallel import parallel_process


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''

    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (
                n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def batch_evaluate_html(self, pred_htmls, true_htmls):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
        '''
        if self.n_jobs == 1:
            scores = [self.evaluate(pred_html, true_html) for (
                pred_html, true_html) in zip(pred_htmls, true_htmls)]
        else:
            inputs = [{"pred": pred_html, "true": true_html} for (
                pred_html, true_html) in zip(pred_htmls, true_htmls)]

            scores = parallel_process(
                inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        return scores


if __name__ == '__main__':
    import json
    import pprint

    with open('sample_pred.json') as fp:
        pred_json = json.load(fp)
    with open('sample_gt.json') as fp:
        true_json = json.load(fp)
    teds = TEDS(n_jobs=4)
    scores = teds.batch_evaluate(pred_json, true_json)
    pp = pprint.PrettyPrinter()
    pp.pprint(scores)

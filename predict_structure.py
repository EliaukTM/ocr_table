# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import os

import utility

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import time
import json
from table_logging import get_logger
from utility import parse_args
from ppocr.operators import *
from ppocr.table_ops import *
from postprocess.db_postprocess import *
from postprocess.rec_postprocess import *
from postprocess.table_postprocess import *

logger = get_logger()


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists

def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    return any([path.lower().endswith(e) for e in img_end])

def draw_rectangle(img_path, boxes):
    boxes = np.array(boxes)
    img = cv2.imread(img_path)
    img_show = img.copy()
    for box in boxes.astype(int):
        x1, y1, x2, y2 = box
        cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img_show

def check_and_read(img_path):
    if os.path.basename(img_path)[-3:].lower() == 'gif':
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True, False
    elif os.path.basename(img_path)[-3:].lower() == 'pdf':
        from paddle.utils import try_import
        try_import('fitz')
        from PIL import Image
        imgs = []
        with fitz.open(img_path) as pdf:
            for pg in range(0, pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
            return imgs, False, True
    return None, False, False



def build_pre_process_list(args):
    resize_op = {'ResizeTableImage': {'max_len': args.table_max_len, }}
    pad_op = {
        'PaddingTableImage': {
            'size': [args.table_max_len, args.table_max_len]
        }
    }
    normalize_op = {
        'NormalizeImage': {
            'std': [0.229, 0.224, 0.225] if
            args.table_algorithm not in ['TableMaster'] else [0.5, 0.5, 0.5],
            'mean': [0.485, 0.456, 0.406] if
            args.table_algorithm not in ['TableMaster'] else [0.5, 0.5, 0.5],
            'scale': '1./255.',
            'order': 'hwc'
        }
    }
    to_chw_op = {'ToCHWImage': None}
    keep_keys_op = {'KeepKeys': {'keep_keys': ['image', 'shape']}}
    if args.table_algorithm not in ['TableMaster']:
        pre_process_list = [
            resize_op, normalize_op, pad_op, to_chw_op, keep_keys_op
        ]
    else:
        pre_process_list = [
            resize_op, pad_op, normalize_op, to_chw_op, keep_keys_op
        ]
    return pre_process_list


class TableStructurer(object):
    def __init__(self, args):
        self.args = args
        self.use_onnx = args.use_onnx
        pre_process_list = build_pre_process_list(args)
        if args.table_algorithm not in ['TableMaster']:
            postprocess_params = {
                'name': 'TableLabelDecode',
                "character_dict_path": args.table_char_dict_path,
                'merge_no_span_structure': args.merge_no_span_structure
            }
        else:
            postprocess_params = {
                'name': 'TableMasterLabelDecode',
                "character_dict_path": args.table_char_dict_path,
                'box_shape': 'pad',
                'merge_no_span_structure': args.merge_no_span_structure
            }

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'table', logger)

        if args.benchmark:
            import auto_log
            pid = os.getpid()
            gpu_id = utility.get_infer_gpuid()
            self.autolog = auto_log.AutoLogger(
                model_name="table",
                model_precision=args.precision,
                batch_size=1,
                data_shape="dynamic",
                save_path=None,  #args.save_log_path,
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=gpu_id if args.use_gpu else None,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    def __call__(self, img):
        starttime = time.time()
        if self.args.benchmark:
            self.autolog.times.start()

        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()
        if self.args.benchmark:
            self.autolog.times.stamp()
        if self.use_onnx:
            input_dict = {}
            input_dict[self.input_tensor.name] = img
            outputs = self.predictor.run(self.output_tensors, input_dict)
        else:
            self.input_tensor.copy_from_cpu(img)
            self.predictor.run()
            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
            if self.args.benchmark:
                self.autolog.times.stamp()

        preds = {}
        preds['structure_probs'] = outputs[1]
        preds['loc_preds'] = outputs[0]

        shape_list = np.expand_dims(data[-1], axis=0)
        post_result = self.postprocess_op(preds, [shape_list])

        structure_str_list = post_result['structure_batch_list'][0]
        bbox_list = post_result['bbox_batch_list'][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = [
            '<html>', '<body>', '<table>'
        ] + structure_str_list + ['</table>', '</body>', '</html>']
        elapse = time.time() - starttime
        if self.args.benchmark:
            self.autolog.times.end(stamp=True)
        return (structure_str_list, bbox_list), elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    table_structurer = TableStructurer(args)
    count = 0
    total_time = 0
    os.makedirs(args.output, exist_ok=True)
    with open(
            os.path.join(args.output, 'infer.txt'), mode='w',
            encoding='utf-8') as f_w:
        for image_file in image_file_list:
            img, flag, _ = check_and_read(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                logger.info("error in loading image:{}".format(image_file))
                continue
            structure_res, elapse = table_structurer(img)
            structure_str_list, bbox_list = structure_res
            bbox_list_str = json.dumps(bbox_list.tolist())
            logger.info("result: {}, {}".format(structure_str_list,
                                                bbox_list_str))
            f_w.write("result: {}, {}\n".format(structure_str_list,
                                                bbox_list_str))

            if len(bbox_list) > 0 and len(bbox_list[0]) == 4:
                img = draw_rectangle(image_file, bbox_list)
            else:
                img = utility.draw_boxes(img, bbox_list)
            img_save_path = os.path.join(args.output,
                                         os.path.basename(image_file))
            cv2.imwrite(img_save_path, img)
            logger.info("save vis result to {}".format(img_save_path))
            if count > 0:
                total_time += elapse
            count += 1
            logger.info("Predict time of {}: {}".format(image_file, elapse))
    if args.benchmark:
        table_structurer.autolog.report()

def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        print(op_name + "==="+str(param))
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def build_post_process(config, global_config=None):
    support_dict = [
        'DBPostProcess', 'EASTPostProcess', 'SASTPostProcess', 'FCEPostProcess',
        'CTCLabelDecode', 'AttnLabelDecode', 'ClsPostProcess', 'SRNLabelDecode',
        'PGPostProcess', 'DistillationCTCLabelDecode', 'TableLabelDecode',
        'DistillationDBPostProcess', 'NRTRLabelDecode', 'SARLabelDecode',
        'SEEDLabelDecode', 'VQASerTokenLayoutLMPostProcess',
        'VQAReTokenLayoutLMPostProcess', 'PRENLabelDecode',
        'DistillationSARLabelDecode', 'ViTSTRLabelDecode', 'ABINetLabelDecode',
        'TableMasterLabelDecode', 'SPINLabelDecode',
        'DistillationSerPostProcess', 'DistillationRePostProcess',
        'VLLabelDecode', 'PicoDetPostProcess', 'CTPostProcess',
        'RFLLabelDecode', 'DRRGPostprocess', 'CANLabelDecode',
        'SATRNLabelDecode'
    ]

    if config['name'] == 'PSEPostProcess':
        support_dict.append('PSEPostProcess')

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


if __name__ == "__main__":
    main(parse_args())

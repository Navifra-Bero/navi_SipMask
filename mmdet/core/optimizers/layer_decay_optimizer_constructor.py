# Copyright (c) OpenMMLab. All rights reserved.
import json

from mmcv.runner import DefaultOptimizerConstructor, get_dist_info

from mmdet.utils import get_root_logger
from .builder import OPTIMIZER_BUILDERS


def get_layer_id_for_convnext(var_name, max_layer_id):
    """Get the layer id to set the different learning rates in ``layer_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_layer_id (int): Maximum layer id.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    else:
        return max_layer_id + 1


def get_stage_id_for_convnext(var_name, max_stage_id):
    """Get the stage id to set the different learning rates in ``stage_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_stage_id (int): Maximum stage id.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        return 0
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        return stage_id + 1
    else:
        return max_stage_id - 1


@OPTIMIZER_BUILDERS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimizerConstructor):

    def add_params(self, params, module, **kwargs):
        """Add all parameters of module to the params list.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """
        logger = get_root_logger()
        parameter_groups = {}
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')

        # ConvNeXt 관련 파라미터
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights

            # 파라미터 그룹 이름과 weight_decay 설정
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            # ConvNeXt 전용 설정
            if 'ConvNeXt' in module.backbone.__class__.__name__:
                if 'layer_wise' in decay_type:
                    layer_id = get_layer_id_for_convnext(
                        name, self.paramwise_cfg.get('num_layers'))
                    logger.info(f'set param {name} as id {layer_id}')
                elif decay_type == 'stage_wise':
                    layer_id = get_stage_id_for_convnext(name, num_layers)
                    logger.info(f'set param {name} as id {layer_id}')
                group_name = f'layer_{layer_id}_{group_name}'
                scale = decay_rate**(num_layers - layer_id - 1)
                lr = scale * self.base_lr
            else:
                # ConvNeXt가 아닌 경우 기본 학습률 사용
                layer_id = 0
                group_name = f'{group_name}_default'
                scale = 1.0
                lr = self.base_lr

            # 파라미터 그룹 생성 및 추가
            if group_name not in parameter_groups:
                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)

        # 로깅 정보
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {
                key: {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                } for key in parameter_groups
            }
            logger.info(f'Param groups = {json.dumps(to_display, indent=2)}')

        # 파라미터 그룹 추가
        params.extend(parameter_groups.values())

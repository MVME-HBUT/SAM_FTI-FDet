import copy
import warnings
import einops
import numpy as np
import torch
from mmcv.cnn import build_norm_layer, ConvModule
from mmcv.ops import point_sample
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from peft import get_peft_config, get_peft_model
from torch import nn, Tensor
from transformers import SamConfig
from transformers.models.sam.modeling_sam import SamVisionEncoder, SamMaskDecoder, SamPositionalEmbedding, SamTwoWayTransformer, \
    SamPromptEncoder, SamModel, SamVisionEncoderOutput
from typing import List, T, Tuple, Optional, Dict, Union
from mmdet.models import MaskRCNN, StandardRoIHead, FCNMaskHead, SinePositionalEncoding, Mask2Former, Mask2FormerHead, \
    MaskFormerFusionHead, BaseDetector
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import unpack_gt_instances, empty_instances, multi_apply, \
    get_uncertain_point_coords_with_randomness
from mmdet.registry import MODELS
from mmdet.structures import SampleList, DetDataSample, OptSampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import OptConfigType, MultiConfig, ConfigType, InstanceList, reduce_mean
import torch.nn.functional as F
from typing import List, Tuple, Type
from mmpretrain.models import LayerNorm2d
# from .heatmap import plot_attention_heatmap
# from .featuremap import visualize_feature_map

@MODELS.register_module(force=True)
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

@MODELS.register_module()
class PrompterQuery(Mask2Former):
    def __init__(
            self,
            shared_image_embedding,
            encoder_freeze=False,
            decoder_freeze=True,
            *args,
            **kwargs):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)
        self.decoder_freeze = decoder_freeze
        self.encoder_freeze = encoder_freeze
        self.with_mask2formerhead = False if isinstance(self.panoptic_head, FTIMask2FormerHead) else True
        self.shared_image_embedding = MODELS.build(shared_image_embedding)

        self.frozen_modules = []
        if peft_config is None:
            self.frozen_modules += [self.backbone]
            
        if self.encoder_freeze:
            self.frozen_modules += [self.backbone]
            
        if self.decoder_freeze:
            self.frozen_modules += [
                self.shared_image_embedding,
                self.panoptic_head.mask_decoder,
            ]
        self._set_grad_false(self.frozen_modules)

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

    def get_image_wide_positional_embeddings(self, size):
        target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        vision_outputs = self.backbone(batch_inputs)
        if isinstance(vision_outputs, SamVisionEncoderOutput):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs[1]
        elif isinstance(vision_outputs, tuple):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs[1]
        else:
            raise NotImplementedError

        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1])
        # repeat with batch size
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
        x = self.neck(vision_hidden_states)
        return x, image_embeddings, image_positional_embeddings

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:

        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)

        if self.with_mask2formerhead:
            losses = self.panoptic_head.loss(x, batch_data_samples)
        else:
            losses = self.panoptic_head.loss(x, batch_data_samples,
                                             image_embeddings=image_embeddings,
                                             image_positional_embeddings=image_positional_embeddings)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:

        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)
        if self.with_mask2formerhead:
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(x, batch_data_samples)
        else:
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                x, batch_data_samples,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings
            )

        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results,
            mask_pred_results,
            batch_data_samples,
            rescale=rescale)
        results = self.add_pred_to_datasample(batch_data_samples, results_list)

        return results

@MODELS.register_module()
class FTIMask2FormerHead(Mask2FormerHead, BaseModule):
    def __init__(
            self,
            mask_decoder,
            decoder_plus,
            with_sincos=True,
            per_pointset_point=1,
            multimask_output=False,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_plus = decoder_plus
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        self.mask_decoder = MODELS.build(mask_decoder)

        prompt_encoder = dict(
            type='FTISamPromptEncoder',
            hf_pretrain_name=copy.deepcopy(mask_decoder.get('hf_pretrain_name')),
            init_cfg=copy.deepcopy(mask_decoder.get('init_cfg')),
        )
        prompt_encoder = MODELS.build(prompt_encoder)
        prompt_encoder.init_weights()
        if self.decoder_plus:
            self.sam_mask_embed = prompt_encoder.prompt_encoder.mask_embed
        else:
            self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed
            del self.mask_embed

        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos

        self.feat_channels = kwargs['feat_channels']
        self.out_channels = kwargs['out_channels']
        if with_sincos:
            num_sincos = 2
        else:
            num_sincos = 1
        self.point_emb = nn.Sequential(
            nn.Linear(self.feat_channels, self.feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels // 2, self.feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels // 2, self.out_channels * num_sincos * per_pointset_point)
        )
        del self.cls_embed
        self.cls_embed = nn.Sequential(
            nn.Linear(self.feat_channels, self.feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels, self.num_classes + 1))

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int],
                      image_embeddings=None,
                      image_positional_embeddings=None
                      ) -> Tuple[Tensor]:
        img_bs = image_embeddings.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]

        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (batch_size, num_queries, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (batch_size, num_queries, c)
        point_embedings = self.point_emb(decoder_out)

        point_embedings = einops.rearrange(point_embedings, 'b n_set (n_point c) -> b n_set n_point c', n_point=self.per_pointset_point)
        if self.with_sincos:
            point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]

        # B, N_set, N_point, C => (B, N_set), 1, N_point, C
        sparse_embeddings = einops.rearrange(point_embedings, 'b n_set n_point c -> (b n_set) n_point c')
        sparse_embeddings = sparse_embeddings.unsqueeze(1)

        if self.decoder_plus:
            # shape (num_queries, batch_size, h, w)
            mask_embed = self.mask_embed(decoder_out)
            mask_pred_plus = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)

            input_masks = mask_pred_plus.detach()
            input_masks = einops.repeat(input_masks, 'b n h w -> (b n) c h w', c=1)
            # (bs num_q) c h w
            dense_embeddings = self.sam_mask_embed(input_masks)
        else:
            mask_pred_plus = None
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(img_bs, -1, image_embedding_size[0], image_embedding_size[1])

        image_embeddings = torch.repeat_interleave(image_embeddings, repeats=self.num_queries, dim=0)
        image_positional_embeddings = torch.repeat_interleave(image_positional_embeddings, repeats=self.num_queries, dim=0)
        mask_pred, iou_predictions, mask_dencoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            attention_similarity=self.attention_similarity,
            target_embedding=self.target_embedding,
            output_attentions=self.output_attentions,
        )
        mask_pred = mask_pred.reshape(img_bs, -1, *mask_pred.shape[-2:])
        if not self.decoder_plus:
            h, w = mask_pred.shape[-2:]
            # shape (batch_size, num_queries, h, w)
            attn_mask_pred = mask_pred.reshape(img_bs, -1, h, w)
        else:
            attn_mask_pred = mask_pred_plus
        attn_mask = F.interpolate(attn_mask_pred, attn_mask_target_size, mode='bilinear', align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()
        return cls_pred, mask_pred, attn_mask, mask_pred_plus

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList,
                image_embeddings=None,
                image_positional_embeddings=None
                ) -> Tuple[List[Tensor]]:
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask).to(
                decoder_input.dtype)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        mask_pred_plus_list = []
        attn_mask = None

        cls_pred, mask_pred, attn_mask, mask_pred_plus = self._forward_head(query_feat, mask_features, multi_scale_memorys[0].shape[-2:], image_embeddings, image_positional_embeddings)
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        mask_pred_plus_list.append(mask_pred_plus)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            if attn_mask is not None:
                # if a mask is all True(all background), then set it all False.
                mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
                attn_mask = attn_mask & mask_sum

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask, mask_pred_plus = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                image_embeddings, image_positional_embeddings)

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            mask_pred_plus_list.append(mask_pred_plus)
        return cls_pred_list, mask_pred_list, mask_pred_plus_list

    def loss(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        image_embeddings=None,
        image_positional_embeddings=None
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the panoptic
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)

        # forward
        all_cls_scores, all_mask_preds, all_mask_preds_plus = self(x, batch_data_samples, image_embeddings, image_positional_embeddings)
        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances,
                                                batch_gt_semantic_segs)
        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds, all_mask_preds_plus,
                                   batch_gt_instances, batch_img_metas)
        return losses

    def loss_by_feat(self,
                     all_cls_scores: Tensor,
                     all_mask_preds: Tensor,
                     all_mask_preds_plus,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, losses_mask_plus, losses_dice_plus = multi_apply(
            self._loss_by_feat_single,
            all_cls_scores, all_mask_preds,
            all_mask_preds_plus,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        loss_dict['loss_mask_plus'] = losses_mask_plus[-1]
        loss_dict['loss_dice_plus'] = losses_dice_plus[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_mask_plus_i, loss_dice_plus_i in zip(
            losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_mask_plus[:-1], losses_dice_plus[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_mask_plus'] = loss_mask_plus_i
            loss_dict[f'd{num_dec_layer}.loss_dice_plus'] = loss_dice_plus_i

            num_dec_layer += 1
        return loss_dict

    def _loss_by_feat_single(self,
                             cls_scores: Tensor,
                             mask_preds: Tensor,
                             mask_preds_plus,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        mask_preds_plus_list = [mask_preds_plus[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_plus_list,
                                        batch_gt_instances, batch_img_metas)

        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        mask_preds_plus = mask_preds_plus[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            loss_dice_plus = mask_preds_plus.sum()
            loss_mask_plus = mask_preds_plus.sum()
            return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # points_coords = points_coords.to(mask_preds.dtype)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).to(mask_preds.dtype), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)
        mask_point_preds_plus = point_sample(
            mask_preds_plus.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)
        loss_dice_plus = self.loss_dice(
            mask_point_preds_plus, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)

        mask_point_preds_plus = mask_point_preds_plus.reshape(-1)

        # loss_mask = self.loss_mask(
        #     mask_point_preds,
        #     mask_point_targets,
        #     avg_factor=num_total_masks * self.num_points)
        # to avoid nan in fp16 when num_total_masks * self.num_points
        loss_mask = self.loss_mask(mask_point_preds, mask_point_targets)
        loss_mask_plus = self.loss_mask(mask_point_preds_plus, mask_point_targets)
        return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus

    def predict(self, x: Tuple[Tensor],
                batch_data_samples: SampleList,
                image_embeddings=None,
                image_positional_embeddings=None
                ) -> Tuple[Tensor]:
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        all_cls_scores, all_mask_preds, all_mask_preds_plus = self(
            x, batch_data_samples, image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        mask_pred_plus_results = all_mask_preds_plus[-1]
        # upsample masks
        try:
            img_shape = batch_img_metas[0]['batch_input_shape']
        except:
            img_shape = batch_img_metas[0]['pad_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        return mask_cls_results, mask_pred_results


@MODELS.register_module()
class FTIMaskFormerFusionHead(MaskFormerFusionHead):
    def predict(self,
                mask_cls_results: Tensor,
                mask_pred_results: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs) -> List[dict]:
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        panoptic_on = self.test_cfg.get('panoptic_on', True)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', False)
        assert not semantic_on, 'segmantic segmentation '\
            'results are not supported yet.'
        results = []
        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, batch_img_metas):
            # remove padding
            img_height, img_width = meta['img_shape'][:2]
            ori_img_height, ori_img_width = meta['ori_shape'][:2]
            scale_factor = meta['scale_factor']
            ori_scaled_height = int(ori_img_height * scale_factor[1])
            ori_scaled_width = int(ori_img_width * scale_factor[0])
            mask_pred_result = mask_pred_result[:, :ori_scaled_height, :ori_scaled_width]

            if rescale:
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]

            result = dict()
            if panoptic_on:
                pan_results = self.panoptic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['pan_results'] = pan_results

            if instance_on:
                ins_results = self.instance_postprocess(
                    mask_cls_result, mask_pred_result)
                result['ins_results'] = ins_results

            if semantic_on:
                sem_results = self.semantic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['sem_results'] = sem_results

            results.append(result)

        return results



@MODELS.register_module()
class FTISamPositionalEmbedding(SamPositionalEmbedding, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.shared_image_embedding = SamPositionalEmbedding(sam_config)

    def forward(self, *args, **kwargs):
        return self.shared_image_embedding(*args, **kwargs)


@MODELS.register_module()
class FTISamVisionEncoder(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            peft_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        # vision_encoder = SamVisionEncoder(sam_config)
        vision_encoder_cfg = dict(
            type='mmpretrain.TinyViTSAM',
        )
        vision_encoder = MODELS.build(vision_encoder_cfg)
        #load checkpoint
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[(r'^image_encoder\.', '')])

        if peft_config is not None and isinstance(peft_config, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_config)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)



@MODELS.register_module()
class FTISamPromptEncoder(SamPromptEncoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.prompt_encoder = SamPromptEncoder(sam_config, shared_patch_embedding=None)

    def forward(self, *args, **kwargs):
        return self.prompt_encoder(*args, **kwargs)


@MODELS.register_module()
class FTISamMaskDecoder(SamMaskDecoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.mask_decoder = SamMaskDecoder(sam_config)

    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)



# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


@MODELS.register_module()
class FTIFPN(BaseModule):
    def __init__(
            self,
            feature_aggregator=None,
            feature_spliter=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if feature_aggregator is not None:
            self.feature_aggregator = MODELS.build(feature_aggregator)
        if feature_spliter is not None:
            self.feature_spliter = MODELS.build(feature_spliter)

    def forward(self, inputs):
        
        if hasattr(self, 'feature_aggregator'):
            x = self.feature_aggregator(inputs)
        else:
            x = inputs
        if hasattr(self, 'feature_spliter'):
            x = self.feature_spliter(x)
        else:
            x = (x,)
        return x


@MODELS.register_module()
class PseudoFeatureAggregator(BaseModule):
    def __init__(
            self,
            in_channels,
            hidden_channels=64,
            out_channels=256,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.channel_fusion = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(hidden_channels, eps=1e-6),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(hidden_channels, eps=1e-6),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels, eps=1e-6),
        )

    def forward(self, inputs):
        x = inputs[0]
        x = self.channel_fusion(x)
        return x
    
    
@MODELS.register_module()
class FeatureAggregator(BaseModule):
    in_channels_dict = {
        'tiny': [320] * (2+1),
        'large': [1024] * (24+1),
        'huge': [1280] * (32+1),
    }

    def __init__(
            self,
            in_channels,
            hidden_channels=64,
            out_channels=256,
            select_layers=range(1, 12, 2),
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, str)
        model_arch ='tiny' 
        self.in_channels = self.in_channels_dict[model_arch]
        self.select_layers = select_layers

        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            self.downconvs.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i_layer], hidden_channels, 1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.hidden_convs = nn.ModuleList()
        for _ in self.select_layers:
            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in inputs]
        features = []
        for idx, i_layer in enumerate(self.select_layers):
            features.append(self.downconvs[idx](inputs[i_layer]))

        x = None
        for hidden_state, hidden_conv in zip(features, self.hidden_convs):
            if x is not None:
                hidden_state = x + hidden_state
            residual = hidden_conv(hidden_state)
            x = hidden_state + residual
        x = self.fusion_conv(x)
        return x



@MODELS.register_module()
class SAMSegMaskRCNN(MaskRCNN):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)

        if peft_config is None:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_feat(self, batch_inputs: Tensor) -> Tensor:
        vision_outputs = self.backbone(batch_inputs)
        if isinstance(vision_outputs, SamVisionEncoderOutput):
            image_embeddings = vision_outputs.last_hidden_state
            vision_hidden_states = vision_outputs.hidden_states
        elif isinstance(vision_outputs, tuple):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs[1]
        else:
            raise NotImplementedError
        x = self.neck(vision_hidden_states)
        return x


# @MODELS.register_module()
# class SAMSegMask2Former(Mask2Former):
#     def __init__(
#             self,
#             *args,
#             **kwargs,
#     ):
#         peft_config = kwargs.get('backbone', {}).get('peft_config', {})
#         super().__init__(*args, **kwargs)

#         if peft_config is None:
#             self.backbone.eval()
#             for param in self.backbone.parameters():
#                 param.requires_grad = False

#     def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
#         vision_outputs = self.backbone(batch_inputs)
#         if isinstance(vision_outputs, SamVisionEncoderOutput):
#             image_embeddings = vision_outputs.last_hidden_state
#             vision_hidden_states = vision_outputs.hidden_states
#         elif isinstance(vision_outputs, tuple):
#             image_embeddings = vision_outputs[0]
#             vision_hidden_states = vision_outputs
#         else:
#             raise NotImplementedError

#         x = self.neck(vision_hidden_states)
#         return x


@MODELS.register_module()
class SimpleFPN(BaseModule):
    def __init__(self,
                 backbone_channel: int,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2,
                               self.backbone_channel // 4, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        #self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        #pyt
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))


        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input: Tensor) -> tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        inputs = []
        inputs.append(self.fpn1(input))
        inputs.append(self.fpn2(input))
        inputs.append(self.fpn3(input))
        inputs.append(self.fpn4(input))
        

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)



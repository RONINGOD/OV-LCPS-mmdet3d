import torch
import torch.nn as nn
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmdet.models.losses import accuracy
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmdet.models.losses import CrossEntropyLoss,FocalLoss
from mmengine.structures import InstanceData

from mmcv.ops import SubMConv3d

def get_classification_logits(x, text_classifier, logit_scale):
    # x in shape of [B, *, C]
    # text_classifier in shape of [num_classes, C]
    # logit_scale is a learnable scalar https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L201
    # return: [B, *, num_classes]
    x = F.normalize(x, dim=-1) # [1, 250, 768]
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    pred_logits = logit_scale * x @ text_classifier.T # B, *, N + 1 [128, 13]
    return pred_logits

# Ref: https://github.com/NVlabs/ODISE/blob/e97b06c424c575fec9fc5368dd4b3e050d91abc4/odise/modeling/meta_arch/odise.py#L923
class MaskPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        """
        Args:
            x: [b, V, C]
            mask: [b, N, V]
        """
        if not x.shape[-2] == mask.shape[-1]:
            # reshape mask to x
            mask = F.interpolate(mask, size=[mask.shape[-2],x.shape[-2]], mode='bilinear', align_corners=False)

        with torch.no_grad():
            mask = mask.detach()
            mask = (mask > 0).to(mask.dtype)
            denorm = mask.sum(dim=-1, keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum("bvc,bnv->bnc", x, mask / denorm)
        return mask_pooled_x

class MLP(nn.Module):
    def __init__(self, channels,act_type='GELU',bias=False):
        super().__init__()
        self.mlp = nn.ModuleList()
        for cc in range(len(channels) - 2):
            self.mlp.append(
                nn.Sequential(
                    nn.Linear(
                        channels[cc],
                        channels[cc + 1],
                        bias=bias),
                    build_norm_layer(
                        dict(type='LN'), channels[cc + 1])[1],
                    build_activation_layer(
                        dict(type=act_type))))
        self.mlp.append(
            nn.Linear(channels[-2], channels[-1]))
        
    def forward(self, input):
        for layer in self.mlp:
            input = layer(input)
        return input

@MODELS.register_module()
class _PFCHead(nn.Module):
    """P3Former head for 3D panoptic segmentation."""

    def __init__(self,
                 num_classes,
                 num_queries,
                 embed_dims,
                 thing_class,
                 stuff_class,
                 ignore_index,
                 num_decoder_layers=6,
                 vision_clip_dim=768,
                 pos_dim=3,
                 transformer_decoder_cfg=dict(type='_Transformer_Decoder'),
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 use_pa_seg_loss=True,
                 use_dice_loss=True,
                 use_sem_loss=True,
                 panoptic_use_sigmoid = True,
                 use_lable_weight = True,
                 cal_sem_loss = True,
                 assigner_zero_layer_cfg=None,
                 assigner_cfg=None,
                 sampler_cfg=None,
                 cls_channels=(128, 128, 20),
                 mask_channels=(128, 128, 128, 128, 128),
                 pe_type='mpe',
                 use_pa_seg=True,
                 pa_seg_weight = 0.2,
                 score_thr=0.4,
                 iou_thr=0.8,
                 mask_score_thr=0.5,
                 geometric_ensemble_alpha=0.0,
                 geometric_ensemble_beta=1.0,
                 init_logit_scale = 4.6052,
                 grid_size=[480, 360, 32],
                 point_cloud_range=[]):
        super().__init__()

        self.queries = SubMConv3d(embed_dims, num_queries, indice_key="logit", 
                                    kernel_size=1, stride=1, padding=0, bias=False) # 256,128,1
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mask_score_thr = mask_score_thr
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.use_pa_seg = use_pa_seg
        self.thing_class = thing_class
        self.stuff_class = stuff_class
        self.base_thing_class = None
        self.base_stuff_class = None
        self.novel_stuff_class = None
        self.novel_thing_class = None
        self.num_thing_classes = len(thing_class)
        self.num_stuff_classes = len(stuff_class)
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        self.vision_clip_dim = vision_clip_dim
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        self.use_pa_seg_loss = use_pa_seg_loss
        self.use_dice_loss = use_dice_loss
        self.panoptic_use_sigmoid = panoptic_use_sigmoid
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.mask_pooling = MaskPooling()
        # self.mask_features = nn.Conv1d(in_channels=vision_clip_dim, out_channels=vision_clip_dim, kernel_size=1, stride=1, padding=0)
        vision_clip_proj_channels = (vision_clip_dim,embed_dims,embed_dims,embed_dims,embed_dims,embed_dims)
        self.pe_vision_proj = MLP(vision_clip_proj_channels,act_type='GELU',bias=True)

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

        self.use_sem_loss = use_sem_loss
        self.use_lable_weight = use_lable_weight
        self.cal_sem_loss = cal_sem_loss
        
        if use_sem_loss:
            self.loss_ce = MODELS.build(dict(
                            type='mmdet.CrossEntropyLoss',
                            use_sigmoid=False,
                            class_weight=None,
                            loss_weight=1.0))
            self.loss_lovasz = MODELS.build(dict(type='LovaszLoss',
                                                reduction='none',))
            self.sem_queries = nn.Conv3d(embed_dims, vision_clip_dim, kernel_size=1, stride=1, padding=0, bias=False)

        # build assigner
        if assigner_zero_layer_cfg is not None:
            self.zero_assigner = TASK_UTILS.build(assigner_zero_layer_cfg)
        if assigner_cfg is not None:
            self.assigner = TASK_UTILS.build(assigner_cfg)
        if sampler_cfg is not None:
            self.sampler = TASK_UTILS.build(sampler_cfg)


        # build pe
        self.pe_type = pe_type
        self.point_cloud_range = point_cloud_range
        self.grid_size = grid_size
        if self.pe_type == 'polar' or self.pe_type == 'cart':
            self.position_proj = nn.Linear(pos_dim, embed_dims)
            self.position_norm = build_norm_layer(dict(type='LN'),
                                                  embed_dims)[1]
            self.feat_conv = nn.Sequential(
                nn.Linear(embed_dims, embed_dims, bias=False),
                build_norm_layer(dict(type='LN'), embed_dims)[1],
                build_activation_layer(dict(type='GELU')))
        elif self.pe_type == 'mpe':
            self.polar_proj = nn.Linear(pos_dim, embed_dims)
            self.polar_norm = build_norm_layer(dict(type='LN'), embed_dims)[1]
            self.cart_proj = nn.Linear(pos_dim, embed_dims)
            self.cart_norm = build_norm_layer(dict(type='LN'), embed_dims)[1]

            self.pe_conv = nn.ModuleList()
            self.pe_conv.append(
                nn.Linear(embed_dims, embed_dims, bias=False))
            self.pe_conv.append(
                build_norm_layer(dict(type='LN'), embed_dims)[1])
            self.pe_conv.append(build_activation_layer(dict(type='ReLU', inplace=True),))    
            
        else:
            self.feat_conv = nn.Sequential(
                nn.Linear(embed_dims, embed_dims, bias=False),
                build_norm_layer(dict(type='LN'), embed_dims)[1],
                build_activation_layer(dict(type='GELU')))

        # build transformer decoder
        transformer_decoder_cfg.update(embed_dims=embed_dims)
        self.transformer_decoder = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.transformer_decoder.append(MODELS.build(transformer_decoder_cfg))

        # build pa_seg
        self.fc_cls = nn.ModuleList()
        self.fc_cls.append(None)
        self.fc_mask = nn.ModuleList()
        self.fc_mask.append(MLP(mask_channels))
        if use_pa_seg:
            self.fc_coor_mask = nn.ModuleList()
            self.fc_coor_mask.append(MLP(mask_channels))
            self.pa_seg_weight = pa_seg_weight    
        for _ in range(num_decoder_layers):
            self.fc_cls.append(MLP(cls_channels))
            self.fc_mask.append(MLP(mask_channels))
            if use_pa_seg:
                self.fc_coor_mask.append(MLP(mask_channels))

    def init_inputs(self,
                    features,
                    voxel_coors,
                    batch_size,
                    text_features):

        pe_features, mpe = self.mpe(features, voxel_coors, batch_size)
        queries = self.queries.weight.clone().squeeze(0).squeeze(0).repeat(batch_size,1,1).permute(0,2,1) # [1,1,1,256,128] -> [1,128,256]
        queries = [queries[i] for i in range(queries.shape[0])]

        sem_preds = []
        if self.use_sem_loss:
            sem_queries = self.sem_queries.weight.clone().squeeze(-1).squeeze(-1).repeat(1,1,batch_size).permute(2,0,1) # [768,256,1,1,1] -> [1,768,256]
            for b in range(len(pe_features)):
                # sem_pred = torch.einsum("nc,vc->vn", sem_queries[b], pe_features[b]) # [n,c]*[v,c] -> [v,n] [37660,17]
                sem_que = torch.einsum("nc,vn->vc", sem_queries[b], text_features) # [768,256]*[13,768] -> [v,n] [13,256]
                sem_fea = torch.einsum("nc,vc->vn",sem_queries[b],pe_features[b]) # [768,256] [V,256]-> [v,768]
                sem_pred = get_classification_logits(sem_fea,text_features,self.logit_scale)
                sem_preds.append(sem_pred)
                # stuff_queries = sem_queries[b][self.stuff_class] # [11,256]
                base_stuff_queries = sem_que[self.base_stuff_class] # [6,256]
                queries[b] = torch.cat([queries[b], base_stuff_queries], dim=0) # [134,256]

        return queries, pe_features, mpe, sem_preds

    def mpe(self, features, voxel_coors, batch_size):
        """Encode features with sparse indices."""

        if self.pe_type is not None:
            normed_polar_coors = [
                voxel_coor[:, 1:] / voxel_coor.new_tensor(self.grid_size)[None, :].float()
                for voxel_coor in voxel_coors
            ]

        if self.pe_type == 'cart' or self.pe_type == 'mpe':
            normed_cat_coors = []
            for idx in range(len(normed_polar_coors)):
                normed_polar_coor = normed_polar_coors[idx].clone()
                polar_coor = normed_polar_coor.new_zeros(normed_polar_coor.shape)
                for i in range(3):
                    polar_coor[:, i] = normed_polar_coor[:, i]*(
                                        self.point_cloud_range[i+3] -
                                        self.point_cloud_range[i]) + \
                                        self.point_cloud_range[i]
                x = polar_coor[:, 0] * torch.cos(polar_coor[:, 1])
                y = polar_coor[:, 0] * torch.sin(polar_coor[:, 1])
                cat_coor = torch.stack([x, y, polar_coor[:, 2]], 1)
                normed_cat_coor = cat_coor / (
                    self.point_cloud_range[3] - self.point_cloud_range[0])
                normed_cat_coors.append(normed_cat_coor)

        if self.pe_type == 'polar':
            mpe = []
            for i in range(batch_size):
                pe = self.position_norm(
                    self.position_proj(normed_polar_coors[i].float()))
                features[i] = features[i] + pe
                features[i] = self.feat_conv(features[i])
                if self.use_pa_seg:
                    mpe.append(pe)

        elif self.pe_type == 'cart':
            mpe = []
            for i in range(batch_size):
                pe = self.position_norm(
                    self.position_proj(normed_cat_coors[i].float()))
                features[i] = features[i] + pe
                features[i] = self.feat_conv(features[i])
                if self.use_pa_seg:
                    mpe.append(pe)

        elif self.pe_type == 'mpe':
            mpe = []
            for i in range(batch_size):
                cart_pe = self.cart_norm(
                    self.cart_proj(normed_cat_coors[i].float()))
                polar_pe = self.polar_norm(
                    self.polar_proj(normed_polar_coors[i].float()))
                for pc in self.pe_conv:
                    polar_pe = pc(polar_pe)
                    cart_pe = pc(cart_pe)
                    features[i] = pc(features[i])
                pe = cart_pe + polar_pe
                features[i] = features[i] + pe # 
                if self.pa_seg:
                    mpe.append(pe)

        else:
            for i in range(batch_size):
                features[i] = self.feat_conv(features[i])
            mpe = None

        return features, mpe

    def forward(self,
                features,
                voxel_coors,
                text_features):
        class_preds_buffer = []
        mask_preds_buffer = []
        pos_mask_preds_buffer = []

        batch_size = voxel_coors[:, 0].max().item() + 1
        feature_split = []
        voxel_coor_split = []
        for i in range(batch_size):
            feature_split.append(features[voxel_coors[:, 0] == i])
            voxel_coor_split.append(voxel_coors[voxel_coors[:, 0] == i])

        queries, features, mpe, sem_preds = self.init_inputs(
            feature_split, voxel_coor_split, batch_size,text_features)
        _, mask_preds, pos_mask_preds = self.pa_seg(queries, features, mpe, layer=0)
        class_preds_buffer.append(None)
        mask_preds_buffer.append(mask_preds)
        pos_mask_preds_buffer.append(pos_mask_preds)
        for i in range(self.num_decoder_layers):
            queries = self.transformer_decoder[i](queries, features, mask_preds) # queries [139,256] features [37510, 256] mask_preds [139, 37510]
            class_preds, mask_preds, pos_mask_preds = self.pa_seg(queries, features, mpe, layer=i+1)
            class_preds_buffer.append(class_preds)
            mask_preds_buffer.append(mask_preds)
            pos_mask_preds_buffer.append(pos_mask_preds)
        return class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer, sem_preds

    def loss(self, batch_inputs, batch_data_samples, train_cfg):
        self.thing_class = batch_inputs['thing_class'][0]
        self.stuff_class = batch_inputs['stuff_class'][0]
        self.base_thing_class = batch_inputs['base_thing_class'][0]
        self.base_stuff_class = batch_inputs['base_stuff_class'][0]
        self.novel_stuff_class = torch.tensor(sorted(list(set(self.stuff_class.tolist()) - set(self.base_stuff_class.tolist()))),device =self.stuff_class.device)
        self.novel_thing_class = torch.tensor(sorted(list(set(self.thing_class.tolist()) - set(self.base_thing_class.tolist()))),device =self.thing_class.device)
        text_features = batch_inputs['text_features'][0].float()
        voxel_vision_clip = batch_inputs['features']
        projed_voxel_features = self.pe_vision_proj(voxel_vision_clip) # [V，256]
        class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer, sem_preds = self.forward(projed_voxel_features, batch_inputs['voxels']['voxel_coors'],text_features)
        cls_targets_buffer, mask_targets_buffer, label_weights_buffer = self.bipartite_matching(class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer, batch_data_samples,text_features)
        losses = dict()
        for i in range(self.num_decoder_layers+1):
            losses.update(self.loss_single_layer(class_preds_buffer[i], mask_preds_buffer[i], pos_mask_preds_buffer[i],
                                                cls_targets_buffer[i], mask_targets_buffer[i], label_weights_buffer[i], i, 
                                                text_features,batch_data_samples))
        if self.use_sem_loss and self.cal_sem_loss:
            gt_semantic_segs = [
                data_sample.gt_pts_seg.voxel_semantic_mask
                for data_sample in batch_data_samples
            ]
            for b in range(len(sem_preds)):
                # get seen part
                grid_mask = batch_data_samples[b].gt_pts_seg.grid_mask
                sem_preds[b] = sem_preds[b][grid_mask]
            seg_label = torch.cat(gt_semantic_segs) # [46838]
            sem_preds = torch.cat(sem_preds, dim=0) # [46838, 20]
            losses['loss_ce'] = self.loss_ce(
                sem_preds, seg_label, ignore_index=self.ignore_index)
            losses['loss_lovasz'] = self.loss_lovasz(
                sem_preds, seg_label, ignore_index=self.ignore_index)
        return losses
    
    def bipartite_matching(self, class_preds, mask_preds, pos_mask_preds, batch_data_samples,text_features):
        gt_classes, gt_masks = self.generate_mask_class_target(batch_data_samples) # [19] [19,41589]

        gt_thing_classes = []
        gt_thing_masks = []
        gt_stuff_classes = []
        gt_stuff_masks = []

        cls_targets_buffer = []
        mask_targets_buffer = []
        label_weights_buffer = []

        for b in range(len(gt_classes)):
            is_thing_class = (torch.isin(gt_classes[b],self.thing_class)) & (gt_classes[b]!=self.ignore_index)
            is_stuff_class = (torch.isin(gt_classes[b],self.stuff_class)) & (gt_classes[b]!=self.ignore_index)
            gt_thing_classes.append(gt_classes[b][is_thing_class])
            gt_thing_masks.append(gt_masks[b][is_thing_class])
            gt_stuff_classes.append(gt_classes[b][is_stuff_class])
            gt_stuff_masks.append(gt_masks[b][is_stuff_class])

        sampling_results = []
        for b in range(len(mask_preds[0])):
            thing_masks_pred_detach = mask_preds[0][b][:self.num_queries,:].detach() # [128, 13948]
            # get seen part grid_mask错了
            grid_mask = batch_data_samples[b].gt_pts_seg.grid_mask
            thing_masks_pred_detach = thing_masks_pred_detach.permute(1,0)[grid_mask].permute(1,0) # [128, 13555]
            sampled_gt_instances = InstanceData(
                labels=gt_thing_classes[b], masks=gt_thing_masks[b]) # [27], [27, 9493]
            sampled_pred_instances = InstanceData(masks=thing_masks_pred_detach)

            assign_result = self.zero_assigner.assign(
                sampled_pred_instances,
                sampled_gt_instances,
                img_meta=None,)
            sampling_result = self.sampler.sample(assign_result,
                                                    sampled_pred_instances,
                                                    sampled_gt_instances)
            sampling_results.append(sampling_result)

        cls_targets, mask_targets, label_weights, _ = self.get_targets(sampling_results, gt_stuff_masks, gt_stuff_classes)
        cls_targets_buffer.append(cls_targets)
        mask_targets_buffer.append(mask_targets)
        label_weights_buffer.append(label_weights)

        for layer in range(self.num_decoder_layers):
            sampling_results = []
            for b in range(len(mask_preds[0])):
                if class_preds[layer] is not None:
                    thing_class_pred_detach = class_preds[layer][b][:self.num_queries,:].detach()
                else:
                    # for layer 1, we don't have class_preds from layer 0, so we use class_preds from layer 1 for matching
                    thing_class_pred_detach = class_preds[layer+1][b][:self.num_queries,:].detach()
                    
                thing_class_pred_detach = get_classification_logits(thing_class_pred_detach,text_features,self.logit_scale)
                thing_masks_pred_detach = thing_masks_pred_detach = mask_preds[layer][b][:self.num_queries,:].detach()
                
                # get seen part
                grid_mask = batch_data_samples[b].gt_pts_seg.grid_mask
                thing_masks_pred_detach = thing_masks_pred_detach.permute(1,0)[grid_mask].permute(1,0)
                sampled_gt_instances = InstanceData(
                    labels=gt_thing_classes[b], masks=gt_thing_masks[b])
                sampled_pred_instances = InstanceData(
                    scores=thing_class_pred_detach, masks=thing_masks_pred_detach)
                assign_result = self.assigner.assign(
                    sampled_pred_instances,
                    sampled_gt_instances,
                    img_meta=None)
                sampling_result = self.sampler.sample(assign_result,
                                                      sampled_pred_instances,
                                                      sampled_gt_instances)
                sampling_results.append(sampling_result)

            cls_targets, mask_targets, label_weights, _ = self.get_targets(sampling_results, gt_stuff_masks, gt_stuff_classes)
            cls_targets_buffer.append(cls_targets)
            mask_targets_buffer.append(mask_targets)
            label_weights_buffer.append(label_weights)

        return cls_targets_buffer, mask_targets_buffer, label_weights_buffer

    def loss_single_layer(self, class_preds, mask_preds, pos_mask_preds, class_targets, mask_targets, label_weights, layer,
                          text_features,batch_data_samples, reduction_override=None):
        batch_size = len(mask_preds)
        losses = dict()

        class_targets = torch.cat(class_targets, 0)
        pos_inds = (class_targets != self.ignore_index) & (
            class_targets < self.num_classes)
        bool_pos_inds = pos_inds.type(torch.bool)
        bool_pos_inds_split = bool_pos_inds.reshape(batch_size, -1)

        if class_preds is not None:
            class_preds = [get_classification_logits(preds,text_features,self.logit_scale) for preds in class_preds]
            class_preds = torch.cat(class_preds, 0)  # [B*N]
            label_weights = torch.cat(label_weights, 0)  # [B*N]
            num_pos = pos_inds.sum().float()
            avg_factor = reduce_mean(num_pos)

            losses[f'loss_cls_{layer}'] = self.loss_cls(
                class_preds,
                class_targets,
                label_weights if self.use_lable_weight else None, 
                avg_factor=avg_factor,
                reduction_override=reduction_override)

        # mask loss
        loss_mask = 0
        valid_bs = 0
        for mask_idx, (mpred, mtarget) in enumerate(
                zip(mask_preds, mask_targets)):
            mp = mpred[bool_pos_inds_split[mask_idx]]
            mt = mtarget[bool_pos_inds_split[mask_idx]]
            grid_mask = batch_data_samples[mask_idx].gt_pts_seg.grid_mask
            mp = mp.permute(1,0)[grid_mask].permute(1,0)
            if len(mp) > 0:
                valid_bs += 1
                loss_mask += self.loss_mask(
                    mp.reshape(-1, 1), (1 - mt).long().reshape(
                        -1))  # (1 - mt) for binary focal loss  [2529252, 1] [2529252]
        if valid_bs > 0:
            losses[f'loss_mask_{layer}'] = loss_mask / valid_bs
        else:
            losses[f'loss_mask_{layer}'] = class_preds.sum() * 0.0

        loss_dice = 0
        valid_bs = 0
        if self.use_dice_loss:
            for mask_idx, (mpred, mtarget) in enumerate(
                    zip(mask_preds, mask_targets)):
                mp = mpred[bool_pos_inds_split[mask_idx]]
                mt = mtarget[bool_pos_inds_split[mask_idx]]
                grid_mask = batch_data_samples[mask_idx].gt_pts_seg.grid_mask
                mp = mp.permute(1,0)[grid_mask].permute(1,0)
                if len(mp) > 0:
                    valid_bs += 1
                    loss_dice += self.loss_dice(mp, mt) # [54, 46838] [54, 46838]

            if valid_bs > 0:
                losses[f'loss_dice_{layer}'] = loss_dice / valid_bs
            else:
                losses[f'loss_dice_{layer}'] = class_preds.sum() * 0.0

        if self.use_pa_seg and self.use_pa_seg_loss:
            loss_dice_pos = 0
            valid_bs = 0
            for mask_idx, (mpred, mtarget) in enumerate(
                    zip(pos_mask_preds, mask_targets)):
                mp = mpred[bool_pos_inds_split[mask_idx]]
                mt = mtarget[bool_pos_inds_split[mask_idx]]
                grid_mask = batch_data_samples[mask_idx].gt_pts_seg.grid_mask
                mp = mp.permute(1,0)[grid_mask].permute(1,0)
                if len(mp) > 0:
                    valid_bs += 1
                    loss_dice_pos += self.loss_dice(mp, mt) * self.pa_seg_weight

            if valid_bs > 0:
                losses[f'loss_dice_pos_{layer}'] = loss_dice_pos / valid_bs
            else:
                losses[f'loss_dice_pos_{layer}'] = class_preds.sum() * 0.0

        return losses

    def get_targets(
        self,
        sampling_results,
        gt_sem_masks=None,
        gt_sem_classes=None,
        positive_weight=1.0,
    ):
        if gt_sem_masks is None:
            gt_sem_masks = [None] * len(sampling_results)
            gt_sem_classes = [None] * len(sampling_results)

        pos_inds = [sr.pos_inds for sr in sampling_results]
        neg_inds = [sr.neg_inds for sr in sampling_results]
        pos_gt_masks = [sr.pos_gt_masks for sr in sampling_results]
        if hasattr(sampling_results[0], 'pos_gt_labels'):
            pos_gt_labels = [sr.pos_gt_labels for sr in sampling_results]
        else:
            pos_gt_labels = [None] * len(sampling_results)

        (labels, mask_targets, label_weights, mask_weights) = multi_apply(
            self._get_target_single,
            pos_inds,
            neg_inds,
            pos_gt_masks,
            pos_gt_labels,
            gt_sem_masks,
            gt_sem_classes,
            positive_weight=positive_weight)

        return (labels, mask_targets, label_weights, mask_weights)

    def _get_target_single(
        self,
        positive_indices,
        negative_indices,
        positive_gt_masks,
        positive_gt_labels,
        gt_sem_masks,
        gt_sem_classes,
        positive_weight,
    ):
        num_pos = positive_indices.shape[0]
        num_neg = negative_indices.shape[0]
        num_samples = num_pos + num_neg
        num_points = positive_gt_masks.shape[-1]
        labels = positive_gt_masks.new_full((num_samples, ),
                                            self.ignore_index,
                                            dtype=torch.long)
        label_weights = positive_gt_masks.new_zeros(num_samples,
                                                    self.num_classes)
        mask_targets = positive_gt_masks.new_zeros(num_samples, num_points)
        mask_weights = positive_gt_masks.new_zeros(num_samples, num_points)

        if num_pos > 0:
            positive_weight = 1.0 if positive_weight <= 0 else positive_weight

            if positive_gt_labels is not None:
                labels[positive_indices] = positive_gt_labels
            label_weights[positive_indices] = positive_weight
            mask_targets[positive_indices, ...] = positive_gt_masks
            mask_weights[positive_indices, ...] = positive_weight

        if num_neg > 0:
            label_weights[negative_indices] = 1.0

        if gt_sem_masks is not None and gt_sem_classes is not None:
            sem_labels = positive_gt_masks.new_full((len(self.stuff_class), ),
                                                    self.num_classes,
                                                    dtype=torch.long)
            sem_targets = positive_gt_masks.new_zeros(len(self.stuff_class),
                                                      num_points)
            sem_weights = positive_gt_masks.new_zeros(len(self.stuff_class),
                                                      num_points)
            sem_stuff_weights = torch.eye(
                len(self.stuff_class), device=positive_gt_masks.device)
            sem_label_weights = label_weights.new_zeros(len(self.stuff_class),self.num_classes).float()
            sem_label_weights[:, self.stuff_class] = sem_stuff_weights

            if len(gt_sem_classes > 0):
                sem_inds = gt_sem_classes - self.stuff_class[0]
                sem_inds = sem_inds.long()
                sem_labels[sem_inds] = gt_sem_classes.long()
                sem_targets[sem_inds] = gt_sem_masks
                sem_weights[sem_inds] = 1

            label_weights[:, self.stuff_class] = 0
            label_weights[:, self.ignore_index] = 0
            labels = torch.cat([labels, sem_labels])
            label_weights = torch.cat([label_weights, sem_label_weights])
            mask_targets = torch.cat([mask_targets, sem_targets])
            mask_weights = torch.cat([mask_weights, sem_weights])

        target_dict_assign = dict()
        target_dict_assign['labels'] = labels
        target_dict_assign['masks'] = mask_targets

        weight_dict_assign = dict()
        weight_dict_assign['labels'] = label_weights
        weight_dict_assign['masks'] = mask_weights

        return labels, mask_targets, label_weights, mask_weights

    def predict(self, batch_inputs, batch_data_samples):
        self.thing_class = batch_inputs['thing_class'][0]
        self.stuff_class = batch_inputs['stuff_class'][0]
        self.base_thing_class = batch_inputs['base_thing_class'][0]
        self.base_stuff_class = batch_inputs['base_stuff_class'][0]
        self.novel_stuff_class = torch.tensor(sorted(list(set(self.stuff_class.tolist()) - set(self.base_stuff_class.tolist()))),device =self.stuff_class.device)
        self.novel_thing_class = torch.tensor(sorted(list(set(self.thing_class.tolist()) - set(self.base_thing_class.tolist()))),device =self.thing_class.device)
        text_features = batch_inputs['text_features'][0].float()
        voxel_vision_clip = batch_inputs['features']
        projed_voxel_features = self.pe_vision_proj(voxel_vision_clip) # [V，256]
        class_preds_buffer, mask_preds_buffer, _, _ = self.forward(projed_voxel_features, batch_inputs['voxels']['voxel_coors'],text_features)
        voxel_coors = batch_inputs['voxels']['voxel_coors']
        batch_size = voxel_coors[:,0].max().item()+1
        mask_cls_results = class_preds_buffer[-1] # [134, 768]
        mask_pred_results = mask_preds_buffer[-1] # [134, 9813]
        alpha = self.geometric_ensemble_alpha
        beta = self.geometric_ensemble_beta
        category_overlapping_mask = batch_inputs['category_overlapping_mask'][0] 
        class_results_buffer = []
        
        for b in range(batch_size):
            clip_feature = voxel_vision_clip[voxel_coors[:, 0] == b].unsqueeze(0)
            mask_cls = mask_cls_results[b]
            mask_for_pooling = mask_pred_results[b].unsqueeze(0)
            pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling).squeeze(0)
            in_vocabulary_class_preds = get_classification_logits(mask_cls, text_features, self.logit_scale) # [134, 17]
            out_vocabulary_class_preds = get_classification_logits(pooled_clip_feature,text_features,self.logit_scale)
            # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
            in_vocabulary_class_preds = in_vocabulary_class_preds.softmax(-1)
            out_vocabulary_class_preds = out_vocabulary_class_preds.softmax(-1)
            cls_logits_seen = (
                (in_vocabulary_class_preds ** (1 - alpha) * out_vocabulary_class_preds**alpha)
                * category_overlapping_mask
            )
            cls_logits_unseen = (
                (in_vocabulary_class_preds ** (1 - beta) * out_vocabulary_class_preds**beta)
                * (~ category_overlapping_mask)
            ) 
            cls_results = cls_logits_seen + cls_logits_unseen
            class_results_buffer.append(cls_results)
            
        semantic_preds, instance_ids = self.generate_panoptic_results(class_results_buffer, mask_pred_results)
        semantic_preds = torch.cat(semantic_preds)
        instance_ids = torch.cat(instance_ids)
        pts_semantic_preds = []
        pts_instance_preds = []
        
        base_novel_mapping = batch_inputs['base_novel_mapping'][0]
        for batch_idx in range(len(batch_data_samples)):
            semantic_sample = semantic_preds[voxel_coors[:, 0] == batch_idx]
            instance_sample = instance_ids[voxel_coors[:, 0] == batch_idx]
            point2voxel_map = batch_data_samples[
                batch_idx].gt_pts_seg.point2voxel_map.long()
            point_semantic_sample = semantic_sample[point2voxel_map]
            point_instance_sample = instance_sample[point2voxel_map]
            # base novel map
            point_semantic_sample = base_novel_mapping[point_semantic_sample]
            pts_semantic_preds.append(point_semantic_sample.cpu().numpy())
            pts_instance_preds.append(point_instance_sample.cpu().numpy())

        return pts_semantic_preds, pts_instance_preds

    def pa_seg(self, queries, features, mpe, layer):
        if mpe is None:
            mpe = [None] * len(features)
        class_preds, mask_preds, pos_mask_preds = multi_apply(
            self.pa_seg_single, queries, features, mpe, [layer] * len(features))
        return class_preds, mask_preds, pos_mask_preds

    def pa_seg_single(self, queries, features, mpe, layer):
        """Get Predictions of a single sample level."""
        mask_queries = queries # [139,256]
        mask_queries = self.fc_mask[layer](mask_queries) # [139,256]
        mask_pred = torch.einsum('nc,vc->nv', mask_queries, features)  # [139, 37660]

        if self.use_pa_seg:
            pos_mask_queries = queries
            pos_mask_queries = self.fc_coor_mask[layer](pos_mask_queries)
            pos_mask_pred = torch.einsum('nc,vc->nv', pos_mask_queries, mpe) # [139,37660]
            mask_pred = mask_pred + pos_mask_pred
        else:
            pos_mask_pred = None

        if layer != 0:
            cls_queries = queries
            cls_pred = self.fc_cls[layer](cls_queries) # [133,768]
        else:
            cls_pred = None

        return cls_pred, mask_pred, pos_mask_pred

    def generate_mask_class_target(self, batch_data_samples):
        labels = []
        masks = []

        for idx in range(len(batch_data_samples)):
            semantic_label = batch_data_samples[idx].gt_pts_seg.voxel_semantic_mask # [46680]
            instance_label = batch_data_samples[idx].gt_pts_seg.voxel_instance_mask # [46680]

            gt_panoptici_label = (instance_label << 16) + semantic_label
            unique_semantic_label = torch.unique(semantic_label)
            unique_panoptic_label = torch.unique(gt_panoptici_label)

            mask = []
            label = []

            for unq_pan in unique_panoptic_label:
                unq_sem = unq_pan & 0xFFFF
                if unq_sem in self.thing_class:
                    label.append(unq_sem)
                    mask.append(gt_panoptici_label == unq_pan)

            for unq_sem in unique_semantic_label:
                if (unq_sem in self.thing_class) or (unq_sem
                                                     == self.ignore_index):
                    continue
                label.append(unq_sem)
                mask.append(semantic_label == unq_sem)

            if len(label) > 0:
                label = torch.stack(label, dim=0)
                mask = torch.stack(mask, dim=0)
            else:
                label = semantic_label.new_zeros(size=[0])
                mask = semantic_label.new_zeros(
                    size=[0, semantic_label.shape[-1]])

            label, mask = label.long(), mask.long()
            labels.append(label)
            masks.append(mask)

        return (labels, masks)

    def generate_panoptic_results(self, class_preds, mask_preds):
        """Get panoptic results from mask predictions and corresponding class
        predictions.

        Args:
            class_preds (list[torch.Tensor]): Class predictions.
            mask_preds (list[torch.Tensor]): Mask predictions.

        Returns:
            tuple[list[torch.Tensor]]: Semantic predictions and
                instance predictions.
        """
        semantic_preds = []
        instance_ids = []
        for i in range(len(class_preds)):
            class_pred = class_preds[i]
            mask_pred = mask_preds[i]
            thing_novel_stuff_class = torch.cat([self.thing_class,self.novel_stuff_class])
            thing_novel_stuff_scores = class_pred[:self.num_queries][:, thing_novel_stuff_class]
            if self.panoptic_use_sigmoid:
                thing_novel_stuff_scores, thing_novel_stuff_labels = thing_novel_stuff_scores.sigmoid().max(dim=1)
            else:
                thing_novel_stuff_scores, thing_novel_stuff_labels = thing_novel_stuff_scores.max(dim=1)
            thing_novel_stuff_labels += self.thing_class[0]
            base_thing_mask = torch.isin(thing_novel_stuff_labels,self.base_thing_class)
            novel_thing_mask = torch.isin(thing_novel_stuff_labels,self.novel_thing_class-(self.novel_thing_class[0]-self.base_thing_class[-1])+1)
            thing_mask = torch.logical_or(base_thing_mask,novel_thing_mask)
            novel_stuff_mask = torch.isin(thing_novel_stuff_labels,self.novel_stuff_class-(self.novel_thing_class[0]-self.base_thing_class[-1])+1)
            if isinstance(self.loss_cls,FocalLoss):
                thing_novel_stuff_scores[thing_mask] *= 2
            elif isinstance(self.loss_cls,CrossEntropyLoss):
                thing_novel_stuff_scores[thing_mask] *= 2
            thing_novel_stuff_labels[novel_thing_mask]+=(self.novel_thing_class[0]-self.base_thing_class[-1]-1)
            thing_novel_stuff_labels[novel_stuff_mask]+=(self.novel_thing_class[0]-self.base_thing_class[-1]-1)
            if self.panoptic_use_sigmoid:
                base_stuff_scores = class_pred[self.num_queries:][:, self.base_stuff_class].diag().sigmoid()
            else:
                base_stuff_scores = class_pred[self.num_queries:][:, self.base_stuff_class].diag()
            base_stuff_labels = self.base_stuff_class


            scores = torch.cat([thing_novel_stuff_scores, base_stuff_scores], dim=0)
            labels = torch.cat([thing_novel_stuff_labels, base_stuff_labels], dim=0)

            keep = ((scores > self.score_thr) & (labels != self.ignore_index))
            cur_scores = scores[keep]  # [pos_proposal_num]

            cur_classes = labels[keep]  # [pos_proposal_num]
            cur_masks = mask_pred[keep]  # [pos_proposal_num, pt_num]
            cur_masks = cur_masks.sigmoid()

            semantic_pred = cur_classes.new_full((cur_masks.shape[-1], ),
                                                 self.ignore_index)
            instance_id = cur_classes.new_full((cur_masks.shape[-1], ),
                                               0)

            if cur_masks.shape[0] == 0:
                semantic_preds.append(semantic_pred)
                instance_ids.append(instance_id)
                continue

            cur_prob_masks = cur_masks * cur_scores.reshape(-1, 1)
            cur_mask_ids = cur_prob_masks.argmax(0)
            id = 1

            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class in self.thing_class
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                if mask_area > 0 and original_area > 0: 
                    if mask_area / original_area < self.iou_thr:
                        continue
                    semantic_pred[mask] = pred_class
                    if isthing:
                        instance_id[mask] = id
                        id += 1
            semantic_preds.append(semantic_pred)
            instance_ids.append(instance_id)
        return (semantic_preds, instance_ids)
s
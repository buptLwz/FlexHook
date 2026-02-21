import CLIP_my as clip
from .swin_transformer_rope import *
from .vit_rope import *
from .mymodel import MyModel
from .mymodel_src import MyModel_noLRE
import torchvision
from transformers import RobertaModel,BertModel


def build_model(config, is_pretrain=False):
    model_type = config.VISUAL
    text_type = config.TEXT


    if model_type == 'rope-swin-tiny':
        model = swin_rope_mixed_tiny_patch4_window7_224(pretrained=True,img_size=config.DATA.IMG_SIZE)
        layer_dim_list=None
        last_patch_h=2

    elif model_type == 'swin-tiny':
        model = swin_tiny_patch4_window7_224(pretrained=True,img_size=config.DATA.IMG_SIZE)
        layer_dim_list=None
        last_patch_h=2

    elif model_type == 'rope-swin-small':
        model = swin_rope_mixed_small_patch4_window7_224(pretrained=True,img_size=config.DATA.IMG_SIZE)
        layer_dim_list=None
        last_patch_h=2

    elif model_type == 'rope-swin-base':
        model = swin_rope_mixed_base_patch4_window7_224(pretrained=True,img_size=config.DATA.IMG_SIZE)
        layer_dim_list=None
        last_patch_h=2

    elif model_type=='rope-deit':
        model = rope_mixed_deit_small_patch16_LS(pretrained=True,img_size=config.DATA.IMG_SIZE)
        layer_dim_list=[384]
        last_patch_h=4

    elif model_type=='resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        #print(m)
        model = torchvision.models._utils.IntermediateLayerGetter(model,
                {'layer1': '0','layer2': '1','layer3': '2', 'layer4': '3'})  
        model.fc=None
        model.avgpool=None
        layer_dim_list=[256,512,1024,2048]
        last_patch_h=2

    elif model_type=='resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        #print(m)
        model = torchvision.models._utils.IntermediateLayerGetter(model,
                {'layer1': '0','layer2': '1','layer3': '2', 'layer4': '3'})  
        model.fc=None
        model.avgpool=None
        layer_dim_list=[64,128,256,512]
        last_patch_h=2

    elif model_type=='resnet34':
        model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        #print(m)
        model = torchvision.models._utils.IntermediateLayerGetter(model,
                {'layer1': '0','layer2': '1','layer3': '2', 'layer4': '3'})  
        model.fc=None
        model.avgpool=None
        layer_dim_list=[64,128,256,512]
        last_patch_h=2
        

    elif model_type=='clip':
        model_clip,_ = clip.load(config.CLIP_PATH)
        #print(m)
        model = model_clip.visual.float()
        model.attnpool =None 
        layer_dim_list=[256,512,1024,2048]
        last_patch_h=2

    if text_type=='roberta':
        text_encoder=RobertaModel.from_pretrained(config.ROBERTA_PATH)
        text_encoder.pooler=None
        text_dim=768

    elif text_type=='bert':
        text_encoder=BertModel.from_pretrained(config.BERT_PATH)
        text_encoder.pooler=None
        text_dim=768

    elif text_type=='clip':
        if model_type=='clip':
            text_encoder=model_clip
        else:
            text_encoder=clip.load(config.CLIP_PATH)
            
        text_encoder.visual=None
        text_encoder.logit_scale =None
        text_encoder.float()
        text_dim=1024

    if config.LRE == 0:
        func = MyModel_noLRE
    else:
        func = MyModel
    mymodel = func(backbone=model,text_encoder=text_encoder,sample_expression_num=config.sample_expression_num,layer_dim_list=layer_dim_list,last_patch_h=last_patch_h,text_len=config.text_len,text_dim=text_dim,cfg=config)

    return mymodel

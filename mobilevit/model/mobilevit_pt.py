from transformers import MobileViTForImageClassification, MobileViTForSemanticSegmentation, TFMobileViTForSemanticSegmentation

import pdb

def get_mobilevit_pt(model_name: str, task: str, name: str = 'hf', save: bool = False):
    """
    Pytorch Model of MobileViT

    Args:
        model_name (str): Name of the model
        task (str): Downstream Task (Classification/Segmentation)

    Return:
        model: Pytorch model
    """
    pretrained_model = {
        "classify": {"mobilevit_xxs": "apple/mobilevit-xx-small", "mobilevit_xs": "apple/mobilevit-x-small", "mobilevit_s": "apple/mobilevit-small"},
        "segment": {"mobilevit_xxs": "apple/deeplabv3-mobilevit-xx-small", "mobilevit_xs": "apple/deeplabv3-mobilevit-x-small", "mobilevit_s": "apple/deeplabv3-mobilevit-small"},
    }

    if task == 'classify':
        model = MobileViTForImageClassification.from_pretrained(pretrained_model[task][model_name])
    else:
        model = MobileViTForSemanticSegmentation.from_pretrained(pretrained_model[task][model_name])

    return model


if __name__ == "__main__":
    pdb.set_trace()
    x = get_mobilevit_pt("mobilevit_s", 'segment')
    print(x.state_dict().keys())

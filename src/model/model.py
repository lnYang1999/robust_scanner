from mindspore import nn
from src.model.backbone import ResNet31
from src.model.head import RobustScannerHead

class BaseModel(nn.Cell):
    def __init__(self, config):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()

        # build backbone
        self.backbone = ResNet31()

        # # build head
        config["Head"]['in_channels'] = 3
        self.head = RobustScannerHead(config)

        # 是否返回所有数据
        self.return_all_feats = config.get("return_all_feats", False)

    def construct(self, x, data=None):
        y = dict()
        x = self.backbone(x)
        y["backbone_out"] = x
        x = self.head(x, targets=data)
        if isinstance(x, dict):
            y.update(x)
        else:
            y["head_out"] = x
        if self.return_all_feats:
            return y
        else:
            return x

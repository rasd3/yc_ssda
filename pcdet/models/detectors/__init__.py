from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .pv_rcnn_ssl import PVRCNN_SSL
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .centerpoint import CenterPoint
from .centerpoint_pp_rcnn_v2 import CenterPoint_PointPillar_RCNNV2
from .centerpoint_pp_rcnn_v2_ssl import CenterPoint_PointPillar_RCNNV2_SSL
from .centerpoint_dp_ori import CenterPoint_PointPillar_SingleHead

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'SECONDNetIoU': SECONDNetIoU,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PVRCNN_SSL': PVRCNN_SSL,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'CenterPoint': CenterPoint,
    'CenterPoint_PointPillar_RCNNV2': CenterPoint_PointPillar_RCNNV2,
    'CenterPoint_PointPillar_RCNNV2_SSL': CenterPoint_PointPillar_RCNNV2_SSL,
    'CenterPoint_PointPillar_SingleHead': CenterPoint_PointPillar_SingleHead,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model

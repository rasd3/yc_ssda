from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .roi_head_template import RoIHeadTemplate
from .second_head import SECONDHead
from .center_roi_head import CenterRoIHead
from .roi_head_pillar_dn_v2 import RoIHeadDynamicPillarV2
from .roi_head_template_centerpoint_pointpillar import RoIHeadTemplate_CenterPoint_PointPillar

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'PointRCNNHead': PointRCNNHead,
    'SECONDHead': SECONDHead,
    'CenterRoIHead': CenterRoIHead,
    'RoIHeadDynamicPillarV2': RoIHeadDynamicPillarV2,
    'RoIHeadTemplate_CenterPoint_PointPillar': RoIHeadTemplate_CenterPoint_PointPillar,
}

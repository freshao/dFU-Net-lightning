import collections
from mmcv.utils import build_from_cfg
from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or config dict to be composed
    """
    def __int__(self, transforms):
        # 判断是否支持 此类型接口
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                # 从cfg build model 从配置构建模型
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('Transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially


        :param data (dict): A result dict contains the data to transform.
        :return: dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
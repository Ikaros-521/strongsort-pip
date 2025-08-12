import numpy as np
import torch
import os

from strongsort.reid_multibackend import ReIDDetectMultiBackend
from strongsort.sort.detection import Detection
from strongsort.sort.nn_matching import NearestNeighborDistanceMetric
from strongsort.sort.tracker import Tracker
from strongsort.gsi import gs_interpolation
from strongsort.aflink import AFLink, PostLinker, LinkData

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    if isinstance(x, torch.Tensor):
        y = x.clone()
    else:
        y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

class StrongSORT(object):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        max_dist=0.2,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
        enable_aflink=False,
        enable_gsi=False,
        aflink_model_path=None,
        gsi_interval=20,
        gsi_tau=10,
    ):
        from pathlib import Path
        model_weights = Path(model_weights)
        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)

        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric("cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        
        # AFLink和GSI配置
        self.enable_aflink = enable_aflink
        self.enable_gsi = enable_gsi
        self.aflink_model_path = aflink_model_path
        self.gsi_interval = gsi_interval
        self.gsi_tau = gsi_tau
        
        # 初始化AFLink模型
        if self.enable_aflink and self.aflink_model_path and os.path.exists(self.aflink_model_path):
            self.aflink_model = PostLinker()
            self.aflink_model.load_state_dict(torch.load(self.aflink_model_path))
            self.aflink_dataset = LinkData('', '')
        else:
            self.aflink_model = None
            self.aflink_dataset = None

    def update(self, dets, ori_img):
        import torch  # 在方法开始时导入torch
        
        xyxys = dets[:, :4]
        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        # 处理输入类型，支持numpy数组和PyTorch张量
        if isinstance(clss, torch.Tensor):
            classes = clss.numpy()
        else:
            classes = clss
            
        if isinstance(xyxys, torch.Tensor):
            xywhs = xyxy2xywh(xyxys.numpy())
        else:
            xywhs = xyxy2xywh(xyxys)
            
        if isinstance(confs, torch.Tensor):
            confs = confs.numpy()
        else:
            confs = confs
            
        # 检查图像是否有效
        if ori_img is None:
            raise ValueError("ori_img cannot be None")
        self.height, self.width = ori_img.shape[:2]

        # generate detections
        features = self._get_features(xywhs, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confs)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        # 确保传递给tracker的是PyTorch张量
        if isinstance(clss, torch.Tensor):
            clss_tensor = clss
        else:
            clss_tensor = torch.from_numpy(clss).long()
            
        if isinstance(confs, torch.Tensor):
            confs_tensor = confs
        else:
            confs_tensor = torch.from_numpy(confs).float()
            
        self.tracker.update(detections, clss_tensor, confs_tensor)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)

            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
            
            # 应用GSI处理
            if self.enable_gsi and len(outputs) > 0:
                outputs = self._apply_gsi(outputs)
            
            # 应用AFLink处理
            if self.enable_aflink and self.aflink_model is not None and len(outputs) > 0:
                outputs = self._apply_aflink(outputs)
                
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features
    
    def _apply_gsi(self, tracks):
        """应用GSI处理"""
        try:
            # 检查输入
            if tracks is None or len(tracks) == 0:
                return tracks
            
            # 转换为标准格式 [frame, id, x, y, w, h, conf, -1, -1, -1]
            formatted_tracks = []
            for track in tracks:
                try:
                    x1, y1, x2, y2, track_id, class_id, conf = track
                    x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    formatted_tracks.append([0, track_id, x, y, w, h, conf, -1, -1, -1])
                except (ValueError, IndexError) as e:
                    print(f"GSI: 轨迹数据格式错误: {track}, 错误: {e}")
                    continue
            
            if len(formatted_tracks) == 0:
                return tracks
            
            formatted_tracks = np.array(formatted_tracks)
            processed_tracks = gs_interpolation(formatted_tracks, self.gsi_interval, self.gsi_tau)
            
            # 检查处理结果
            if processed_tracks is None or len(processed_tracks) == 0:
                return tracks
            
            # 转换回原始格式
            result_tracks = []
            for track in processed_tracks:
                try:
                    if len(track) >= 7:
                        frame, track_id, x, y, w, h, conf = track[:7]
                        x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
                        result_tracks.append([x1, y1, x2, y2, track_id, 0, conf])
                except (ValueError, IndexError) as e:
                    print(f"GSI: 处理结果格式错误: {track}, 错误: {e}")
                    continue
            
            return np.array(result_tracks) if result_tracks else tracks
        except Exception as e:
            print(f"GSI处理失败: {e}")
            return tracks
    
    def _apply_aflink(self, tracks):
        """应用AFLink处理"""
        try:
            # 创建临时文件进行AFLink处理
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f_in:
                # 转换为标准格式
                for track in tracks:
                    x1, y1, x2, y2, track_id, class_id, conf = track
                    x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    f_in.write(f"0,{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n")
                path_in = f_in.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f_out:
                path_out = f_out.name
            
            try:
                # 执行AFLink
                linker = AFLink(
                    path_in=path_in,
                    path_out=path_out,
                    model=self.aflink_model,
                    dataset=self.aflink_dataset,
                    thrT=(0, 30),
                    thrS=75,
                    thrP=0.05
                )
                linker.link()
                
                # 读取结果
                processed_tracks = np.loadtxt(path_out, delimiter=',')
                
                # 转换回原始格式
                result_tracks = []
                for track in processed_tracks:
                    frame, track_id, x, y, w, h, conf, _, _, _ = track
                    x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
                    result_tracks.append([x1, y1, x2, y2, track_id, 0, conf])
                
                return np.array(result_tracks) if result_tracks else tracks
                
            finally:
                # 清理临时文件
                os.unlink(path_in)
                os.unlink(path_out)
                
        except Exception as e:
            print(f"AFLink处理失败: {e}")
            return tracks

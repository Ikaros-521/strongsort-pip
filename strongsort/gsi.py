"""
@Author: Du Yunhao
@Filename: gsi.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/1 9:18
@Discription: Gaussian-smoothed interpolation
"""
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR


def linear_interpolation(input_, interval):
    """线性插值"""
    try:
        # 检查输入数组是否为空或形状不正确
        if len(input_) == 0 or input_.ndim != 2:
            return input_
        
        # 确保输入数组有足够的列
        if input_.shape[1] < 2:
            return input_
        
        input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]  # 按ID和帧排序
        output_ = input_.copy()
        
        id_pre, f_pre, row_pre = -1, -1, np.zeros((input_.shape[1],))
        for row in input_:
            f_curr, id_curr = row[:2].astype(int)
            if id_curr == id_pre:  # 同ID
                if f_pre + 1 < f_curr < f_pre + interval:
                    for i, f in enumerate(range(f_pre + 1, f_curr), start=1):  # 逐框插值
                        step = (row - row_pre) / (f_curr - f_pre) * i
                        row_new = row_pre + step
                        output_ = np.append(output_, row_new[np.newaxis, :], axis=0)
            else:  # 不同ID
                id_pre = id_curr
            row_pre = row
            f_pre = f_curr
        
        if len(output_) > 0:
            output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])]
        
        return output_
    except Exception as e:
        print(f"线性插值失败: {e}")
        return input_


def gaussian_smooth(input_, tau):
    """高斯平滑"""
    output_ = list()
    
    # 检查输入数组是否为空或形状不正确
    if len(input_) == 0 or input_.ndim != 2:
        return output_
    
    try:
        ids = set(input_[:, 1])
        for id_ in ids:
            tracks = input_[input_[:, 1] == id_]
            
            # 检查轨迹数据是否足够
            if len(tracks) < 2:
                # 如果轨迹点太少，直接使用原始数据
                for track in tracks:
                    output_.append([track[0], track[1], track[2], track[3], track[4], track[5], track[6], -1, -1, -1])
                continue
            
            try:
                len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
                gpr = GPR(RBF(len_scale, 'fixed'))
                t = tracks[:, 0].reshape(-1, 1)
                x = tracks[:, 2].reshape(-1, 1)
                y = tracks[:, 3].reshape(-1, 1)
                w = tracks[:, 4].reshape(-1, 1)
                h = tracks[:, 5].reshape(-1, 1)
                
                # 拟合和预测
                gpr.fit(t, x)
                xx = gpr.predict(t).flatten()
                gpr.fit(t, y)
                yy = gpr.predict(t).flatten()
                gpr.fit(t, w)
                ww = gpr.predict(t).flatten()
                gpr.fit(t, h)
                hh = gpr.predict(t).flatten()
                
                # 确保所有数组长度一致
                min_len = min(len(xx), len(yy), len(ww), len(hh), len(t))
                for i in range(min_len):
                    output_.append([t[i, 0], id_, xx[i], yy[i], ww[i], hh[i], 1, -1, -1, -1])
                    
            except Exception as e:
                # 如果高斯过程回归失败，使用原始数据
                print(f"高斯平滑失败，使用原始数据: {e}")
                for track in tracks:
                    output_.append([track[0], track[1], track[2], track[3], track[4], track[5], track[6], -1, -1, -1])
                    
    except Exception as e:
        print(f"高斯平滑处理失败: {e}")
        # 返回原始数据
        for track in input_:
            output_.append([track[0], track[1], track[2], track[3], track[4], track[5], track[6], -1, -1, -1])
    
    return output_


def gs_interpolation(tracks, interval=20, tau=10):
    """
    GSI: Gaussian-smoothed interpolation
    
    Args:
        tracks: numpy array with shape (N, 10) containing tracking results
        interval: maximum gap for linear interpolation
        tau: parameter for Gaussian smoothing
    
    Returns:
        processed tracks with interpolation and smoothing
    """
    try:
        # 检查输入
        if tracks is None or len(tracks) == 0:
            return tracks
        
        # 确保tracks是numpy数组
        if not isinstance(tracks, np.ndarray):
            tracks = np.array(tracks)
        
        # 检查数组形状
        if tracks.ndim != 2:
            print(f"GSI: 输入数组维度错误: {tracks.shape}")
            return tracks
        
        # 检查是否有足够的列
        if tracks.shape[1] < 7:  # 至少需要7列: frame, id, x, y, w, h, conf
            print(f"GSI: 输入数组列数不足: {tracks.shape[1]}")
            return tracks
        
        # 执行GSI处理
        li = linear_interpolation(tracks, interval)
        gsi = gaussian_smooth(li, tau)
        
        if len(gsi) > 0:
            return np.array(gsi)
        else:
            return tracks
            
    except Exception as e:
        print(f"GSI处理失败: {e}")
        return tracks


def gs_interpolation_from_file(path_in, path_out, interval=20, tau=10):
    """
    GSI from file to file
    
    Args:
        path_in: input file path
        path_out: output file path
        interval: maximum gap for linear interpolation
        tau: parameter for Gaussian smoothing
    """
    input_ = np.loadtxt(path_in, delimiter=',')
    li = linear_interpolation(input_, interval)
    gsi = gaussian_smooth(li, tau)
    np.savetxt(path_out, gsi, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d') 
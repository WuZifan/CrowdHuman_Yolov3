1、在utils/utils.py里面，进行了两处修改：

    1.1 在non_max_suppression方法内的while循环中，增加了对nan以及inf的处理。
    
```python
while detections.size(0):
            # 处理nan and inf
            temp = np.array(np.isnan(detections[0]), dtype=np.int8)
            if np.sum(temp) > 0 or np.inf in detections[0]:
                # print('there are nan or inf in detections[0],drop', detections[0])
                detections = np.delete(detections, 0, axis=0)
                continue
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]

```
     1.2 在build_targets中，增加了对bbox的边界处理
```python
    gi, gj = gxy.long().t()

    # 就下面这四句话
    gi[gi < 0] = 0
    gj[gj < 0] = 0
    gi[gi > nG - 1] = nG - 1
    gj[gj > nG - 1] = nG - 1

    obj_mask[b, best_n, gj, gi] = 1 # 1表示有obj，0表示没有
    noobj_mask[b, best_n, gj, gi] = 0 # 1表示没有obj，0表示有
```
DMFR-YOLO: an infrared small hotspot detection algorithm based on double multi-scale feature fusion<br>
这是DMFR模块的实现，改进YOLOv8以适应小目标检测任务。<br>
[论文地址]https://iopscience.iop.org/article/10.1088/1361-6501/ad8e77<br>
模块代码应该被复制到YOLOv8的conv.py文件中，并添加import<br>
在task.pyd的parse_model()方法添加：<br>
```python
        elif m in {Concat, MLFR}:
            c2 = sum(ch[x] for x in f)

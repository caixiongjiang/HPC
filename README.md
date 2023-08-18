# HPC
高性能计算课程&amp;CUDA编程实例&amp;深度学习推理框架


* MIT 6.172:[https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/)

* UC Berkeley CS267:[https://sites.google.com/lbl.gov/cs267-spr2021?pli=1](https://sites.google.com/lbl.gov/cs267-spr2021?pli=1)

* 中科大CUDA编程:[https://www.bilibili.com/video/BV1kx411m7Fk/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=841bd3506b40b195573d34fef4c5bdf7](https://www.bilibili.com/video/BV1kx411m7Fk/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=841bd3506b40b195573d34fef4c5bdf7)

* 从零实现深度学习推理框架:[https://www.bilibili.com/video/BV118411f7yM/?spm_id_from=333.1007.top_right_bar_window_history.content.click](https://www.bilibili.com/video/BV118411f7yM/?spm_id_from=333.1007.top_right_bar_window_history.content.click)
    * Armadillo,用于线性代数和科学计算的C++库:[https://arma.sourceforge.net/docs.html](https://arma.sourceforge.net/docs.html)
    * 拉取带有编译环境以及关联库的Docker镜像：
    ```shell
    docker pull registry.cn-hangzhou.aliyuncs.com/hellofss/kuiperinfer:datawhale
    ```
    * 关于学习项目过程中留下的作业（实现一些类内方法和一些opset等）已经在`从零自制深度学习推理框架/src`中实现并通过Google Test
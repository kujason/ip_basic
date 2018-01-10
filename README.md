## Image Processing for Basic Depth Completion (IP-Basic)
Depth completion is the task of converting a sparse depth map D<sub>sparse</sub> into a dense depth map D<sub>dense</sub>. This algorithm was originally created to help visualize 3D object detection results for [AVOD](https://arxiv.org/abs/1712.02294).

### Method
This method uses an unguided approach (images are ignored, only LIDAR projections are used). Basic depth completion is done with OpenCV and NumPy operations in Python. See an earlier version of the algorithm in action [here (2 top views)](https://www.youtube.com/watch?v=Q1f-s6_yHtw). More info (and code?) coming soon.

### Results
|        Method |  `iRMSE` |   `iMAE` |    **RMSE** |      `MAE` | `Runtime` | `FPS` |
|:-------------:|:--------:|:--------:|:-----------:|:----------:|:---------:|:-----:|
|     NadarayaW |     6.34 |     1.84 |     1852.60 |     416.77 |    0.05 s |    20 |
|   SparseConvs |     4.94 |     1.78 |     1601.33 |     481.27 |    0.01 s |   100 |
|        NN+CNN | **3.25** | **1.29** |     1419.75 |     416.14 |    0.02 s |    50 |
|  **IP-Basic** |     3.75 | **1.29** | **1288.46** | **302.60** |   0.017 s |    60 |

Table: Results on the [KITTI Depth Completion benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) (accessed Jan 6, 2018)

### Examples
#### Cars
[sample_006338]: images/006338/006338.png "Sample 006338"
[lidar_006338]: images/006338/lidar.png "Colourized LIDAR points"
[completed_006338]: images/006338/completed.png "Points after depth completion"
- Image:
    ![alt text][sample_006338]
- Colourized LIDAR points (showing ground truth boxes):
    ![alt text][lidar_006338]
- After depth completion (showing ground truth boxes):
    ![alt text][completed_006338]

#### People
[sample_000043]: images/000043/000043.png "Sample 006338"
[lidar_000043]: images/000043/lidar.png "Colourized LIDAR points"
[completed_000043]: images/000043/completed.png "Points after depth completion"
- Image:
    ![alt text][sample_000043]
- Colourized LIDAR points (showing ground truth boxes):
    ![alt text][lidar_000043]
- After depth completion (showing ground truth boxes):
    ![alt text][completed_000043]

import pyrealsense2 as rs
import numpy as np
import cv2
from sympy import *

import mediapipe as mp

# 包含了mediapipe库，实现468关键点检测
class FaceMeshDetector:

    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode    # 持续追踪
        self.maxFaces = maxFaces        # 最大脸数量
        self.minDetectionCon = minDetectionCon  # 阈值 
        self.minTrackCon = minTrackCon          

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh    # 主要依靠facemesh类

        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                 max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        找点：
            img: 输入rgb图像
            param draw: 画or不画
        """
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)   # 主要检测函数
        faces = []
        if self.results.multi_face_landmarks:
            # 对每个脸
            for faceLms in self.results.multi_face_landmarks:
                # 画出脸
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                # 对脸的每个点
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)  # 多个脸（不需要）
        return img, faces

def get_3d_camera_coordinate(depth_pixel, depth_frame, depth_intrin):
    """
    获取3d坐标：
        depth_pixel: 坐标
        depth_frame：深度图
        depth_intrin: 参数
    """
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = depth_frame.get_distance(x, y)        # 获取该像素点对应的深度
    # print ('depth: ',dis)       # 深度单位是m
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate


if __name__ == '__main__':

    # -------------------------------- 配置RealSense ------------------------------- #
    pipeline = rs.pipeline()  # 该管道简化了用户与设备和计算机视觉处理模块的交互。
    config = rs.config()  # 配置允许管道用户请求管道流和设备选择和配置的过滤器。
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))# 提取设备产品线信息

    # 判断是否有RGB camera 
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)# 启动深度图像流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)# 启动彩色图像流，注意分辨率为640*480与深度图保持一致，注意是bgr8

    # Start 
    pipeline.start(config)

    # ------------------------------------ 主循环 ----------------------------------- #
    detector = FaceMeshDetector(maxFaces=1)
    idList_l = [7,33,246,161,160,159,158,157,173,133,155,154,153,145,144,163]   # 左眼：16
    idList_r = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]# 右眼：16
    color = (255, 0, 255)
    x_l = 0
    y_l = 0
    x_r = 0
    y_r = 0
    center_x = 0
    center_y = 0
    dif_x = 0
    dif_y = 0
    deg = 0
    safe_last =0

    
    while True:
        sumx_l = 0
        suny_l = 0
        sumx_r = 0
        suny_r = 0
        # 获取深度图像和rgb图像
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame() # 变量类型为pyrealsense2.depth_frame
        color_frame = frames.get_color_frame() # 变量类型为pyrealsense2.video_frame，get_infrared_frame获得的变量类型也是pyrealsense2.video_frame

        # 获取坐标需要
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics     # 获取深度参数（像素坐标系转相机坐标系会用到）
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics     # 获取相机内参
            
        # 转换为numpy格式
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # 左右镜像
        depth_image = cv2.flip(depth_image,1)
        color_image = cv2.flip(color_image,1)


        # --------------------------------- 找到脸的轮廓并画出 -------------------------------- #
        
        color_image, faces = detector.findFaceMesh(color_image, draw=1) 
        if faces:
            face = faces[0]
            for id in range(0,467):
                # cv2.circle(color_image, face[id], 1, color, cv2.FILLED) #标定左眼
                if id in idList_l:
                    sumx_l += face[id][0]
                    suny_l += face[id][1]
                if id in idList_r:
                    sumx_r += face[id][0]
                    suny_r += face[id][1]
        else:
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # images = np.hstack((color_image, depth_colormap))# 将彩色图和深度图沿着第二轴(宽度)进行连接
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(1)
            continue
        
        # ---------------------------------- 计算关键点2D坐标 --------------------------------- #
        center_x = face[6][0]
        center_y = face[6][1]

        headtop_x = face[152][0]
        headtop_y = face[152][1]

        headdown_x = face[10][0]
        headdown_y = face[10][1]

        # 左眼球坐标[x_l,y_l] 
        x_l = round(sumx_l/16)
        y_l = round(suny_l/16)
        eye_l = [x_l,y_l]
        #cv2.circle(color_image, [x_l,y_l], 1, [255,0,0], cv2.FILLED)
        
        # 右眼球坐标[x_r,y_r]
        x_r = round(sumx_r/16)
        y_r = round(suny_r/16)
        eye_r = [x_r,y_r]
        #cv2.circle(color_image, [x_r,y_r], 1, [255,0,0], cv2.FILLED)

        # 脸部中心坐标 [center_x,center_y]
        center = [center_x,center_y]

        if headtop_x not in range(0,640) or headtop_y not in range(0,480):
            continue
        if headdown_x not in range(0,640) or headdown_y not in range(0,480):
            continue
        if x_r > 640 or x_r < 0 or y_r > 480 or y_r < 0:
            continue
        if x_l > 640 or x_l < 0 or y_l > 480 or y_l < 0:
            continue
        if center_x > 640 or center_x < 0 or center_y > 480 or center_y < 0:
            continue

        # -------------------------------- 根据深度图获取3D坐标 ------------------------------- #
        # 深度图增强
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 找坐标
        dis, camera_coordinate = get_3d_camera_coordinate(center, depth_frame, depth_intrin)
        
        # ----------------------------------- 判断与显示 ---------------------------------- #
        # 判断头与桌子的距离
        safe = camera_coordinate[1]*(-1) + 0.16
        # if safe>1:
        #     continue
        if safe == 0.16 or safe >0.5:
            safe = safe_last
        # cv2.putText(color_image,"Dis:"+str(safe)+" m", (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[0,255,255])
        if safe<0.35:
            cv2.putText(color_image,"Unsafe", (80,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[0,0,255])
        safe_last = safe
        # 判断头的倾斜程度
        dif_x = abs(headtop_x - headdown_x)    
        dif_y = abs(headtop_y - headdown_y)
        deg = atan(dif_x/dif_y) /3.1415926535 *180
        # print(deg)
        if(deg >20):
            cv2.putText(color_image,"Head", (80,160), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[0,0,255])

        cv2.circle(color_image, center, 8, [255,0,255], thickness=-1)
        #cv2.putText(color_image,"Dis:"+str(dis)+" m", (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[0,0,255])
        # cv2.putText(color_image,"X:"+str(camera_coordinate[0])+" m", (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
        # cv2.putText(color_image,"Y:"+str(camera_coordinate[1])+" m", (80,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
        # cv2.putText(colordis_image,"Z:"+str(camera_coordinate[2])+" m", (80,160), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])

        # 画面显示
        # images = np.hstack((color_image, depth_colormap))# 将彩色图和深度图沿着第二轴(宽度)进行连接
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)


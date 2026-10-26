"""Receive-only ZED images, upright once; no policy or robot control."""
import time


class SceneCamera:
    def __init__(self,output):
        import pyzed.sl as sl
        import cv2
        self.sl,self.cv2,self.output,self.index=sl,cv2,output,0
        self.camera=sl.Camera();config=sl.InitParameters()
        config.camera_resolution,config.camera_fps,config.depth_mode=sl.RESOLUTION.HD720,30,sl.DEPTH_MODE.NONE
        if self.camera.open(config)>sl.ERROR_CODE.SUCCESS:raise RuntimeError('Live ZED unavailable')
        self.mat,self.runtime=sl.Mat(),sl.RuntimeParameters()
        output.mkdir(parents=True,exist_ok=False)

    def capture(self):
        import numpy as np
        if self.camera.grab(self.runtime)>self.sl.ERROR_CODE.SUCCESS:raise RuntimeError('Fresh camera grab failed')
        if self.camera.retrieve_image(self.mat,self.sl.VIEW.LEFT,self.sl.MEM.CPU,self.sl.Resolution(640,360))>self.sl.ERROR_CODE.SUCCESS:
            raise RuntimeError('Live image retrieval failed')
        stamp=time.monotonic();frame=np.asarray(self.mat.get_data()).copy()[::-1,::-1]
        bgr=self.cv2.cvtColor(frame,self.cv2.COLOR_BGRA2BGR) if frame.shape[-1]==4 else frame
        path=self.output/f'camera_{self.index:05d}.png';self.index+=1
        if not self.cv2.imwrite(str(path),bgr):raise RuntimeError('Cannot preserve current image')
        return dict(camera_frame=str(path),camera_received_monotonic_s=stamp)

    def close(self):self.camera.close()

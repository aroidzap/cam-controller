import os
import cv2
import numpy as np
import keyboard
import requests

class Compute():
    def _load_model(self):
        # directory
        if not os.path.exists("model"):
            os.makedirs("model")
        # model
        prototxt = ["model/face_detector.prototxt", 
            "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt"]
        caffemodel = ["model/face_detector.caffemodel", 
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"]
        # download
        for file_url in [prototxt, caffemodel]:
            if not os.path.exists(file_url[0]):
                with open(file_url[0], 'wb') as file:
                    print("Downloading '{}' ...".format(file_url[1]))
                    file.write(requests.get(file_url[1], allow_redirects=True).content)
                    print("Done.")
        # load
        return cv2.dnn.readNetFromCaffe(prototxt[0], caffemodel[0])

    def __init__(self):
        self.BBOX_OK_THRESHOLD = 30

        self.STATIC_POSITION_VELOCITY = 15
        self.ZONE_SIZE = 30
        self.DEAD_ZONE = 15

        self.KEY_LEFT = 'left'
        self.KEY_RIGHT = 'right'
        self.KEY_UP = 'up'
        self.KEY_DOWN = 'down'

        self.ARROW_PTS = np.array([[[-3, 0], [-1, 2], 
            [-1, 1], [3, 1], [3, -1], [-1, -1], [-1, -2]]]) / 3

        self.model = self._load_model()
        
        self.tracker = None
        self.tracking_stage = False
        self.enable = True
        self.bbox = None
        self.bbox_ok_cnt = 0
        
        self.last_static_position = None
        self.position = None
        self.frame_size = None

        self.pressed_keys = []

        cv2.namedWindow("Frame", flags=cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow("Frame", 0, 0)

    def compute(self, frame):
        if self.enable:
            if self.tracking_stage:
                if self.tracker is None:
                    if self.bbox is not None:
                        self.tracker = cv2.TrackerCSRT_create()
                        tracker_bbox = self.bbox
                        tracker_bbox[2:4] -= tracker_bbox[0:2]
                        tracker_bbox = tuple(tracker_bbox)
                        self.tracker.init(frame, tracker_bbox)
                        self.last_static_position = None
            
                tracker_ok, tracker_bbox = self.tracker.update(frame)
                tracker_bbox = np.array(tracker_bbox)
                tracker_bbox[2:4] += tracker_bbox[0:2]
                if tracker_ok:
                    self.bbox = tracker_bbox.astype(int)
                    self.bbox_ok_cnt += 1
                else:
                    self.bbox = None
                    self.bbox_ok_cnt -= 1

                if self.bbox_ok_cnt > 0:
                    self.bbox_ok_cnt = 0
                if self.bbox_ok_cnt < -0.5 * self.BBOX_OK_THRESHOLD:
                    self.bbox_ok_cnt = 0
                    self.tracking_stage = False

            else:
                if self.tracker is not None:
                    self.tracker = None

                H, W = 300, 300
                h, w = frame.shape[:2]

                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (H, W)), 1.0, (H, W), (104.0, 177.0, 123.0))

                self.model.setInput(blob)
                detections = self.model.forward()

                self.bbox = None
                self.bbox_ok_cnt -= 1
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.6:
                        self.bbox = np.array(detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
                        self.bbox_ok_cnt += 2
                        break
                
                if self.bbox_ok_cnt < 0:
                    self.bbox_ok_cnt = 0
                if self.bbox_ok_cnt > self.BBOX_OK_THRESHOLD:
                    self.bbox_ok_cnt = 0
                    self.tracking_stage = True

            # update position
            if self.bbox is not None:
                self.position = np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])
                self.frame_size = np.array(frame.shape[:2])[::-1]
        else:
            self.tracker = None
            self.tracking_stage = False
            self.bbox = None
            self.bbox_ok_cnt = 0
            self.last_static_position = None
            self.position = None
            self.frame_size = None
            self.release_keys()

    def draw_gui(self, frame, size = 1):
        gui_frame = frame.copy()
        if self.bbox is not None:
            bbox_center = np.array([
                (self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2
            ])
            bbox_circle_radius = np.linalg.norm([
                (self.bbox[2] - self.bbox[0]) / 2, 
                (self.bbox[3] - self.bbox[1]) / 2
            ])
            bbox_progress = self.bbox_ok_cnt / self.BBOX_OK_THRESHOLD
            if self.tracking_stage:
                cv2.circle(gui_frame, tuple(bbox_center.astype(int)), int(bbox_circle_radius), [255, 128, 128], -1, cv2.LINE_AA)
                cv2.circle(gui_frame, tuple(bbox_center.astype(int)), int(bbox_circle_radius), [255, 64, 64], 7, cv2.LINE_AA)
            else:
                cv2.circle(gui_frame, tuple(bbox_center.astype(int)), int(bbox_circle_radius), [128, 128, 128], -1, cv2.LINE_AA)
                cv2.ellipse(gui_frame, tuple(bbox_center.astype(int)), (int(bbox_circle_radius), int(bbox_circle_radius)), 
                    270, 0, -360 * bbox_progress, [255, 64, 64], 7, cv2.LINE_AA)
            # arrows
            transformed_arrow = None
            if self.KEY_RIGHT in self.pressed_keys:
                transform = np.eye(2)
                transformed_arrow = np.array([[transform @ pt for pt in self.ARROW_PTS[0]]])
            elif self.KEY_LEFT in self.pressed_keys:
                transform = -np.eye(2)
                transformed_arrow = np.array([[transform @ pt for pt in self.ARROW_PTS[0]]])
            elif self.KEY_UP in self.pressed_keys:
                transform = np.eye(2)[::-1,:] * [1, -1]
                transformed_arrow = np.array([[transform @ pt for pt in self.ARROW_PTS[0]]])
            elif self.KEY_DOWN in self.pressed_keys:
                transform = np.eye(2)[::-1,:] * [-1, 1]
                transformed_arrow = np.array([[transform @ pt for pt in self.ARROW_PTS[0]]])
            if transformed_arrow is not None:
                cv2.fillPoly(gui_frame, (bbox_circle_radius * 0.8 * transformed_arrow + bbox_center).astype(np.int32), 
                    [255, 255, 255], cv2.LINE_AA)

        gui_frame = (0.5 * gui_frame + 0.5 * frame).astype(np.uint8)
        if self.tracking_stage:
            if self.last_static_position is not None:
                cv2.circle(gui_frame, tuple(self.last_static_position.astype(int)), 5, [255, 255, 255], -1, cv2.LINE_AA)
            if self.position is not None:
                cv2.circle(gui_frame, tuple(self.position.astype(int)), int(self.ZONE_SIZE), [255, 255, 255], 1, cv2.LINE_AA)
                cv2.circle(gui_frame, tuple(self.position.astype(int)), int(self.ZONE_SIZE + self.DEAD_ZONE), [255, 255, 255], 1, cv2.LINE_AA)
        gui_frame = cv2.resize(gui_frame, (0,0), fx=size, fy=size, interpolation=cv2.INTER_AREA)[:,::-1]
        cv2.imshow("Frame", gui_frame)
        cv2.waitKey(1)

    def release_keys(self):
        for key in self.pressed_keys:
            keyboard.release(key)
            print("RELEASE: {}".format(key))
        self.pressed_keys = []
    
    def press_key(self, key):
        self.pressed_keys.append(key)
        print("PRESS: {}".format(key))
        keyboard.press(key)

    def evaluate_keypress(self):
        if self.tracking_stage and self.position is not None:
            if self.last_static_position is None:
                self.last_static_position = self.position
            # get vector
            vec = self.position - self.last_static_position

            # update last static position
            center_vec = (self.frame_size / 2) - self.last_static_position
            center_dist = np.linalg.norm(center_vec)
            if center_dist > 0:
                center_vec = center_vec / center_dist
            center_force = self.STATIC_POSITION_VELOCITY * (1.5 - 1 / (1 + (center_dist / np.min(self.frame_size / 4)) ** 2))
            if center_force > 0.5 * center_dist:
                center_force = 0.5 * center_dist
            static_vel_clamp = 1
            if np.linalg.norm(vec) > self.STATIC_POSITION_VELOCITY:
                static_vel_clamp *= self.STATIC_POSITION_VELOCITY / np.linalg.norm(vec)
            self.last_static_position = self.last_static_position + static_vel_clamp * vec + center_force * center_vec

            # check vectors
            remap_vec = vec * [-1, -1]
            if len(self.pressed_keys):
                if np.linalg.norm(remap_vec) < self.ZONE_SIZE:
                    self.release_keys()
            else:
                if np.linalg.norm(remap_vec) > self.ZONE_SIZE + self.DEAD_ZONE:
                    if np.argmax(np.abs(remap_vec)) == 0:
                        if remap_vec[0] > 0:
                            self.press_key(self.KEY_RIGHT)
                        else:
                            self.press_key(self.KEY_LEFT)
                    else:
                        if remap_vec[1] > 0:
                            self.press_key(self.KEY_UP)
                        else:
                            self.press_key(self.KEY_DOWN)

class Camera():
    def __init__(self, width = 640, height = 480, exposure = None, gain = None):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if exposure is not None:
            self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self._gain = gain

    def __del__(self):
        self.cam.release()

    def frame(self):
        _, img = self.cam.read()
        if self._gain is not None:
            img = np.minimum(255, (img.astype(np.float32) * self._gain)).astype(np.uint8)
        return img

if __name__ == "__main__":

    cam = Camera(640, 480, exposure = None, gain = None)
    compute = Compute()
    
    while True:
        try:
            frame = cam.frame()
            compute.compute(frame)
            compute.evaluate_keypress()
            compute.draw_gui(frame)
        except KeyboardInterrupt:
            break
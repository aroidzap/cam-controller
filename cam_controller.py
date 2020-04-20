import os
import sys
import argparse
import cv2
import numpy as np
import pyautogui
import requests
import simpleaudio
import queue
import threading

def makeWindowAlwaysOnTop(window_name):
    pass

if os.name == "nt":
    try:
        import win32gui
        import win32con
        
        def makeWindowAlwaysOnTop(window_name):
            hwnd = win32gui.FindWindow(None, window_name)
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE + win32con.SWP_NOSIZE)

    except ImportError:
        pass

def text_image(text, width, height, size = 1, color = [255, 255, 255]):
    font, size, thickness = cv2.FONT_HERSHEY_SIMPLEX, size * 0.8, size * 1
    img = np.zeros((height, width, 3), np.uint8)
    txt_size = cv2.getTextSize(text, font, size, thickness)[0]
    origin = np.array([(img.shape[1] - txt_size[0]) / 2, (img.shape[0] + txt_size[1]) / 2])
    cv2.putText(img, text, tuple(origin.astype(int)), font, size, color, thickness, cv2.LINE_AA)
    return img

class Compute():
    def _show_text(self, text):
        gui_img = text_image(text, 320, 240)
        cv2.resizeWindow(self.WINDOW_NAME, gui_img.shape[:2][::-1])
        cv2.imshow(self.WINDOW_NAME, gui_img)
        cv2.waitKey(1)

    def _download_file(self, file_url):
        if not os.path.exists(file_url[0]):
            if not os.path.exists(os.path.dirname(file_url[0])):
                os.makedirs(os.path.dirname(file_url[0]))
            with open(file_url[0], 'wb') as file:
                self._show_text('Downloading data...')
                print("Downloading '{}' ...".format(file_url[1]))
                file.write(requests.get(file_url[1], allow_redirects=True).content)
                print("Done.")

    def _load_model(self):
        # model
        prototxt = ["data/face_detector.prototxt", 
            "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt"]
        caffemodel = ["data/face_detector.caffemodel", 
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"]
        # download
        for file_url in [prototxt, caffemodel]:
            self._download_file(file_url)
        # load
        return cv2.dnn.readNetFromCaffe(prototxt[0], caffemodel[0])

    def __init__(self, keys = ['up', 'left', 'down', 'right'], always_on_top = True, tracker_downscale = 4.0):
        self.BBOX_OK_THRESHOLD = 30

        self.WINDOW_NAME = "Camera Controller"

        self.STATIC_POSITION_VELOCITY = 15
        self.ZONE_SIZE = 30
        self.DEAD_ZONE = 15

        self.TRACKER_DOWNSCALE = tracker_downscale
        self.TRACKER_DOWNSCALE_INTER = cv2.INTER_LINEAR

        self.KEY_LEFT = keys[1]
        self.KEY_RIGHT = keys[3]
        self.KEY_UP = keys[0]
        self.KEY_DOWN = keys[2]

        self.ARROW_PTS = np.array([[[-3, 0], [-1, 2], 
            [-1, 1], [3, 1], [3, -1], [-1, -1], [-1, -2]]]) / 3
        self.ARROW_PTS = {
            'L' : np.array([[-np.eye(2) @ pt for pt in self.ARROW_PTS[0]]]), 
            'R' : np.array([[np.eye(2) @ pt for pt in self.ARROW_PTS[0]]]), 
            'U' : np.array([[(np.eye(2)[::-1,:] * [1, -1]) @ pt for pt in self.ARROW_PTS[0]]]), 
            'D' : np.array([[(np.eye(2)[::-1,:] * [-1, 1]) @ pt for pt in self.ARROW_PTS[0]]])
        }
        
        self.tracker = None
        self.tracking_stage = False
        self.enable = True
        self.bbox = None
        self.bbox_ok_cnt = 0
        
        self.last_static_position = None
        self.position = None
        self.frame_size = None

        self.pressed_keys = []

        cv2.namedWindow(self.WINDOW_NAME, flags=cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.WINDOW_NAME, 0, 0)
        if always_on_top:
            makeWindowAlwaysOnTop(self.WINDOW_NAME)

        self.model = self._load_model()

        try:
            self.OK_SOUND = simpleaudio.WaveObject.from_wave_file("data/ok.wav")
        except FileNotFoundError:
            self.OK_SOUND = None
        try:
            self.FAIL_SOUND = simpleaudio.WaveObject.from_wave_file("data/fail.wav")
        except FileNotFoundError:
            self.FAIL_SOUND = None

        self._show_text('Loading...')

    def __del__(self):
        cv2.destroyAllWindows()

    def reset(self):
        self.tracker = None
        self.tracking_stage = False
        self.bbox = None
        self.bbox_ok_cnt = 0
        self.last_static_position = None
        self.position = None
        self.frame_size = None
        self.release_keys()

    def compute(self, frame):
        if self.enable:
            if self.tracking_stage:
                tracker_frame = cv2.resize(frame, (0, 0), fx = (1 / self.TRACKER_DOWNSCALE), 
                    fy = (1 / self.TRACKER_DOWNSCALE), interpolation = self.TRACKER_DOWNSCALE_INTER)
                if self.tracker is None:
                    if self.bbox is not None:
                        self.tracker = cv2.TrackerCSRT_create()
                        tracker_bbox = (self.bbox / self.TRACKER_DOWNSCALE).astype(int)
                        tracker_bbox[2:4] -= tracker_bbox[0:2]
                        tracker_bbox = tuple(tracker_bbox)
                        self.tracker.init(tracker_frame, tracker_bbox)
                        self.last_static_position = None

                tracker_ok, tracker_bbox = self.tracker.update(tracker_frame)
                tracker_bbox = np.array(tracker_bbox)
                tracker_bbox[2:4] += tracker_bbox[0:2]
                if tracker_ok:
                    self.bbox = (tracker_bbox * self.TRACKER_DOWNSCALE).astype(int)
                    self.bbox_ok_cnt += 1
                else:
                    self.bbox = None
                    self.bbox_ok_cnt -= 1

                if self.bbox_ok_cnt > 0:
                    self.bbox_ok_cnt = 0
                if self.bbox_ok_cnt < -0.5 * self.BBOX_OK_THRESHOLD:
                    self.reset()
                    if self.FAIL_SOUND is not None:
                        self.FAIL_SOUND.play()

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
                    if self.OK_SOUND is not None:
                        self.OK_SOUND.play()

            # update position
            if self.bbox is not None:
                self.position = np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])
                self.frame_size = np.array(frame.shape[:2])[::-1]
        else:
            self.reset()

    def draw_gui(self, frame, size = 0.5):
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
            
            arrow = None
            if self.KEY_LEFT in self.pressed_keys:
                arrow = self.ARROW_PTS['L']
            elif self.KEY_RIGHT in self.pressed_keys:
                arrow = self.ARROW_PTS['R']
            elif self.KEY_UP in self.pressed_keys:
                arrow = self.ARROW_PTS['U']
            elif self.KEY_DOWN in self.pressed_keys:
                arrow = self.ARROW_PTS['D']
            if arrow is not None:
                cv2.fillPoly(gui_frame, (bbox_circle_radius * 0.8 * arrow + bbox_center).astype(np.int32), 
                    [255, 255, 255], cv2.LINE_AA)

        gui_frame = (0.5 * gui_frame + 0.5 * frame).astype(np.uint8)
        if self.tracking_stage:
            if self.last_static_position is not None:
                cv2.circle(gui_frame, tuple(self.last_static_position.astype(int)), 5, [255, 255, 255], -1, cv2.LINE_AA)
            if self.position is not None:
                cv2.circle(gui_frame, tuple(self.position.astype(int)), int(self.ZONE_SIZE), [255, 255, 255], 1, cv2.LINE_AA)
                cv2.circle(gui_frame, tuple(self.position.astype(int)), int(self.ZONE_SIZE + self.DEAD_ZONE), [255, 255, 255], 1, cv2.LINE_AA)
        gui_frame = cv2.resize(gui_frame, (0,0), fx=size, fy=size, interpolation=cv2.INTER_AREA)[:,::-1]
        
        cv2.resizeWindow(self.WINDOW_NAME, *gui_frame.shape[:2][::-1])
        cv2.imshow(self.WINDOW_NAME, gui_frame)
        key = cv2.waitKey(1)
        if key == 32:
            self.reset()
        run = key != 27
        run = run and cv2.getWindowProperty(self.WINDOW_NAME, 0) >= 0
        return run

    def release_keys(self):
        for key in self.pressed_keys:
            pyautogui.keyUp(key)
            print("RELEASE: {}".format(key))
        self.pressed_keys = []
    
    def press_key(self, key):
        self.pressed_keys.append(key)
        print("PRESS: {}".format(key))
        pyautogui.keyDown(key)

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
    def __init__(self, device_id = 0, width = 640, height = 480, exposure = None, gain = None, buffer_frames = False):
        self.width = width
        self.height = height
        self.buffer_frames = buffer_frames
        if os.name == "nt":
            self.cam = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        else:
            self.cam = cv2.VideoCapture(device_id)
        if not self.cam.isOpened():
            self.cam = None

        if self.cam is not None:
            if not self.buffer_frames:
                self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if exposure is not None:
                self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self._gain = gain

            self._run_camera = True
            self._queue = queue.Queue()
            self._thread = threading.Thread(target=self._camera_reader)
            self._thread.start()

    def stop(self):
        self._run_camera = False

    def __del__(self):
        if self.cam is not None:
            self.stop()
            self._thread.join()
            self.cam.release()

    def _camera_reader(self):
        while self._run_camera:
            frame_ok, frame = self.cam.read()
            if self._gain is not None:
                frame = np.minimum(255, (frame.astype(np.float32) * self._gain)).astype(np.uint8)
            if not frame_ok:
                break
            if (not self.buffer_frames) and (not self._queue.empty()):
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

    def frame(self):
        if self.cam is not None:
            return self._queue.get()
        else:
            return text_image("!!! CAMERA ERROR !!!", self.width, self.height, 2, [0, 0, 255])[:,::-1]
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Camera Controller - head tracking controller for playing simple games')
    parser.add_argument('--no-top', action='store_true', help='disable always on top mode')
    parser.add_argument('--buffer-frames', action='store_true', help='buffer all frames from camera for tracking')
    parser.add_argument('--tracker-downscale', metavar='SCALE', type=float, default=4.0, 
        help='set image downscaling for tracking stage (default is 4.0)')
    parser.add_argument('--wasd', action='store_true', help='enable WASD keys mode (default are arrows)')
    parser.add_argument('--ijkl', action='store_true', help='enable IJKL keys mode (default are arrows)')
    args = parser.parse_args(sys.argv[1:])

    ALWAYS_ON_TOP = not args.no_top
    TRACKER_DOWNSCALE = args.tracker_downscale
    BUFFER_FRAMES = args.buffer_frames

    if args.wasd:
        UP_LEFT_DOWN_RIGHT = ['w', 'a', 's', 'd']
    elif args.wasd:
        UP_LEFT_DOWN_RIGHT = ['i', 'j', 'k', 'l']
    else:
        UP_LEFT_DOWN_RIGHT = ['up', 'left', 'down', 'right']

    cam = Camera(device_id = 0, width = 640, height = 480, exposure = None, gain = None, buffer_frames = BUFFER_FRAMES)
    compute = Compute(keys = UP_LEFT_DOWN_RIGHT, always_on_top = ALWAYS_ON_TOP, tracker_downscale = TRACKER_DOWNSCALE)
    
    while True:
        try:
            frame = cam.frame()
            compute.compute(frame)
            compute.evaluate_keypress()
            if not compute.draw_gui(frame):
                break
        except KeyboardInterrupt:
            break

    cam.stop()
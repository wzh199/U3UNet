import base64

import cv2
import numpy
import numpy as np
import websocket
import _thread


def on_message(ws, message):
    data = np.frombuffer(message, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    cv2.imshow("Video Stream", image)
    cv2.waitKey(1)


def on_error(ws, error):
    print("Error:", error)


def on_close(ws):
    print("Closed")


def on_open(ws):
    # cap = cv2.VideoCapture(0)
    # while True:
    # ret, frame = cap.read()
    # if ret:
    # data = cv2.imencode(".jpg", frame)[1].tobytes()
    print("onopen")
    frame = cv2.imread("frontframe.jpg")
    imgencode = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY),95])
    # 建立矩阵
    data = numpy.array(imgencode)
    print(data)
    # 将numpy矩阵转换成字符形式，以便在网络中传输
    stringData = data.tobytes()
    print(str(stringData))
    # ws.send(str(stringData))
    # ws.send(b"123")

    f = open("frontframe.jpg", "rb")
    ls_f = base64.b64encode(f.read())
    print(type(ls_f))
    ws.send(str(ls_f))




cv2.destroyAllWindows()

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://150.158.137.155:8080/webSocket/1/front",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

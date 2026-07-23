import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

BLUE_ID = 1
# > 파란색 큐브?ID: 1
GREEN_ID = 2
# > 초록색 큐브?ID: 2

# HSV(OpenCV 기준 H:0~179) 범위. Isaac Sim 큐브 색상에 맞춰 조정 필요할 수 있음.
BLUE_LOWER, BLUE_UPPER = (100, 80, 50), (130, 255, 255)
GREEN_LOWER, GREEN_UPPER = (40, 80, 50), (80, 255, 255)

MIN_PIXELS = 500  # 이 이하면 색상이 화면에 없다고 판단하고 발행 안 함


class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.bridge = CvBridge()
        # > 뭐하는 코드? : 번역기(CvBridge)를 하나 만듦 self.bridge라는 이름으로 보관해두는 줄
        # > 왜 이렇게 짬? : 사진이 올 때마다 번역기를 새로 만들면 낭비니까, 미리 하나 만들어서 계속 재사용하려고.
        # > 흐름: 나중에 사진이 도착하면 저장해둔 번역기(self.bridge)를 꺼내서 사용
        self.create_subscription(Image, '/rgb', self.on_image, 10)
        # > 뭐하는 코드?: "/rgb"라는 우편함을 계속 지켜보다가, 편지(사진)가 오면 on_image 함수를 자동으로 실행해줘"라고 예약하는 줄
        # > 왜 이렇게 짬?: 우리가 직접 "사진 왔나?"하고 계속 확인할 필요 없이, 오면 알아서 실행되게 하려고
        #                  마지막 숫자 10은 "한 번에 최대 10개까지만 밀린 편지를 쌓아둔다"라는 뜻
        # > 흐름: 이건 "예약"만 해두는 것. 진짜 실행은 나중에 사진이 실제로 도착했을 때 일어남.
        self.pub = self.create_publisher(Int32, '/color_id', 10)
        # > 뭐하는 코드?: "/color_id"라는 우편함으로 숫자(Int32)를 보낼 수 있는 발송 창구"를 만들어서 self.pub에 저장해두는 줄
        # > 왜 이렇게 짬?: 나중에 판단 결과(파랑=1, 초록=2)를 로봇 쪽으로 보내야 하니깐, 그 보내는 창구를 미리 만들어 두는 것
        # > 흐름: 아직 아무것도 안 보냄. 나중에 'self.pub.publish(...)'를 호출할 때 진짜로 발송됨.


    def on_image(self, msg: Image):
        # > on_image: 사진이 도착할 때마다 자동으로 실행되는 부분
        # > 뭐하는 코드?: /rgb에 사진이 도착할 때마다 자동으로 실행되는 함수
        # > 왜 이렇게 짬?: 위에서 예약해둔 대로, "사진이 오면 이 함수 실행해줘"의 그 함수가 바로 이것.
        # > 흐름: 사진 한 장 도착 = 이 함수 한 번 실행. 사진이 계속 오면 이 함수도 계속 반복 실행
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # > 뭔 코드?: ROS 포장지에 싸인 사진(msg)을, 우리가 다룰 수 있는 보통 사진(image)으로 풀어주는 줄.
        # > 왜 이렇게 짬?: OpenCV 함수들은 ROS 포맷을 못 읽음.
        # > 흐름: 이제부터 image는 진짜 사진 데이터라서, 아래 줄부터 마음대로 분석할 수 있다.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # > 뭔 코드?: 사진의 색 표현 방식을 BGR(파랑/초록/빨강 조합)에서 HSV(색상/채도/밝기)로 바꾸는 줄
        # > 왜 이렇게 짬?: BGR은 조명이 밝냐 어둡냐에 따라 숫자가 크게 흔들려서 "이게 파란색이다"라고 판단하기 어려움.
        #                HSV는 "색상(Hue)"이 조명 영향을 덜 받아서, 색 구분이 훨씬 쉬움.
        # > 흐름: 이제부터는 HSV를 가지고 색을 판별 

        blue_count = cv2.countNonZero(cv2.inRange(hsv, np.array(BLUE_LOWER), np.array(BLUE_UPPER)))
        # > 뭔 코드?: 사진 안에서 "파란색 범위에 들어가는 픽셀이 몇 개인지" 세는 줄이다.
        # > 왜 이렇게 짬?: cv2.inRange가 파란색 범위 안에 있는 픽셀 흰색(1), 아니면 검은색(0)으로 칠해준 지도를 만들어줌,
        #                 countNonZero가 그 흰색 칸 개수를 세준다. 
        # > 흐름: 결과는 숫자 하나.
        green_count = cv2.countNonZero(cv2.inRange(hsv, np.array(GREEN_LOWER), np.array(GREEN_UPPER)))
        # > 뭔 코드?: 'blue_count'랑 같은 방식, 초록색 기준으로 센다.
        # > 흐름: 이제 blue_count, green_count 두 숫자가 다 준비됌.
        if blue_count < MIN_PIXELS and green_count < MIN_PIXELS:
            return
        # > 뭔 코드?: 둘 다 너무 적으면(500개 미만) 함수를 여기서 그냥 끝내버리는 줄이다.
        # > 왜 이렇게 짬?: 큐브가 화면에 없는데 엉뚱한 색을 발표하는 걸 막는 용도
        # > 흐름: 조건이 맞으면 여기서 함수 끝. 안 맞으면 담 줄로 넘어감.

        color_id = BLUE_ID if blue_count > green_count else GREEN_ID
        # > 뭔 코드?: 파란색 개수가 더 많으면 1, 아니면(초록이 더 많으면) 2를 고르는 줄이다.
        # > 왜 이렇게 짬?: "더 많이 보이는 색"을 정답으로 삼는 게 제일 간단하고 확실한 방법
        # > 흐름: 이제 color_id에 1 또는 2가 딱 정해짐
        self.pub.publish(Int32(data=color_id))
        # > 뭔 코드?: 아까 만들어둔 발송 창구(self.pub)로 color_id 값을 진짜로 발송하는 줄
        # > 왜 이렇게 짬?: 이 노드의 최종 목표. -- 판단 결과를 로봇 쪽에 전달 하는 것.
        # > 흐름: 이 줄이 실행되면 /color_id 토픽을 구독하고 있는 다른 노드(나중에 로봇)가 이 값을 받게 됌.
        self.get_logger().info(f'color_id={color_id} (blue={blue_count}, green={green_count})')
        # > 뭔 코드?: 지금 어떤 판단을 내렸는지 화면(터미널)에 글자로 찍어주는 줄이다.
        # > 왜 이렇게 짬?: 로봇 제어랑은 상관X 우리가 눈으로 "잘 작동하나?"확인하기 위한 디버깅용
        # > 흐름: 로그만 찍고 함수는 여기서 끝. 다음 사진이 오면 on_image가 처음부터 다시 실행됌.

def main():
    rclpy.init()
    node = ColorDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

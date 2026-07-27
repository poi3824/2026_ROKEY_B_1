# amr_node -> behavior_node : 도착/이동/오류 상태 보고
# PUB /amr/status

uint8 STATE_IDLE = 0
uint8 STATE_MOVING = 1
uint8 STATE_ARRIVED = 2
uint8 STATE_ERROR = 3

uint8 state
string station_id
string message

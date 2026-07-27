# behavior_node -> fleet_manager_node : 작업 완료/실패 보고
# PUB /fleet/report

string job_id
string station_id
bool success
string message
builtin_interfaces/Time stamp

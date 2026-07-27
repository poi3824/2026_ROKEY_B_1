# fleet_manager_node -> behavior_node : 처리할 작업 지시
# PUB /fleet/job

string job_id
string station_id
string job_type       # ASSEMBLE, FASTEN 등
string target
builtin_interfaces/Time stamp

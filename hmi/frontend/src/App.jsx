import { useEffect, useRef, useState, useCallback } from 'react'
import { io } from 'socket.io-client'
import './App.css'

const STATIONS = Array.from({ length: 6 }, (_, index) => {
  const number = index + 1
  return {
    id: `station_${number}`,
    number,
    available: number >= 1 && number <= 3,
    status: number >= 1 && number <= 3 ? 'READY' : '미설정',
  }
})

const MANUAL_TASK_GROUPS = [
  {
    id: 'busbar',
    title: '버스바 공정',
    subtitle: '스캔부터 최종 체결까지 단계별 제어',
    tasks: [
      ['SCAN_BATTERY', '배터리 스캔'],
      ['SCAN_BUSBAR', '버스바 스캔'],
      ['PICK_BUSBAR', '버스바 파지'],
      ['MOVE_BATTERY_CENTER', '배터리 중심 이동'],
      ['FINE_ALIGNMENT', '정밀 정렬 실행'],
      ['ASSEMBLE_BUSBAR', '버스바 체결 실행'],
      ['RETURN_HOME', '너트 준비 자세 복귀'],
    ],
  },
  {
    id: 'nut1',
    title: '너트 1 공정',
    subtitle: '첫 번째 너트 스캔·파지·체결',
    tasks: [
      ['SCAN_NUT1', '너트 1 스캔'],
      ['PICK_NUT1', '너트 1 파지'],
      ['ASSEMBLE_NUT1', '너트 1 체결 실행'],
    ],
  },
  {
    id: 'nut2',
    title: '너트 2 공정',
    subtitle: '두 번째 너트 스캔·파지·체결',
    tasks: [
      ['SCAN_NUT2', '너트 2 스캔'],
      ['PICK_NUT2', '너트 2 파지'],
      ['ASSEMBLE_NUT2', '너트 2 체결 실행'],
    ],
  },
]

const EMPTY_STATE = {
  task_command: null,
  fsm_state: null,
  alignment: null,
  emergency_stop: false,
  isaac_phase: null,
  isaac_progress: 0,
  isaac_status: null,
  amr_status: null,
  amr_pose: { x: 0, y: 0, theta: 0 },
  last_job: null,
  last_report: null,
  camera: {
    selected: 'battery_4',
    topic: '/camera_bolt/rgb',
    online: false,
    fps: 0,
    width: 0,
    height: 0,
  },
  manual_task: {
    task_type: null,
    status: 'IDLE',
    message: null,
  },
  updated_at: null,
}

function StatusPill({ text, tone }) {
  return <span className={`pill pill-${tone}`}>{text ?? '—'}</span>
}

function isaacStatusTone(status) {
  if (!status) return 'idle'
  if (status === 'SUCCESS') return 'ok'
  if (status.startsWith('FAILURE')) return 'error'
  return 'idle'
}

function amrStateTone(state) {
  if (state === 'ARRIVED') return 'ok'
  if (state === 'ERROR') return 'error'
  if (state === 'MOVING') return 'busy'
  return 'idle'
}

function fsmStateTone(state) {
  if (!state || state === 'IDLE') return 'idle'
  if (state === 'FAILURE' || state === 'EMERGENCY_STOP') return 'error'
  if (state === 'SUCCESS') return 'ok'
  return 'busy'
}

function App() {
  const [state, setState] = useState(EMPTY_STATE)
  const [connected, setConnected] = useState(false)
  const [station, setStation] = useState('station_1')
  const [log, setLog] = useState([])
  const [cameras, setCameras] = useState([])
  const [streamKey, setStreamKey] = useState(0)
  const [cameraExpanded, setCameraExpanded] = useState(false)
  const [debugOpen, setDebugOpen] = useState(false)
  const [controlMode, setControlMode] = useState('production')
  const cameraPanelRef = useRef(null)
  const previousStateRef = useRef(EMPTY_STATE)

  const pushLog = useCallback((text) => {
    setLog((prev) => [{ text, at: new Date().toLocaleTimeString() }, ...prev].slice(0, 20))
  }, [])

  useEffect(() => {
    fetch('/api/state')
      .then((r) => r.json())
      .then(setState)
      .catch(() => pushLog('초기 상태 조회 실패 (백엔드 실행 중인지 확인)'))
    fetch('/api/cameras')
      .then((r) => r.json())
      .then(setCameras)
      .catch(() => pushLog('카메라 목록 조회 실패'))

    const socket = io('/', { path: '/socket.io' })
    socket.on('connect', () => setConnected(true))
    socket.on('disconnect', () => setConnected(false))
    socket.on('state', (payload) => setState(payload))
    return () => socket.disconnect()
  }, [pushLog])

  useEffect(() => {
    const previous = previousStateRef.current
    if (previous.fsm_state && previous.fsm_state !== state.fsm_state) {
      pushLog(`FSM 전환: ${previous.fsm_state} → ${state.fsm_state}`)
    }
    if (previous.amr_status?.state && previous.amr_status.state !== state.amr_status?.state) {
      pushLog(`AMR 상태: ${previous.amr_status.state} → ${state.amr_status?.state}`)
    }
    if (previous.emergency_stop !== state.emergency_stop) {
      pushLog(state.emergency_stop ? '비상정지 상태 감지' : '비상정지 해제 상태 감지')
    }
    previousStateRef.current = state
  }, [state, pushLog])

  const startJob = async () => {
    if (state.emergency_stop) {
      pushLog('Job 발행 차단: 비상정지를 먼저 해제하세요')
      return
    }
    if (!STATIONS.find((item) => item.id === station)?.available) {
      pushLog(`${station} 작업 차단: 좌표가 설정되지 않았습니다`)
      return
    }
    if (!window.confirm(`${station}에서 전체 조립 공정을 시작하시겠습니까?`)) return
    pushLog(`전체 조립 공정 요청 → ${station}`)
    try {
      const res = await fetch('/api/job', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ station_id: station, job_type: 'ASSEMBLE', target: 'busbar_and_nut' }),
      })
      const data = await res.json()
      if (res.ok) pushLog(`Job 발행됨: ${data.job_id} (${data.station_id})`)
      else pushLog(`Job 발행 실패: ${data.error ?? res.status}`)
    } catch {
      pushLog('Job 발행 실패 (네트워크 오류)')
    }
  }

  const runManualTask = async (taskType, label) => {
    if (state.emergency_stop) {
      pushLog('세부 작업 차단: 비상정지를 먼저 해제하세요')
      return
    }
    if (!STATIONS.find((item) => item.id === station)?.available) {
      pushLog(`${station} 작업 차단: 좌표가 설정되지 않았습니다`)
      return
    }
    if (!window.confirm(`${station}에서 '${label}' 작업을 실행하시겠습니까?\n\n선행 단계와 작업 영역 안전을 확인하세요.`)) return
    pushLog(`세부 작업 요청: ${label} → ${station}`)
    try {
      const res = await fetch('/api/manual-task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ station_id: station, task_type: taskType }),
      })
      const data = await res.json()
      if (res.ok) pushLog(`${label} 요청 전송됨`)
      else pushLog(`${label} 실행 실패: ${data.error ?? data.message ?? res.status}`)
    } catch {
      pushLog(`${label} 실행 실패 (네트워크 오류)`)
    }
  }

  const moveAmrOnly = async (targetKind, label) => {
    if (state.emergency_stop) {
      pushLog('AMR 이동 차단: 비상정지를 먼저 해제하세요')
      return
    }
    if (!window.confirm(`${station}의 ${label}(으)로 AMR만 이동하시겠습니까?`)) return
    pushLog(`AMR 단독 이동 요청: ${label} → ${station}`)
    try {
      const res = await fetch('/api/amr/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ station_id: station, target_kind: targetKind }),
      })
      const data = await res.json()
      if (res.ok) pushLog(`AMR 이동 명령 전송됨: ${label}`)
      else pushLog(`AMR 이동 실패: ${data.error ?? res.status}`)
    } catch {
      pushLog('AMR 이동 실패 (네트워크 오류)')
    }
  }

  const resetSystem = async () => {
    if (!window.confirm('진행 중인 작업을 모두 폐기하고 Isaac 월드, AMR, 로봇 팔과 부품을 초기 상태로 되돌리시겠습니까?')) return
    pushLog('시스템 제어 상태 초기화 요청')
    try {
      const res = await fetch('/api/system-reset', { method: 'POST' })
      const data = await res.json()
      if (res.ok) pushLog('제어 상태 초기화 명령 전송됨')
      else pushLog(`초기화 실패: ${data.error ?? res.status}`)
    } catch {
      pushLog('초기화 실패 (네트워크 오류)')
    }
  }

  const setEmergencyStop = async (enabled) => {
    if (!enabled && !window.confirm('주변 안전을 확인했습니까? 비상정지를 해제하면 새 작업을 시작할 수 있습니다.')) {
      return
    }
    pushLog(enabled ? '비상정지 요청' : '비상정지 해제 요청')
    try {
      const res = await fetch('/api/emergency-stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      })
      const data = await res.json()
      if (res.ok) pushLog(enabled ? '비상정지 활성화됨' : '비상정지 해제됨')
      else pushLog(`비상정지 제어 실패: ${data.error ?? res.status}`)
    } catch {
      pushLog('비상정지 제어 실패 (네트워크 오류)')
    }
  }

  const selectCamera = async (cameraId) => {
    try {
      const res = await fetch('/api/camera/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: cameraId }),
      })
      if (!res.ok) throw new Error()
      setStreamKey((key) => key + 1)
      pushLog(`카메라 전환: ${cameras.find((camera) => camera.id === cameraId)?.label ?? cameraId}`)
    } catch {
      pushLog('카메라 전환 실패')
    }
  }

  const toggleFullscreen = async () => {
    if (!document.fullscreenElement) {
      await cameraPanelRef.current?.requestFullscreen()
    } else {
      await document.exitFullscreen()
    }
  }

  return (
    <div className={`app ${state.emergency_stop ? 'app-estopped' : ''}`}>
      <header className="header">
        <div className="brand">
          <div className="brand-mark">EV</div>
          <div>
            <span className="eyebrow">BATTERY ASSEMBLY CONTROL</span>
            <h1>EV Combine <em>HMI</em></h1>
          </div>
        </div>
        <div className="header-tools">
          <div className="unified-label"><span>LIVE</span> 통합 작업 관제</div>
          <button
            className={`debug-toggle ${debugOpen ? 'active' : ''}`}
            onClick={() => setDebugOpen((value) => !value)}
          >
            {debugOpen ? '관제 화면' : '디버깅'}
          </button>
        </div>
        <div className="header-meta">
          <div className="live-status">
            <span className={`conn-dot ${connected ? 'conn-ok' : 'conn-off'}`} />
            <div>
              <strong>{connected ? 'LIVE' : 'OFFLINE'}</strong>
              <span>{connected ? 'ROS 브리지 연결됨' : '연결 확인 필요'}</span>
            </div>
          </div>
          <div className="update-time">
            <span>LAST UPDATE</span>
            <strong>{state.updated_at ? new Date(state.updated_at * 1000).toLocaleTimeString('ko-KR') : '--:--:--'}</strong>
          </div>
          {state.emergency_stop && <span className="estop-banner">● 비상정지 활성</span>}
        </div>
      </header>

      <section className="overview-strip">
        <div className="overview-item">
          <span>시스템</span>
          <strong className={connected ? 'text-ok' : 'text-error'}>{connected ? 'ONLINE' : 'OFFLINE'}</strong>
        </div>
        <div className="overview-item">
          <span>현재 FSM</span>
          <strong>{state.fsm_state ?? 'NO DATA'}</strong>
        </div>
        <div className="overview-item">
          <span>AMR</span>
          <strong>{state.amr_status?.state ?? 'IDLE'}</strong>
        </div>
        <div className="overview-item">
          <span>정렬 추적</span>
          <strong className={state.alignment?.valid ? 'text-ok' : ''}>{state.alignment?.valid ? 'VALID' : 'STANDBY'}</strong>
        </div>
      </section>

      <main className={`grid unified-grid ${debugOpen ? 'debug-open' : ''}`}>
        <section className="card control-card request-only">
          <div className="card-heading control-heading">
            <span className="card-icon">CTRL</span>
            <div><span>OPERATOR CONTROL</span><h2>작업 제어 콘솔</h2></div>
            <div className="control-mode-switch">
              <button className={controlMode === 'production' ? 'active' : ''} onClick={() => setControlMode('production')}>전체 공정</button>
              <button className={controlMode === 'manual' ? 'active' : ''} onClick={() => setControlMode('manual')}>세부 단계</button>
            </div>
          </div>

          <div className="control-section">
            <div className="control-section-title"><span>01</span><div><strong>작업 스테이션</strong><small>실행할 배터리 스테이션을 선택하세요</small></div></div>
            <div className="station-grid">
              {STATIONS.map((item) => (
                <button
                  key={item.id}
                  className={`${station === item.id ? 'active' : ''} ${item.available ? '' : 'unavailable'}`}
                  onClick={() => setStation(item.id)}
                >
                  <span>STATION</span>
                  <strong>{String(item.number).padStart(2, '0')}</strong>
                  <small>{item.status}</small>
                </button>
              ))}
            </div>
          </div>

          {controlMode === 'production' ? (
            <div className="production-control control-section">
              <div className="control-section-title"><span>02</span><div><strong>전체 조립 공정</strong><small>AMR 이동부터 버스바·너트 체결까지 자동 실행</small></div></div>
              <div className="production-action">
                <div>
                  <span>선택 스테이션</span>
                  <strong>{station.replace('_', ' ').toUpperCase()}</strong>
                  <small>배터리 스캔 → 버스바 파지·체결 → 너트 1·2 체결</small>
                </div>
                <button
                  className="btn btn-primary btn-run-production"
                  onClick={startJob}
                  disabled={state.emergency_stop || !STATIONS.find((item) => item.id === station)?.available}
                >
                  전체 조립 공정 실행
                </button>
              </div>
            </div>
          ) : (
            <div className="manual-control control-section">
              <div className="control-section-title"><span>02</span><div><strong>세부 작업 단계</strong><small>선행 작업 상태를 확인한 후 개별 단계를 실행하세요</small></div></div>
              <div className="manual-status">
                <span>현재 세부 작업</span>
                <strong>{state.manual_task?.task_type ?? '대기 중'}</strong>
                <StatusPill
                  text={state.manual_task?.status}
                  tone={state.manual_task?.status === 'SUCCESS' ? 'ok' : ['FAILED', 'REJECTED', 'CANCELED'].includes(state.manual_task?.status) ? 'error' : state.manual_task?.status === 'RUNNING' ? 'busy' : 'idle'}
                />
                <small>{state.manual_task?.message ?? '실행 중인 세부 작업이 없습니다'}</small>
              </div>
              <div className="task-groups">
                <div className="task-group movement-group">
                  <div><strong>AMR 단독 이동</strong><span>선택한 스테이션의 지정 위치까지만 이동</span></div>
                  <div className="task-buttons">
                    <button
                      disabled={state.emergency_stop || state.fsm_state !== 'IDLE'}
                      onClick={() => moveAmrOnly('battery', '배터리 작업 위치')}
                    >
                      <span>MOVE_BATTERY</span><strong>배터리 위치 이동</strong><i>실행</i>
                    </button>
                    <button
                      disabled={state.emergency_stop || state.fsm_state !== 'IDLE'}
                      onClick={() => moveAmrOnly('busbar', '버스바 공급 위치')}
                    >
                      <span>MOVE_BUSBAR</span><strong>버스바 위치 이동</strong><i>실행</i>
                    </button>
                  </div>
                </div>
                {MANUAL_TASK_GROUPS.map((group) => (
                  <div className="task-group" key={group.id}>
                    <div><strong>{group.title}</strong><span>{group.subtitle}</span></div>
                    <div className="task-buttons">
                      {group.tasks.map(([taskType, label]) => (
                        <button
                          key={taskType}
                          onClick={() => runManualTask(taskType, label)}
                          disabled={
                            state.emergency_stop
                            || state.fsm_state !== 'IDLE'
                            || ['WAITING', 'RUNNING'].includes(state.manual_task?.status)
                            || !STATIONS.find((item) => item.id === station)?.available
                          }
                        >
                          <span>{taskType}</span>
                          <strong>{label}</strong>
                          <i>실행</i>
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="estop-control">
            <div>
              <strong>안전 제어</strong>
              <span>{state.emergency_stop ? '비상정지 활성 · 모든 신규 작업 차단' : '시스템 정상 · 비상정지 대기'}</span>
            </div>
            <button
              className="btn btn-estop"
              disabled={state.emergency_stop}
              onClick={() => setEmergencyStop(true)}
            >
              {state.emergency_stop ? '비상정지 활성' : '비상정지'}
            </button>
            <button
              className="btn btn-release"
              disabled={!state.emergency_stop}
              onClick={() => setEmergencyStop(false)}
            >
              작업 재개
            </button>
            <button className="btn btn-reset" onClick={resetSystem}>
              안전 초기화
            </button>
          </div>
          <div className="operation-summary">
            <div className="summary-title"><span>운전 인터록</span><strong>{state.emergency_stop ? 'LOCKED' : 'READY'}</strong></div>
            <div className="summary-grid">
              <div><span>선택 스테이션</span><strong>{station.replace('station_', 'ST-')}</strong></div>
              <div><span>FSM</span><strong>{state.fsm_state ?? 'NO DATA'}</strong></div>
              <div><span>AMR</span><strong>{state.amr_status?.state ?? 'IDLE'}</strong></div>
              <div><span>정렬</span><strong>{state.alignment?.valid ? 'VALID' : 'STANDBY'}</strong></div>
            </div>
            <p>{state.emergency_stop
              ? '작업 재개를 누르면 인터록을 해제하고 중단된 공정 체크포인트부터 이어갑니다.'
              : '모든 인터록 정상 · HMI 작업 명령 대기 중'}</p>
          </div>
          <p className="hint">
            세부 단계는 Arm Action을 직접 실행합니다. 반드시 선행 단계와 로봇 주변 안전 상태를 확인하세요.
          </p>
        </section>

        <section ref={cameraPanelRef} className={`card camera-card monitor-only ${cameraExpanded ? 'camera-expanded' : ''}`}>
          <div className="card-heading">
            <span className="card-icon">CAM</span>
            <div><span>LIVE VISION FEED</span><h2>작업 카메라</h2></div>
            <div className="camera-toolbar">
              <select
                aria-label="카메라 선택"
                value={state.camera?.selected ?? 'battery_4'}
                onChange={(event) => selectCamera(event.target.value)}
              >
                {cameras.map((camera) => <option key={camera.id} value={camera.id}>{camera.label}</option>)}
              </select>
              <a className="tool-btn" href="/api/camera/snapshot" download title="현재 화면 저장">↓ 저장</a>
              <button className="tool-btn" onClick={() => setCameraExpanded((value) => !value)}>{cameraExpanded ? '축소' : '확대'}</button>
              <button className="tool-btn" onClick={toggleFullscreen}>전체화면</button>
            </div>
          </div>
          <div className="camera-layout">
            <div className="video-stage">
              <img
                key={streamKey}
                src={`/api/camera/stream?v=${streamKey}`}
                alt="ROS 작업 카메라 실시간 영상"
              />
              <div className="camera-grid-overlay" />
              <div className="video-corners" />
              <span className="record-indicator"><i /> LIVE</span>
              {!state.camera?.online && (
                <div className="no-signal">
                  <strong>NO SIGNAL</strong>
                  <span>{state.camera?.topic ?? '카메라 토픽'} 프레임 대기 중</span>
                </div>
              )}
              <span className="camera-name">{cameras.find((camera) => camera.id === state.camera?.selected)?.label ?? state.camera?.selected}</span>
            </div>
            <aside className="camera-telemetry">
              <div><span>STATUS</span><strong className={state.camera?.online ? 'text-ok' : 'text-error'}>{state.camera?.online ? 'STREAMING' : 'WAITING'}</strong></div>
              <div><span>TOPIC</span><strong>{state.camera?.topic ?? '—'}</strong></div>
              <div><span>RESOLUTION</span><strong>{state.camera?.width ? `${state.camera.width} × ${state.camera.height}` : '—'}</strong></div>
              <div><span>FRAME RATE</span><strong>{state.camera?.online ? `${state.camera.fps.toFixed(1)} FPS` : '—'}</strong></div>
              <div><span>ALIGNMENT</span><strong>{state.alignment?.active ? 'CORRECTING' : 'STANDBY'}</strong></div>
              <p>화면 중앙 십자선은 작업 위치 확인용이며 실제 비전 판정에는 영향을 주지 않습니다.</p>
            </aside>
          </div>
        </section>

        <section className="card card-process monitor-only">
          <div className="card-heading"><span className="card-icon">01</span><div><span>PROCESS</span><h2>공정 진행 상태</h2></div></div>
          <div className="row"><span>실제 FSM</span>
            <StatusPill
              text={state.fsm_state}
              tone={fsmStateTone(state.fsm_state)}
            />
          </div>
          <div className="row"><span>현재 Task</span><StatusPill text={state.task_command} tone="idle" /></div>
          <div className="row"><span>Isaac Phase</span><StatusPill text={state.isaac_phase} tone="idle" /></div>
          <div className="row">
            <span>진행률</span>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, state.isaac_progress || 0))}%` }} />
            </div>
            <span className="progress-num">{(state.isaac_progress || 0).toFixed(0)}%</span>
          </div>
          <div className="row"><span>Isaac 결과</span><StatusPill text={state.isaac_status} tone={isaacStatusTone(state.isaac_status)} /></div>
        </section>

        <section className="card card-alignment monitor-only">
          <div className="card-heading"><span className="card-icon">02</span><div><span>VISION ALIGNMENT</span><h2>실시간 정렬 오차</h2></div></div>
          <div className="row"><span>추적 상태</span>
            <StatusPill
              text={state.alignment ? (state.alignment.valid ? (state.alignment.active ? '보정 중' : '추적 준비') : '유효 데이터 없음') : null}
              tone={state.alignment?.valid ? (state.alignment.active ? 'busy' : 'ok') : 'idle'}
            />
          </div>
          <div className="alignment-values">
            <div><span>dx</span><strong>{state.alignment ? `${state.alignment.dx_px > 0 ? '+' : ''}${state.alignment.dx_px} px` : '—'}</strong></div>
            <div><span>dy</span><strong>{state.alignment ? `${state.alignment.dy_px > 0 ? '+' : ''}${state.alignment.dy_px} px` : '—'}</strong></div>
            <div><span>dTheta</span><strong>{state.alignment ? `${state.alignment.dtheta_deg > 0 ? '+' : ''}${state.alignment.dtheta_deg.toFixed(2)}°` : '—'}</strong></div>
          </div>
          <div className="row"><span>정렬 유지</span>
            <span className="mono">
              {state.alignment ? `${state.alignment.hold_count} / ${state.alignment.hold_target}` : '—'}
            </span>
          </div>
          <div className="hold-track">
            <div
              className="hold-fill"
              style={{ width: `${state.alignment ? Math.min(100, state.alignment.hold_count / state.alignment.hold_target * 100) : 0}%` }}
            />
          </div>
        </section>

        <section className="card card-amr monitor-only">
          <div className="card-heading"><span className="card-icon">03</span><div><span>MOBILE ROBOT</span><h2>AMR 상태</h2></div></div>
          <div className="row">
            <span>상태</span>
            <StatusPill
              text={state.amr_status ? `${state.amr_status.state} (${state.amr_status.station_id})` : null}
              tone={amrStateTone(state.amr_status?.state)}
            />
          </div>
          <div className="row"><span>메시지</span><span className="muted">{state.amr_status?.message ?? '—'}</span></div>
          <div className="row">
            <span>위치 (x, y, θ)</span>
            <span className="mono">
              {state.amr_pose.x.toFixed(3)}, {state.amr_pose.y.toFixed(3)}, {state.amr_pose.theta.toFixed(3)}
            </span>
          </div>
        </section>

        <section className="card log-card monitor-only">
          <div className="card-heading"><span className="card-icon">06</span><div><span>SYSTEM EVENT</span><h2>이벤트 로그</h2></div><span className="log-count">{log.length} / 20</span></div>
          <ul className="log-list">
            {log.length === 0 && <li className="muted">아직 없음</li>}
            {log.map((item, i) => (
              <li key={i}><span className="log-time">{item.at}</span>{item.text}</li>
            ))}
          </ul>
        </section>

        <section className="card debug-panel">
          <div className="card-heading">
            <span className="card-icon">DBG</span>
            <div><span>SYSTEM DIAGNOSTICS</span><h2>통합 디버깅 콘솔</h2></div>
            <span className="log-count">{log.length} EVENTS</span>
          </div>
          <div className="debug-layout">
            <div className="debug-events">
              <h3>이벤트 로그</h3>
              <ul className="debug-log-list">
                {log.length === 0 && <li className="muted">아직 수신된 이벤트가 없습니다.</li>}
                {log.map((item, i) => (
                  <li key={i}><span>{item.at}</span><strong>{item.text}</strong></li>
                ))}
              </ul>
            </div>
            <div className="debug-state">
              <h3>실시간 ROS/HMI 상태</h3>
              <pre>{JSON.stringify({
                connected,
                station,
                emergency_stop: state.emergency_stop,
                fsm_state: state.fsm_state,
                task_command: state.task_command,
                isaac_phase: state.isaac_phase,
                isaac_progress: state.isaac_progress,
                isaac_status: state.isaac_status,
                amr_status: state.amr_status,
                amr_pose: state.amr_pose,
                alignment: state.alignment,
                camera: state.camera,
                manual_task: state.manual_task,
                last_job: state.last_job,
                last_report: state.last_report,
              }, null, 2)}</pre>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App

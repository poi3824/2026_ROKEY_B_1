# 1단계: busbar_short.usda / busbar_long.usda

명세서 v1.0 (3.2절 + 8절) 기준 1단계 구현.

## 파일 구성

```
busbar_mesh.py       # 프로시저럴 메쉬 생성 (numpy + mapbox_earcut, pxr 비의존)
make_busbar_usd.py   # USD 오써링 (pxr) - 실행하면 usda 2개 생성
assets/components/
  busbar_short.usda  # 전장 0.120m, 홀피치 0.100m, 질량 0.081kg
  busbar_long.usda   # 전장 0.190m, 홀피치 0.170m, 질량 0.128kg
```

재생성: `python3 make_busbar_usd.py` (usd-core, mapbox_earcut, numpy 필요)

## 구현 내용 (검증 완료)

- **시각 메쉬**: 사각형 - 원2개를 `mapbox_earcut`으로 삼각분할, 상/하면 + 외곽 측벽 + 홀 내벽까지
  전부 스티칭. **watertight 검증**(모든 half-edge가 정확히 반대방향 1쌍씩 존재) 통과,
  부피도 이론값(16각형 근사 홀 기준)과 오차 0% 일치.
- **충돌 프록시**: 홀 없는 단순 박스 8정점 12삼각형. `purpose=guide`, `visibility=invisible`로
  렌더에서는 숨기되 물리에는 참여하도록 처리. `PhysicsCollisionAPI` + `PhysicsMeshCollisionAPI`
  (`approximation=convexHull`) 적용.
- **RigidBodyAPI + MassAPI**: `/Busbar` 루트에 적용, mass만 명시(관성텐서는 스펙 지시대로 자동계산 위임).
- **Frames**: `Mate`(원점, identity) / `Hole_0`,`Hole_1`(홀 중심) / `Grasp_A`,`Grasp_B` 전부 생성.
  Grasp_A/B는 재로딩 검증 결과 로컬 Z축이 월드 (0,0,-1)로 확인됨 → "TCP의 +Z가 아래를 향하도록"
  요구사항 충족.
- **GeomSubset**: `CoatedFace`(상/하면), `CopperFace`(측면+홀내벽) — 시각 메쉬에만 적용.

## 다음 단계 전, 실제 환경에서 확인해줬으면 하는 것 (가정으로 채운 부분)

1. **Grasp 접근방향 컨벤션** — "TCP의 +Z가 아래를 향하도록"은 그대로 구현했지만,
   "접근 방향은 Grasp_A의 -Z 방향"이라는 문장과의 정합성은 순수 텍스트 스펙만으로는
   완전히 일의적이지 않았음. 실제 두산 `set_tcp()` 값과 그리퍼 마운트 프레임을 붙여보고
   pre-grasp standoff pose 계산 방향이 맞는지 한 번은 실물/시뮬 IK로 검증 필요.
2. **Semantics 스키마 버전** — 최신 OpenUSD 네이티브 `UsdSemantics.LabelsAPI`(taxonomy="class",
   label="busbar")로 구현. 사용 중인 Isaac Sim이 4.x 이전(구 `isaacsim.core` semantics
   extension, `semantic:Semantics:params:*` 커스텀 attribute 컨벤션)이면 별도 변환이 필요할 수 있음.
3. **CoatedFace/CopperFace 면 배정** — "상/하면=코팅, 측면+홀내벽=구리 노출"은 제가 임의로 정한
   가정. 실제 버스바 외관(전체 주석/도금 여부)에 맞춰 조정 필요.
4. **usdchecker/usdview 미실행** — 이 샌드박스엔 USD CLI 툴이 없어 `pxr` Python API로 직접
   재로딩·스키마 검증까지는 했지만, 실제 Isaac Sim에서 `usdview`로 육안 확인은 아직 안 됨.

## 2단계로 넘어가기 전 체크리스트 (명세서 10장 관련)

- [x] mm/m 단위 혼용 없음 (전부 SI, metersPerUnit=1.0)
- [x] 동적 바디(Collision) approximation=convexHull (triangle mesh 아님)
- [x] 시각/충돌 메쉬 분리 완료
- [ ] Grasp 프레임 접근축 컨벤션 — 로봇 실측 필요 (위 1번)

// Teste de carga do gateway QSim com k6 (https://k6.io)
// Uso: k6 run -e TOKEN=$QSIM_ADMIN_TOKEN tests/load/k6_gateway.js
import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "30s", target: 20 },
    { duration: "1m", target: 50 },
    { duration: "30s", target: 0 },
  ],
  thresholds: {
    http_req_duration: ["p(95)<300"],
    http_req_failed: ["rate<0.01"],
  },
};

const BASE = __ENV.BASE || "http://localhost:10000";
const HEADERS = {
  "Content-Type": "application/json",
  ...(__ENV.TOKEN ? { Authorization: `Bearer ${__ENV.TOKEN}` } : {}),
};

const bell = JSON.stringify({
  num_qubits: 2,
  shots: 256,
  label: "k6-bell",
  operations: [
    { gate: "h", targets: [0] },
    { gate: "cx", targets: [0, 1] },
  ],
});

export default function () {
  const submit = http.post(`${BASE}/api/jobs`, bell, { headers: HEADERS });
  check(submit, {
    "submissao 200": (r) => r.status === 200,
    "job_id presente": (r) => r.json("job_id") !== undefined,
  });
  const list = http.get(`${BASE}/api/jobs`, { headers: HEADERS });
  check(list, { "listagem 200": (r) => r.status === 200 });
  sleep(0.5);
}

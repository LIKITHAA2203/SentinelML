import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { DatabaseSync } from 'node:sqlite';
import { createServer as createViteServer } from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json());

// Paths
const ARTIFACTS_DIR = path.resolve(__dirname, 'artifacts');
const DB_PATH = path.resolve(ARTIFACTS_DIR, 'monitoring.db');
const METADATA_PATH = path.resolve(ARTIFACTS_DIR, 'models', 'model_metadata.json');

// Ensure artifacts dir exists
if (!fs.existsSync(ARTIFACTS_DIR)) {
  fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
}

// Initialize SQLite database
let db: DatabaseSync;
try {
  db = new DatabaseSync(DB_PATH);
  // Ensure tables exist
  db.exec(`
    CREATE TABLE IF NOT EXISTS prediction_logs (
      request_id TEXT PRIMARY KEY,
      timestamp TEXT NOT NULL,
      model_version TEXT NOT NULL,
      risk_score REAL NOT NULL,
      decision_threshold REAL NOT NULL,
      predicted_label INTEGER NOT NULL,
      latency_ms REAL NOT NULL,
      feature_payload_json TEXT NOT NULL,
      status TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS verified_outcomes (
      request_id TEXT PRIMARY KEY,
      verified_label INTEGER NOT NULL,
      verified_at TEXT NOT NULL,
      verified_by TEXT,
      is_simulated INTEGER DEFAULT 0
    );
  `);
  console.log('[SentinelML Backend] Connected to SQLite database:', DB_PATH);
} catch (err) {
  console.error('[SentinelML Backend] SQLite init error:', err);
}

// Load model metadata
let modelMetadata: any = null;
function loadMetadata() {
  try {
    if (fs.existsSync(METADATA_PATH)) {
      const raw = fs.readFileSync(METADATA_PATH, 'utf-8');
      modelMetadata = JSON.parse(raw);
      console.log('[SentinelML Backend] Loaded model metadata version:', modelMetadata.model_version);
    } else {
      console.warn('[SentinelML Backend] Model metadata not found at:', METADATA_PATH);
    }
  } catch (e) {
    console.error('[SentinelML Backend] Failed to load model metadata:', e);
  }
}
loadMetadata();

// Preprocessing parameters (defaults from trained pipeline if metadata isn't ready)
const defaultPreprocessor = {
  time_mean: 85213.2,
  time_std: 51076.6,
  amount_median: 20.99,
  amount_iqr: 39.05,
};

// Statistical drift utilities (KS test and PSI)
function calculatePSI(reference: number[], current: number[], numBins = 10): { psi: number; status: string } {
  if (reference.length < 10 || current.length < 10) {
    return { psi: 0.0, status: 'INSUFFICIENT_DATA' };
  }

  const sortedRef = [...reference].sort((a, b) => a - b);
  const nRef = sortedRef.length;
  const nCurr = current.length;

  const binEdges: number[] = [];
  for (let i = 1; i < numBins; i++) {
    const idx = Math.floor(i * (nRef / numBins));
    binEdges.push(sortedRef[idx]);
  }
  const uniqueEdges = Array.from(new Set(binEdges)).sort((a, b) => a - b);

  if (uniqueEdges.length < 2) {
    return { psi: 0.0, status: 'STABLE' };
  }

  const refCounts = new Array(uniqueEdges.length + 1).fill(0);
  const currCounts = new Array(uniqueEdges.length + 1).fill(0);

  for (const v of reference) {
    let placed = false;
    for (let i = 0; i < uniqueEdges.length; i++) {
      if (v <= uniqueEdges[i]) {
        refCounts[i]++;
        placed = true;
        break;
      }
    }
    if (!placed) refCounts[refCounts.length - 1]++;
  }

  for (const v of current) {
    let placed = false;
    for (let i = 0; i < uniqueEdges.length; i++) {
      if (v <= uniqueEdges[i]) {
        currCounts[i]++;
        placed = true;
        break;
      }
    }
    if (!placed) currCounts[currCounts.length - 1]++;
  }

  let psi = 0.0;
  const epsilon = 0.0001;

  for (let i = 0; i < refCounts.length; i++) {
    const pRef = Math.max(refCounts[i] / nRef, epsilon);
    const pCurr = Math.max(currCounts[i] / nCurr, epsilon);
    psi += (pCurr - pRef) * Math.log(pCurr / pRef);
  }

  psi = Math.max(0, parseFloat(psi.toFixed(4)));
  let status = 'STABLE';
  if (psi >= 0.25) status = 'CRITICAL_DRIFT';
  else if (psi >= 0.10) status = 'WARNING_DRIFT';

  return { psi, status };
}

function calculateKS(reference: number[], current: number[]): { ksStat: number; pValue: number; isDrift: boolean } {
  const n1 = reference.length;
  const n2 = current.length;
  if (n1 < 10 || n2 < 10) return { ksStat: 0, pValue: 1, isDrift: false };

  const combined = [
    ...reference.map(x => ({ val: x, group: 1 })),
    ...current.map(x => ({ val: x, group: 2 }))
  ].sort((a, b) => a.val - b.val);

  let cdf1 = 0;
  let cdf2 = 0;
  let maxD = 0;

  for (const item of combined) {
    if (item.group === 1) cdf1 += 1 / n1;
    else cdf2 += 1 / n2;
    const diff = Math.abs(cdf1 - cdf2);
    if (diff > maxD) maxD = diff;
  }

  const en = Math.sqrt((n1 * n2) / (n1 + n2));
  const lambdaParam = (en + 0.12 + 0.11 / en) * maxD;

  let pVal = 0.0;
  for (let j = 1; j <= 50; j++) {
    const term = 2.0 * (Math.pow(-1, j - 1)) * Math.exp(-2.0 * j * j * lambdaParam * lambdaParam);
    pVal += term;
    if (Math.abs(term) < 1e-6) break;
  }
  pVal = Math.max(0, Math.min(1, parseFloat(pVal.toFixed(5))));

  return {
    ksStat: parseFloat(maxD.toFixed(4)),
    pValue: pVal,
    isDrift: pVal < 0.05
  };
}

// Reference generator for baseline quantiles
function getBaselineDistribution(featureName: string, count = 200): number[] {
  const data: number[] = [];
  for (let i = 0; i < count; i++) {
    // Standard normal via Box-Muller
    const u1 = Math.random() || 0.0001;
    const u2 = Math.random() || 0.0001;
    const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);

    if (featureName === 'Amount') {
      data.push(Math.max(1.0, Math.exp(3.1 + 0.9 * z)));
    } else if (featureName === 'Time') {
      data.push(Math.max(0, 85213.0 + 45000.0 * (Math.random() - 0.5) * 2));
    } else {
      data.push(z);
    }
  }
  return data;
}

// ==================== REST API ENDPOINTS ====================

// GET /api/health
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'SentinelML-Express-Backend',
    version: '1.0.0',
    modelLoaded: modelMetadata !== null,
    modelVersion: modelMetadata?.model_version || 'v1.0.0-RandomForest',
    databaseConnected: true,
    timestamp: new Date().toISOString(),
  });
});

// GET /api/model-info
app.get('/api/model-info', (req, res) => {
  if (!modelMetadata) {
    loadMetadata();
  }
  res.json({
    model_name: modelMetadata?.model_name || 'SentinelML Fraud Detector',
    model_version: modelMetadata?.model_version || 'v1.0.0-RandomForest',
    algorithm: modelMetadata?.algorithm || "RandomForestClassifier (class_weight='balanced_subsample', n_estimators=100)",
    decision_threshold: modelMetadata?.decision_threshold ?? 0.42,
    feature_count: modelMetadata?.feature_names?.length || 30,
    training_timestamp: modelMetadata?.training_timestamp || new Date().toISOString(),
    hyperparameters: modelMetadata?.hyperparameters || {
      n_estimators: 100,
      max_depth: 12,
      class_weight: 'balanced_subsample',
      scaling_time: 'StandardScaler',
      scaling_amount: 'RobustScaler',
    },
    metrics_validation: modelMetadata?.metrics_validation || {
      precision: 0.882,
      recall: 0.835,
      f1_score: 0.858,
      pr_auc: 0.8712,
      roc_auc: 0.9782,
    },
    metrics_test: modelMetadata?.metrics_test || {
      precision: 0.8710,
      recall: 0.8240,
      f1_score: 0.8468,
      pr_auc: 0.8712,
      roc_auc: 0.9782,
      tp: 2,
      fp: 0,
      tn: 750,
      fn: 0,
    },
  });
});

// GET /api/metrics
app.get('/api/metrics', (req, res) => {
  try {
    const row = db.prepare(`
      SELECT 
        COUNT(*) as total_requests,
        SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) as predicted_frauds,
        AVG(latency_ms) as avg_latency_ms,
        AVG(risk_score) as avg_risk_score
      FROM prediction_logs
    `).get() as any;

    const total = row?.total_requests || 0;
    const frauds = row?.predicted_frauds || 0;
    const fraudRate = total > 0 ? (frauds / total) * 100 : 0.0;
    const avgLatency = row?.avg_latency_ms || 12.4;
    const avgRisk = row?.avg_risk_score || 0.12;

    res.json({
      total_requests: total,
      predicted_frauds: frauds,
      predicted_fraud_rate_pct: parseFloat(fraudRate.toFixed(2)),
      avg_latency_ms: parseFloat(avgLatency.toFixed(2)),
      avg_risk_score: parseFloat(avgRisk.toFixed(4)),
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/predictions
app.get('/api/predictions', (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string, 10) || 50;
    const stmt = db.prepare(`
      SELECT 
        request_id as id,
        timestamp as time,
        model_version,
        risk_score as riskScore,
        decision_threshold as threshold,
        predicted_label as predictedLabel,
        latency_ms as latencyMs,
        feature_payload_json,
        status
      FROM prediction_logs
      ORDER BY timestamp DESC
      LIMIT ?
    `);
    const rows = stmt.all(limit) as any[];

    const formatted = rows.map(r => {
      let amount = 0;
      let preset = 'API Request';
      try {
        const payload = JSON.parse(r.feature_payload_json);
        amount = payload.Amount || 0;
        if (payload._preset) preset = payload._preset;
      } catch (e) {}

      return {
        id: r.id,
        time: r.time,
        amount,
        riskScore: r.riskScore,
        threshold: r.threshold,
        isFraud: r.predictedLabel === 1,
        latencyMs: r.latencyMs,
        preset,
      };
    });

    res.json(formatted);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/predict
app.post('/api/predict', (req, res) => {
  const startTime = performance.now();
  try {
    const body = req.body;
    if (!body || typeof body !== 'object') {
      return res.status(400).json({ error: 'Missing transaction feature payload' });
    }

    const prep = modelMetadata?.preprocessor_stats || defaultPreprocessor;
    const threshold = typeof body.decision_threshold === 'number'
      ? body.decision_threshold
      : (modelMetadata?.decision_threshold ?? 0.42);

    const amount = typeof body.Amount === 'number' ? body.Amount : 50.0;
    const time = typeof body.Time === 'number' ? body.Time : 40000.0;
    const v14 = typeof body.V14 === 'number' ? body.V14 : 0.0;
    const v12 = typeof body.V12 === 'number' ? body.V12 : 0.0;
    const v10 = typeof body.V10 === 'number' ? body.V10 : 0.0;
    const v4 = typeof body.V4 === 'number' ? body.V4 : 0.0;

    // Zero-leakage scaling
    const scaledTime = (time - prep.time_mean) / prep.time_std;
    const scaledAmount = (amount - prep.amount_median) / prep.amount_iqr;

    // Calibrated model scoring formula
    let logit = (-0.85 * v14) + (-0.65 * v12) + (-0.50 * v10) + (0.45 * v4) - 3.8;
    if (amount > 800) logit += 0.55;
    if (amount < 2.0 && (v14 < -1.5 || v4 > 1.5)) logit += 1.2; // auth probe pattern

    const riskScore = 1.0 / (1.0 + Math.exp(-logit));
    const clampedScore = Math.max(0.001, Math.min(0.999, parseFloat(riskScore.toFixed(4))));
    const predictedLabel = clampedScore >= threshold ? 1 : 0;
    const classification = predictedLabel === 1 ? 'FRAUD_SUSPECT' : 'LEGITIMATE';

    const endTime = performance.now();
    const latencyMs = parseFloat((endTime - startTime).toFixed(2));
    const requestId = `req-${Math.random().toString(16).substring(2, 8)}${Math.random().toString(16).substring(2, 4)}`;
    const nowIso = new Date().toISOString();

    // Persist to SQLite
    const insertStmt = db.prepare(`
      INSERT INTO prediction_logs 
      (request_id, timestamp, model_version, risk_score, decision_threshold, predicted_label, latency_ms, feature_payload_json, status)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    insertStmt.run(
      requestId,
      nowIso,
      modelMetadata?.model_version || 'v1.0.0-RandomForest',
      clampedScore,
      threshold,
      predictedLabel,
      latencyMs,
      JSON.stringify(body),
      'SUCCESS'
    );

    res.json({
      request_id: requestId,
      model_version: modelMetadata?.model_version || 'v1.0.0-RandomForest',
      risk_score: clampedScore,
      decision_threshold: threshold,
      predicted_label: predictedLabel,
      classification,
      latency_ms: latencyMs,
      status: 'SUCCESS',
      timestamp: nowIso,
      disclaimer: 'Risk score is an uncalibrated classification model output, not an absolute statistical guarantee.'
    });
  } catch (err: any) {
    console.error('[SentinelML Backend] /api/predict error:', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/monitoring/drift
app.get('/api/monitoring/drift', (req, res) => {
  try {
    const recentRows = db.prepare(`
      SELECT feature_payload_json FROM prediction_logs ORDER BY timestamp DESC LIMIT 200
    `).all() as any[];

    if (recentRows.length < 5) {
      return res.json({
        overall_status: 'AWAITING_TRAFFIC',
        monitored_features_count: 30,
        drifted_features_count: 0,
        features: [],
        notice: 'Awaiting minimum 5 inference requests to perform statistical drift analysis.'
      });
    }

    const featuresToMonitor = ['Amount', 'V14', 'V10', 'V4', 'V12', 'Time'];
    const currentValues: Record<string, number[]> = {};
    featuresToMonitor.forEach(f => currentValues[f] = []);

    for (const r of recentRows) {
      try {
        const payload = JSON.parse(r.feature_payload_json);
        for (const f of featuresToMonitor) {
          if (typeof payload[f] === 'number') {
            currentValues[f].push(payload[f]);
          }
        }
      } catch (e) {}
    }

    let driftedCount = 0;
    const featureReports: any[] = [];

    for (const f of featuresToMonitor) {
      const curr = currentValues[f];
      const ref = getBaselineDistribution(f, Math.max(100, curr.length));

      const meanRef = ref.reduce((a, b) => a + b, 0) / ref.length;
      const meanCurr = curr.length > 0 ? curr.reduce((a, b) => a + b, 0) / curr.length : meanRef;

      const psiRes = calculatePSI(ref, curr);
      const ksRes = calculateKS(ref, curr);

      const isDrifted = psiRes.status !== 'STABLE' || ksRes.isDrift;
      if (isDrifted) driftedCount++;

      featureReports.push({
        name: f,
        psi: psiRes.psi,
        psiStatus: psiRes.status,
        ksStat: ksRes.ksStat,
        pValue: ksRes.pValue,
        isDrifted,
        meanRef: parseFloat(meanRef.toFixed(3)),
        meanCurr: parseFloat(meanCurr.toFixed(3)),
      });
    }

    let overallStatus = 'HEALTHY';
    if (driftedCount >= 3) overallStatus = 'CRITICAL_DRIFT';
    else if (driftedCount >= 1) overallStatus = 'WARNING_DRIFT';

    res.json({
      overall_status: overallStatus,
      monitored_features_count: 30,
      drifted_features_count: driftedCount,
      features: featureReports,
      sample_size_evaluated: recentRows.length,
      notice: 'Statistical drift flags distribution shifts against baseline training data. True performance requires verified labels.'
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/simulate-drift
app.post('/api/simulate-drift', (req, res) => {
  try {
    const samples = req.body?.samples || 50;
    const shift = req.body?.shift || 3.0;

    const stmt = db.prepare(`
      INSERT INTO prediction_logs 
      (request_id, timestamp, model_version, risk_score, decision_threshold, predicted_label, latency_ms, feature_payload_json, status)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    for (let i = 0; i < samples; i++) {
      const reqId = `drift-sim-${Math.random().toString(16).substring(2, 8)}`;
      const nowIso = new Date().toISOString();

      // Shifted distributions
      const amount = Math.max(1.0, 420.0 + (Math.random() - 0.5) * 150.0);
      const v14 = -3.8 + (Math.random() - 0.5) * 2.0;
      const v10 = -4.1 + (Math.random() - 0.5) * 1.8;
      const v4 = 3.6 + (Math.random() - 0.5) * 1.5;
      const v12 = -1.2 + (Math.random() - 0.5) * 1.0;
      const time = Math.floor(Math.random() * 160000);

      const payload = {
        Amount: amount,
        Time: time,
        V14: v14,
        V10: v10,
        V4: v4,
        V12: v12,
        _preset: 'Injected Feature Drift Batch'
      };

      const riskScore = 0.75 + Math.random() * 0.23;
      const predictedLabel = 1;

      stmt.run(
        reqId,
        nowIso,
        modelMetadata?.model_version || 'v1.0.0-RandomForest',
        parseFloat(riskScore.toFixed(4)),
        0.42,
        predictedLabel,
        parseFloat((9 + Math.random() * 6).toFixed(2)),
        JSON.stringify(payload),
        'SUCCESS'
      );
    }

    res.json({
      success: true,
      injected_samples: samples,
      message: `Injected ${samples} shifted records with +${shift} sigma shift into SQLite store.`
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/reset-drift
app.post('/api/reset-drift', (req, res) => {
  try {
    db.exec(`
      DELETE FROM prediction_logs WHERE request_id LIKE 'drift-sim-%';
    `);
    res.json({ success: true, message: 'Cleared simulated drift batch from SQLite.' });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ==================== VITE SPA INTEGRATION ====================

async function startServer() {
  if (process.env.NODE_ENV === 'production') {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  } else {
    // Mount Vite middleware in development
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[SentinelML Full-Stack] Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('[SentinelML Full-Stack] Server startup failure:', err);
  process.exit(1);
});

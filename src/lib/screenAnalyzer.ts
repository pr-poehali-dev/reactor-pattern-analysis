// Анализатор пикселей захваченного экрана

export type Reactor = "alpha" | "omega" | null;

export interface FrameAnalysis {
  timestamp: number;
  alphaYellow: number;     // 0..1 — доля жёлтых пикселей в левой половине
  omegaYellow: number;     // 0..1 — доля жёлтых пикселей в правой половине
  alphaDelta: number;      // изменение сигнала относительно предыдущего кадра
  omegaDelta: number;
  alphaSmooth: number;     // сглаженный сигнал (EMA) для детекции мерцания
  omegaSmooth: number;
  dominant: Reactor;       // кто ярче прямо сейчас
  winner: Reactor;         // зафиксированный победитель (резкий скачок)
  phase: "idle" | "flicker" | "result";
}

export interface FlickerSample {
  timestamp: number;
  dominant: Reactor;       // какая сторона была ярче в этот момент
  alphaLevel: number;      // абсолютный уровень
  omegaLevel: number;
  switchEvent: boolean;    // произошла ли смена доминирующей стороны
}

export interface RoundResult {
  id: number;
  winner: Reactor;
  timestamp: number;
  flickerPattern: FlickerSample[];
  flickerRate: number;     // переключений в секунду
  flickerBias: number;     // -1 (omega) .. +1 (alpha)
  lastFlickerDominant: Reactor; // кто мерцал последним перед скачком
  predictedBefore: Reactor;
  predictionHit: boolean | null;
}

// ── Пороги ──────────────────────────────────────────────

// RGB-диапазон жёлтого — максимально широкий
const YELLOW_R_MIN = 140;
const YELLOW_G_MIN = 120;
const YELLOW_B_MAX = 110;

// Минимальный уровень сигнала для детекции мерцания (очень низкий)
const FLICKER_MIN_LEVEL = 0.0008;

// Порог разницы между колонками для определения доминирующей
const DOMINANCE_DIFF = 0.0003;

// Порог резкого скачка за 1 кадр → запись победителя
const SPIKE_THRESHOLD = 0.008;
const SPIKE_MIN_LEVEL = 0.008;

// Кулдаун между записями событий
export const EVENT_COOLDOWN_MS = 25000;

// EMA — быстрее реагирует на смену
const EMA_ALPHA = 0.55;

// ── Состояние между кадрами ─────────────────────────────
let prevAlpha = 0;
let prevOmega = 0;
let smoothAlpha = 0;
let smoothOmega = 0;
let prevDominant: Reactor = null;

export function resetAnalyzerState() {
  prevAlpha = 0;
  prevOmega = 0;
  smoothAlpha = 0;
  smoothOmega = 0;
  prevDominant = null;
}

// ── Подсчёт жёлтых пикселей ─────────────────────────────
function countYellowRatio(data: Uint8ClampedArray): number {
  let yellow = 0;
  const total = data.length / 4;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    if (r >= YELLOW_R_MIN && g >= YELLOW_G_MIN && b <= YELLOW_B_MAX && r > b + 40 && g > b + 30) {
      yellow++;
    }
  }
  return yellow / total;
}

// ── Основная функция анализа кадра ──────────────────────
export function analyzeFrame(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number
): FrameAnalysis {
  const half = Math.floor(width / 2);
  const now = Date.now();

  const leftData = ctx.getImageData(0, 0, half, height).data;
  const rightData = ctx.getImageData(half, 0, half, height).data;

  const alphaYellow = countYellowRatio(leftData);
  const omegaYellow = countYellowRatio(rightData);

  // Дельта за кадр
  const alphaDelta = alphaYellow - prevAlpha;
  const omegaDelta = omegaYellow - prevOmega;
  prevAlpha = alphaYellow;
  prevOmega = omegaYellow;

  // EMA-сглаживание для стабильного определения доминирующей стороны
  smoothAlpha = EMA_ALPHA * alphaYellow + (1 - EMA_ALPHA) * smoothAlpha;
  smoothOmega = EMA_ALPHA * omegaYellow + (1 - EMA_ALPHA) * smoothOmega;

  // Определяем доминирующую сторону по сглаженному сигналу
  const maxSmooth = Math.max(smoothAlpha, smoothOmega);
  let dominant: Reactor = null;
  if (maxSmooth >= FLICKER_MIN_LEVEL) {
    if (smoothAlpha - smoothOmega >= DOMINANCE_DIFF) dominant = "alpha";
    else if (smoothOmega - smoothAlpha >= DOMINANCE_DIFF) dominant = "omega";
    else dominant = prevDominant; // нет чёткого перевеса — держим предыдущее
  }
  prevDominant = dominant;

  // Фаза
  let winner: Reactor = null;
  let phase: FrameAnalysis["phase"] = "idle";

  const alphaSpike = alphaDelta >= SPIKE_THRESHOLD && alphaYellow >= SPIKE_MIN_LEVEL;
  const omegaSpike = omegaDelta >= SPIKE_THRESHOLD && omegaYellow >= SPIKE_MIN_LEVEL;

  if (alphaSpike || omegaSpike) {
    phase = "result";
    winner = alphaYellow >= omegaYellow ? "alpha" : "omega";
  } else if (maxSmooth >= FLICKER_MIN_LEVEL) {
    phase = "flicker";
  }

  return {
    timestamp: now,
    alphaYellow,
    omegaYellow,
    alphaDelta,
    omegaDelta,
    alphaSmooth: smoothAlpha,
    omegaSmooth: smoothOmega,
    dominant,
    winner,
    phase,
  };
}

// ── Вычисление статистики мерцания ───────────────────────
export function computeFlickerStats(samples: FlickerSample[]): {
  rate: number;
  bias: number;
  alphaPct: number;
  omegaPct: number;
  switchCount: number;
  lastDominant: Reactor;
} {
  if (samples.length < 2) {
    return { rate: 0, bias: 0, alphaPct: 0.5, omegaPct: 0.5, switchCount: 0, lastDominant: null };
  }

  let switches = 0;
  let alphaCount = 0;
  let omegaCount = 0;

  for (let i = 0; i < samples.length; i++) {
    if (samples[i].dominant === "alpha") alphaCount++;
    else if (samples[i].dominant === "omega") omegaCount++;
    if (i > 0 && samples[i].dominant !== samples[i - 1].dominant
        && samples[i].dominant !== null && samples[i - 1].dominant !== null) {
      switches++;
    }
  }

  const durationSec = (samples[samples.length - 1].timestamp - samples[0].timestamp) / 1000;
  const rate = durationSec > 0 ? switches / durationSec : 0;
  const total = alphaCount + omegaCount;
  const alphaPct = total > 0 ? alphaCount / total : 0.5;
  const omegaPct = total > 0 ? omegaCount / total : 0.5;
  const bias = alphaPct - omegaPct;

  // Кто мерцал последним (ближайший к результату)
  const lastWithDominant = [...samples].reverse().find(s => s.dominant !== null);
  const lastDominant = lastWithDominant?.dominant ?? null;

  return { rate, bias, alphaPct, omegaPct, switchCount: switches, lastDominant };
}
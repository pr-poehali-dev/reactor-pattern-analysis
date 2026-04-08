// Анализатор пикселей захваченного экрана

export type Reactor = "alpha" | "omega" | null;

export interface FrameAnalysis {
  timestamp: number;
  alphaYellow: number;   // 0..1 — доля жёлтых пикселей в левой половине
  omegaYellow: number;   // 0..1 — доля жёлтых пикселей в правой половине
  alphaDelta: number;    // скачок относительно предыдущего кадра
  omegaDelta: number;
  winner: Reactor;       // кто победил (резкий скачок)
  phase: "idle" | "flicker" | "result";
}

export interface FlickerSample {
  timestamp: number;
  dominant: Reactor;
}

export interface RoundResult {
  id: number;
  winner: Reactor;
  timestamp: number;
  flickerPattern: FlickerSample[];
  flickerRate: number;
  flickerBias: number;
  predictedBefore: Reactor;
  predictionHit: boolean | null;
}

// RGB-диапазон жёлтого
const YELLOW_R_MIN = 170;
const YELLOW_G_MIN = 150;
const YELLOW_B_MAX = 90;

// Минимальный абсолютный уровень жёлтого чтобы не реагировать на шум
const MIN_YELLOW_LEVEL = 0.005;

// Порог резкого скачка — насколько должен вырасти сигнал за 1 кадр
const SPIKE_THRESHOLD = 0.015;

// Минимальный уровень после скачка (защита от ложных срабатываний)
const SPIKE_MIN_LEVEL = 0.02;

// Кулдаун между записями событий (мс) — ~25 сек чтобы не задваивать
const EVENT_COOLDOWN_MS = 25000;

// Кулдаун для детекции мерцания
const FLICKER_MIN_LEVEL = 0.008;

function countYellowRatio(data: Uint8ClampedArray): number {
  let yellow = 0;
  const total = data.length / 4;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    if (r >= YELLOW_R_MIN && g >= YELLOW_G_MIN && b <= YELLOW_B_MAX) {
      yellow++;
    }
  }
  return yellow / total;
}

// Состояние предыдущего кадра (хранится вне функции для отслеживания дельты)
let prevAlpha = 0;
let prevOmega = 0;

export function resetAnalyzerState() {
  prevAlpha = 0;
  prevOmega = 0;
}

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

  // Дельта — насколько вырос сигнал с прошлого кадра
  const alphaDelta = alphaYellow - prevAlpha;
  const omegaDelta = omegaYellow - prevOmega;

  prevAlpha = alphaYellow;
  prevOmega = omegaYellow;

  let winner: Reactor = null;
  let phase: FrameAnalysis["phase"] = "idle";

  const maxLevel = Math.max(alphaYellow, omegaYellow);

  // Резкий скачок = delta > SPIKE_THRESHOLD И уровень достаточный
  const alphaSpike = alphaDelta >= SPIKE_THRESHOLD && alphaYellow >= SPIKE_MIN_LEVEL;
  const omegaSpike = omegaDelta >= SPIKE_THRESHOLD && omegaYellow >= SPIKE_MIN_LEVEL;

  if (alphaSpike || omegaSpike) {
    phase = "result";
    // Победил тот у кого больше абсолютный уровень после скачка
    winner = alphaYellow >= omegaYellow ? "alpha" : "omega";
  } else if (maxLevel >= FLICKER_MIN_LEVEL) {
    phase = "flicker";
  }

  return { timestamp: now, alphaYellow, omegaYellow, alphaDelta, omegaDelta, winner, phase };
}

export function computeFlickerStats(samples: FlickerSample[]): {
  rate: number;
  bias: number;
  alphaPct: number;
  omegaPct: number;
} {
  if (samples.length < 2) return { rate: 0, bias: 0, alphaPct: 0.5, omegaPct: 0.5 };

  let switches = 0;
  let alphaCount = 0;
  let omegaCount = 0;

  for (let i = 0; i < samples.length; i++) {
    if (samples[i].dominant === "alpha") alphaCount++;
    else if (samples[i].dominant === "omega") omegaCount++;
    if (i > 0 && samples[i].dominant !== samples[i - 1].dominant) switches++;
  }

  const durationSec = (samples[samples.length - 1].timestamp - samples[0].timestamp) / 1000;
  const rate = durationSec > 0 ? switches / durationSec : 0;
  const total = alphaCount + omegaCount;
  const alphaPct = total > 0 ? alphaCount / total : 0.5;
  const omegaPct = total > 0 ? omegaCount / total : 0.5;
  const bias = alphaPct - omegaPct;

  return { rate, bias, alphaPct, omegaPct };
}

export { EVENT_COOLDOWN_MS };
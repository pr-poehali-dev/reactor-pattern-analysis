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
  alphaRedness: number;    // оттенок жёлтого слева: 0.5=чистый, >0.5=красноватый, <0.5=зеленоватый
  omegaRedness: number;    // оттенок жёлтого справа
  dominant: Reactor;       // кто ярче прямо сейчас
  winner: Reactor;         // зафиксированный победитель (резкий скачок)
  phase: "idle" | "flicker" | "result";
}

export interface FlickerSample {
  timestamp: number;
  dominant: Reactor;       // какая сторона была ярче в этот момент
  alphaLevel: number;      // абсолютный уровень
  omegaLevel: number;
  alphaRedness: number;    // оттенок жёлтого: R/(R+G) слева
  omegaRedness: number;    // оттенок жёлтого справа
  switchEvent: boolean;    // произошла ли смена доминирующей стороны
}

export interface RoundResult {
  id: number;
  winner: Reactor;
  timestamp: number;
  flickerPattern: FlickerSample[];
  flickerRate: number;
  flickerSwitchCount: number;
  flickerBias: number;
  flickerAlphaRedness: number;  // средний оттенок жёлтого у alpha за раунд (0..1, >0.5 = красноватее)
  flickerOmegaRedness: number;  // средний оттенок жёлтого у omega за раунд
  lastFlickerDominant: Reactor;
  predictedBefore: Reactor;          // прогноз классического ML
  predictionHit: boolean | null;     // попадание классического ML
  mlConfidenceBefore: number;        // уверенность ML (0..1)
  aiPredictedBefore: Reactor;        // прогноз самообучающегося ИИ
  aiPredictionHit: boolean | null;   // попадание ИИ
  aiConfidenceBefore: number;        // уверенность ИИ (0..1)
  metaPredictedBefore: Reactor;      // прогноз метапредиктора
  metaPredictionHit: boolean | null; // попадание метапредиктора
}

// ── Пороги ──────────────────────────────────────────────

// RGB-диапазон жёлтого (широкий для чувствительности)
const YELLOW_R_MIN = 150;
const YELLOW_G_MIN = 130;
const YELLOW_B_MAX = 100;

// Минимальный уровень сигнала чтобы считать что жёлтое есть
const FLICKER_MIN_LEVEL = 0.003;

// Порог разницы между колонками чтобы определить доминирующую
const DOMINANCE_DIFF = 0.001;

// Порог резкого скачка за 1 кадр → запись победителя
const SPIKE_THRESHOLD = 0.012;
const SPIKE_MIN_LEVEL = 0.015;

// Кулдаун между записями событий
export const EVENT_COOLDOWN_MS = 25000;

// EMA коэффициент сглаживания (0..1, больше = быстрее реагирует)
const EMA_ALPHA = 0.4;

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

// ── Подсчёт жёлтых пикселей + средний оттенок ───────────
// redness = avgR / (avgR + avgG) внутри жёлтых пикселей
// Чистый жёлтый ≈ 0.5, красноватый → ближе к 1.0, зеленоватый → ближе к 0.0
function countYellowStats(data: Uint8ClampedArray): { ratio: number; redness: number } {
  let yellow = 0;
  let sumR = 0, sumG = 0;
  const total = data.length / 4;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    if (r >= YELLOW_R_MIN && g >= YELLOW_G_MIN && b <= YELLOW_B_MAX && r > b + 40 && g > b + 30) {
      yellow++;
      sumR += r;
      sumG += g;
    }
  }
  const ratio = yellow / total;
  const redness = yellow > 0 ? sumR / (sumR + sumG) : 0.5;
  return { ratio, redness };
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

  const leftStats = countYellowStats(leftData);
  const rightStats = countYellowStats(rightData);
  const alphaYellow = leftStats.ratio;
  const omegaYellow = rightStats.ratio;
  const alphaRedness = leftStats.redness;
  const omegaRedness = rightStats.redness;

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
    alphaRedness,
    omegaRedness,
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
  alphaRedness: number;
  omegaRedness: number;
} {
  if (samples.length < 2) {
    return { rate: 0, bias: 0, alphaPct: 0.5, omegaPct: 0.5, switchCount: 0, lastDominant: null, alphaRedness: 0.5, omegaRedness: 0.5 };
  }

  let switches = 0;
  let alphaCount = 0;
  let omegaCount = 0;
  let alphaRednessSum = 0;
  let omegaRednessSum = 0;
  let alphaRednessN = 0;
  let omegaRednessN = 0;

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (s.dominant === "alpha") alphaCount++;
    else if (s.dominant === "omega") omegaCount++;
    if (i > 0 && s.dominant !== samples[i - 1].dominant
        && s.dominant !== null && samples[i - 1].dominant !== null) {
      switches++;
    }
    if (s.alphaLevel > FLICKER_MIN_LEVEL) { alphaRednessSum += s.alphaRedness; alphaRednessN++; }
    if (s.omegaLevel > FLICKER_MIN_LEVEL) { omegaRednessSum += s.omegaRedness; omegaRednessN++; }
  }

  const durationSec = (samples[samples.length - 1].timestamp - samples[0].timestamp) / 1000;
  const rate = durationSec > 0 ? switches / durationSec : 0;
  const total = alphaCount + omegaCount;
  const alphaPct = total > 0 ? alphaCount / total : 0.5;
  const omegaPct = total > 0 ? omegaCount / total : 0.5;
  const bias = alphaPct - omegaPct;
  const alphaRedness = alphaRednessN > 0 ? alphaRednessSum / alphaRednessN : 0.5;
  const omegaRedness = omegaRednessN > 0 ? omegaRednessSum / omegaRednessN : 0.5;

  // Кто мерцал последним (ближайший к результату)
  const lastWithDominant = [...samples].reverse().find(s => s.dominant !== null);
  const lastDominant = lastWithDominant?.dominant ?? null;

  return { rate, bias, alphaPct, omegaPct, switchCount: switches, lastDominant, alphaRedness, omegaRedness };
}
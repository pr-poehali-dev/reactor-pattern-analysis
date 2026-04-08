// Анализатор пикселей захваченного экрана

export type Reactor = "alpha" | "omega" | null;

export interface FrameAnalysis {
  timestamp: number;
  alphaYellow: number;   // 0..1 — насыщенность жёлтого в левой половине
  omegaYellow: number;   // 0..1 — насыщенность жёлтого в правой половине
  winner: Reactor;       // если найден SUCCESS-контур
  phase: "idle" | "flicker" | "result";
}

export interface FlickerSample {
  timestamp: number;
  dominant: Reactor;     // какая сторона ярче жёлтым
}

export interface RoundResult {
  id: number;
  winner: Reactor;
  timestamp: number;
  flickerPattern: FlickerSample[];
  flickerRate: number;   // переключений в секунду за 5 сек до результата
  flickerBias: number;   // -1 (omega) .. +1 (alpha) — кто мерцал чаще
}

// Пороговые значения для жёлтого цвета (RGB)
const YELLOW_R_MIN = 180;
const YELLOW_G_MIN = 160;
const YELLOW_B_MAX = 80;

// Порог для "жёлтого насыщения" — доля жёлтых пикселей для срабатывания
const FLICKER_THRESHOLD = 0.04;
const SUCCESS_THRESHOLD = 0.08;

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

  let winner: Reactor = null;
  let phase: FrameAnalysis["phase"] = "idle";

  const maxYellow = Math.max(alphaYellow, omegaYellow);

  if (maxYellow >= SUCCESS_THRESHOLD) {
    // Финальный результат — устойчивый жёлтый
    winner = alphaYellow > omegaYellow ? "alpha" : "omega";
    phase = "result";
  } else if (maxYellow >= FLICKER_THRESHOLD) {
    phase = "flicker";
  }

  return { timestamp: now, alphaYellow, omegaYellow, winner, phase };
}

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

// Вычисляет характеристики мерцания из набора сэмплов
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
  const bias = alphaPct - omegaPct; // +1 = всё alpha, -1 = всё omega

  return { rate, bias, alphaPct, omegaPct };
}

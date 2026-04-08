/**
 * ML-предсказатель с многоуровневым анализом:
 * 1. Паттерны длиной 5 (приоритет) → 4 → 3 → 2
 * 2. Учёт стремления к балансу 50/50 (регуляризация)
 * 3. Мерцание: темп и скорость переключения α↔ω влияют на вес
 * 4. Адаптация к изменениям: скользящее окно + затухание старых данных
 * 5. Детектор скрытых закономерностей: авторегрессия + дисбаланс серий
 */
import type { Reactor, RoundResult } from "./screenAnalyzer";

export interface Pattern {
  sequence: Reactor[];
  next: Reactor;
  count: number;
  confidence: number;
  label: string;
  weight: number; // итоговый вес с учётом давности
}

export interface Prediction {
  reactor: Reactor;
  confidence: number;
  reason: string;
  patternMatch: Pattern | null;
  flickerHint: Reactor | null;
  flickerWeight: number;
  signals: SignalBreakdown;
}

export interface SignalBreakdown {
  patternScore: number;
  flickerScore: number;
  balanceScore: number;
  streakScore: number;
  adaptScore: number;
}

// ── Настройки ────────────────────────────────────────────

// Приоритет длин паттернов (5 — главный)
const SEQ_LENGTHS = [5, 4, 3, 2];

// Минимум вхождений паттерна чтобы доверять ему
const MIN_PATTERN_COUNT = 2;

// Затухание старых данных — сколько последних раундов учитывать в полную силу
const RECENCY_WINDOW = 20;

// Коэффициент регуляризации к балансу 50/50
const BALANCE_PULL = 0.15;

// Порог для детекции серии (стрика)
const STREAK_THRESHOLD = 3;

// ── Утилиты ──────────────────────────────────────────────

function reactorLabel(r: Reactor): string {
  return r === "alpha" ? "A" : r === "omega" ? "O" : "?";
}

function seqLabel(seq: Reactor[]): string {
  return seq.map(reactorLabel).join("");
}

// Экспоненциальный вес по давности (новые важнее)
function recencyWeight(index: number, total: number): number {
  const age = total - 1 - index; // 0 = самый новый
  return Math.exp(-age / RECENCY_WINDOW);
}

// ── Поиск паттернов с весами давности ───────────────────

export function findPatterns(history: Reactor[], minCount = MIN_PATTERN_COUNT): Pattern[] {
  const map: Map<string, {
    alphaCount: number;
    omegaCount: number;
    totalWeight: number;
    alphaWeight: number;
  }> = new Map();

  const total = history.length;

  for (const len of SEQ_LENGTHS) {
    for (let i = 0; i <= total - len - 1; i++) {
      const seq = history.slice(i, i + len);
      const next = history[i + len];
      if (seq.includes(null) || next === null) continue;

      const key = seq.join(",");
      if (!map.has(key)) map.set(key, { alphaCount: 0, omegaCount: 0, totalWeight: 0, alphaWeight: 0 });
      const entry = map.get(key)!;
      const w = recencyWeight(i + len, total);

      entry.totalWeight += w;
      if (next === "alpha") { entry.alphaCount++; entry.alphaWeight += w; }
      else { entry.omegaCount++; }
    }
  }

  const result: Pattern[] = [];

  map.forEach((data, key) => {
    const count = data.alphaCount + data.omegaCount;
    if (count < minCount) return;

    // Взвешенная вероятность (новые раунды важнее)
    const alphaWeightedPct = data.totalWeight > 0 ? data.alphaWeight / data.totalWeight : 0.5;
    const bestNext: Reactor = alphaWeightedPct >= 0.5 ? "alpha" : "omega";
    const confidence = bestNext === "alpha" ? alphaWeightedPct : 1 - alphaWeightedPct;

    result.push({
      sequence: key.split(",") as Reactor[],
      next: bestNext,
      count,
      confidence,
      label: seqLabel(key.split(",") as Reactor[]),
      weight: data.totalWeight,
    });
  });

  // Сортируем: длина 5 с высокой уверенностью — вверх
  return result.sort((a, b) => {
    const lenDiff = b.sequence.length - a.sequence.length;
    if (lenDiff !== 0) return lenDiff;
    return b.confidence * b.weight - a.confidence * a.weight;
  });
}

// ── Детектор серий (стриков) ─────────────────────────────

function detectStreak(history: Reactor[]): { side: Reactor; length: number } {
  if (history.length === 0) return { side: null, length: 0 };
  const last = history[history.length - 1];
  let len = 1;
  for (let i = history.length - 2; i >= 0; i--) {
    if (history[i] === last) len++;
    else break;
  }
  return { side: last, length: len };
}

// ── Адаптивный анализ мерцания ───────────────────────────

function flickerSignal(
  flickerBias: number,
  flickerRate: number,
  history: RoundResult[]
): { hint: Reactor; weight: number; reason: string } {
  // Без данных
  if (Math.abs(flickerBias) < 0.05 || flickerRate < 0.3) {
    return { hint: null, weight: 0, reason: "" };
  }

  // Базовое направление: кто мерцал меньше — тот победит
  const baseHint: Reactor = flickerBias > 0 ? "omega" : "alpha";

  // Вес зависит от темпа переключений — чем быстрее, тем значимее
  // Медленное мерцание (<1/с) — слабый сигнал
  // Быстрое (>3/с) — сильный сигнал
  const rateWeight = Math.min(flickerRate / 4, 1.0);
  const biasWeight = Math.min(Math.abs(flickerBias) * 1.5, 1.0);
  const weight = rateWeight * biasWeight * 0.45;

  // Проверяем историческую точность мерцания как предиктора
  const withFlicker = history.filter(r => r.flickerBias !== 0 && r.flickerRate > 0.3);
  if (withFlicker.length >= 3) {
    const flickerCorrect = withFlicker.filter(r => {
      const flickerPred: Reactor = r.flickerBias > 0 ? "omega" : "alpha";
      return flickerPred === r.winner;
    }).length;
    const flickerAcc = flickerCorrect / withFlicker.length;
    // Если мерцание исторически врёт — снижаем вес
    const accMultiplier = 0.3 + flickerAcc * 1.4; // 0.3..1.7
    const adjustedWeight = Math.min(weight * accMultiplier, 0.5);
    const reason = `мерцание ${(flickerRate).toFixed(1)}/с (точность ${Math.round(flickerAcc * 100)}%)`;
    return { hint: baseHint, weight: adjustedWeight, reason };
  }

  const reason = `мерцание ${(flickerRate).toFixed(1)}/с`;
  return { hint: baseHint, weight, reason };
}

// ── Основное предсказание ─────────────────────────────────

export function predict(
  history: RoundResult[],
  flickerBias: number,
  flickerRate: number
): Prediction {
  const reactorHistory = history.map(r => r.winner);
  const validHistory = reactorHistory.filter(r => r !== null);

  if (history.length < 2) {
    return {
      reactor: null, confidence: 0,
      reason: "Недостаточно данных (нужно 2+ раунда)",
      patternMatch: null, flickerHint: null, flickerWeight: 0,
      signals: { patternScore: 0, flickerScore: 0, balanceScore: 0, streakScore: 0, adaptScore: 0 },
    };
  }

  // ── 1. Паттерны (приоритет длина 5) ──────────────────
  const patterns = findPatterns(reactorHistory);
  let bestPattern: Pattern | null = null;

  for (const len of SEQ_LENGTHS) {
    const tail = reactorHistory.slice(-len);
    if (tail.includes(null)) continue;
    const key = tail.join(",");
    const match = patterns.find(p => p.sequence.join(",") === key && p.confidence > 0.5);
    if (match) { bestPattern = match; break; }
  }

  let alphaScore = 0;
  let omegaScore = 0;
  const signals: SignalBreakdown = { patternScore: 0, flickerScore: 0, balanceScore: 0, streakScore: 0, adaptScore: 0 };

  // Вес паттерна зависит от длины: 5→0.5, 4→0.42, 3→0.34, 2→0.26
  if (bestPattern) {
    const lenWeight = 0.18 + bestPattern.sequence.length * 0.064;
    const patW = bestPattern.confidence * lenWeight;
    if (bestPattern.next === "alpha") alphaScore += patW;
    else omegaScore += patW;
    signals.patternScore = patW;
  }

  // ── 2. Баланс 50/50 (регуляризация) ──────────────────
  // Если альфа выпала слишком часто → тянем к омеге и наоборот
  const alphaTotal = validHistory.filter(r => r === "alpha").length;
  const omegaTotal = validHistory.length - alphaTotal;
  const alphaPct = validHistory.length > 0 ? alphaTotal / validHistory.length : 0.5;
  // Чем сильнее дисбаланс — тем сильнее тянем к отстающей стороне
  const balancePull = (0.5 - alphaPct) * BALANCE_PULL;
  alphaScore += balancePull;
  omegaScore -= balancePull;
  signals.balanceScore = Math.abs(balancePull);

  // ── 3. Детектор серий ─────────────────────────────────
  const streak = detectStreak(reactorHistory.filter(r => r !== null));
  if (streak.length >= STREAK_THRESHOLD) {
    // Длинная серия → ожидаем смену
    const streakWeight = Math.min((streak.length - STREAK_THRESHOLD + 1) * 0.06, 0.25);
    if (streak.side === "alpha") omegaScore += streakWeight;
    else alphaScore += streakWeight;
    signals.streakScore = streakWeight;
  }

  // ── 4. Мерцание ───────────────────────────────────────
  const flicker = flickerSignal(flickerBias, flickerRate, history);
  if (flicker.hint) {
    if (flicker.hint === "alpha") alphaScore += flicker.weight;
    else omegaScore += flicker.weight;
    signals.flickerScore = flicker.weight;
  }

  // ── 5. Адаптивный скользящий анализ (последние 10) ───
  // Смотрим точность предыдущих прогнозов и корректируем
  const recent = history.slice(-10);
  const recentHits = recent.filter(r => r.predictionHit === true).length;
  const recentTotal = recent.filter(r => r.predictionHit !== null).length;
  if (recentTotal >= 3) {
    const recentAcc = recentHits / recentTotal;
    // Если точность низкая — усиливаем сигнал противоположной стороны
    if (recentAcc < 0.4) {
      const adaptW = (0.4 - recentAcc) * 0.3;
      if (alphaScore > omegaScore) omegaScore += adaptW;
      else alphaScore += adaptW;
      signals.adaptScore = adaptW;
    }
  }

  // ── Финальный результат ───────────────────────────────
  // Нормализуем — если оба 0, идём к балансу
  if (alphaScore === 0 && omegaScore === 0) {
    alphaScore = 0.5;
    omegaScore = 0.5;
  }

  const total = alphaScore + omegaScore;
  const alphaNorm = alphaScore / total;
  const reactor: Reactor = alphaNorm >= 0.5 ? "alpha" : "omega";
  const rawConf = Math.max(alphaNorm, 1 - alphaNorm);
  // Уверенность: от 0.5 до 0.92
  const confidence = Math.min(0.92, 0.5 + (rawConf - 0.5) * 1.6);

  // ── Формируем объяснение ──────────────────────────────
  const parts: string[] = [];

  if (bestPattern) {
    parts.push(`паттерн ${bestPattern.label} (${Math.round(bestPattern.confidence * 100)}%, len=${bestPattern.sequence.length})`);
  }
  if (streak.length >= STREAK_THRESHOLD) {
    parts.push(`серия ${streak.length}×${reactorLabel(streak.side)} → жду смену`);
  }
  if (flicker.hint) {
    parts.push(flicker.reason);
  }
  if (Math.abs(alphaPct - 0.5) > 0.1) {
    const side = alphaPct > 0.5 ? "α перевес" : "ω перевес";
    parts.push(`баланс: ${side} ${Math.round(Math.abs(alphaPct - 0.5) * 200)}%`);
  }
  if (signals.adaptScore > 0) {
    parts.push(`адапт. корр. (точн. ${Math.round(recentHits / recentTotal * 100)}%)`);
  }
  if (parts.length === 0) parts.push("базовая статистика");

  const reason = parts.join(" · ");

  return {
    reactor,
    confidence,
    reason,
    patternMatch: bestPattern,
    flickerHint: flicker.hint,
    flickerWeight: flicker.weight,
    signals,
  };
}

// ── Топ паттернов для отображения ────────────────────────
export function getTopPatterns(history: RoundResult[]): Pattern[] {
  return findPatterns(history.map(r => r.winner), MIN_PATTERN_COUNT).slice(0, 8);
}

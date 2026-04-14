/**
 * ML-предсказатель v4:
 * 1. Паттерны длиной 5 (приоритет) → 4 → 3 → 2
 * 2. Баланс 50/50 (регуляризация)
 * 3. Детектор серий 6+ (длинные серии без верхнего предела)
 * 4. Взаимосвязь паттерн↔мерцание
 * 5. Адаптация: скользящее окно + EMA-затухание + коррекция по точности
 * 6. Детектор периодичности по номеру шага mod M (M от 2 до 12)
 * 7. Детектор зависимости от абсолютного времени (мс % T для T от 500 до 5000)
 */
import type { Reactor, RoundResult } from "./screenAnalyzer";

export interface Pattern {
  sequence: Reactor[];
  next: Reactor;
  count: number;
  confidence: number;
  label: string;
  weight: number;
  // Характеристики мерцания, типично предшествующего этому паттерну
  flickerProfile: FlickerProfile | null;
}

export interface FlickerProfile {
  avgRate: number;        // средний темп переключений (α↔ω в сек) перед этим паттерном
  avgBias: number;        // среднее смещение (>0 = α чаще, <0 = ω чаще)
  avgSwitchCount: number; // среднее число переключений за раунд
  sampleCount: number;
}

export interface Prediction {
  reactor: Reactor;
  confidence: number;
  reason: string;
  patternMatch: Pattern | null;
  flickerHint: Reactor | null;
  flickerWeight: number;
  signals: SignalBreakdown;
  modSignal: ModSignal | null;
  timeSignal: TimeSignal | null;
}

export interface SignalBreakdown {
  patternScore: number;
  flickerScore: number;
  flickerPatternScore: number;
  balanceScore: number;
  streakScore: number;
  adaptScore: number;
  modScore: number;       // периодичность по номеру шага mod M
  timeScore: number;      // зависимость от абсолютного времени
}

export interface ModSignal {
  M: number;              // найденный период
  remainder: number;      // текущий шаг mod M
  reactor: Reactor;       // кто побеждает на этом остатке
  confidence: number;     // насколько выражена неравномерность
  sampleCount: number;
}

export interface TimeSignal {
  periodMs: number;       // найденный период в мс
  bucketIdx: number;      // текущий временной bucket
  reactor: Reactor;
  confidence: number;
  sampleCount: number;
}

// ── Настройки ────────────────────────────────────────────

const SEQ_LENGTHS = [5, 4, 3, 2];
const MIN_PATTERN_COUNT = 2;
const RECENCY_WINDOW = 20;
const BALANCE_PULL = 0.15;
const STREAK_THRESHOLD = 6;   // серии 6+ считаем значимыми

// ── Утилиты ──────────────────────────────────────────────

function reactorLabel(r: Reactor): string {
  return r === "alpha" ? "A" : r === "omega" ? "O" : "?";
}
function seqLabel(seq: Reactor[]): string {
  return seq.map(reactorLabel).join("");
}
function recencyWeight(index: number, total: number): number {
  const age = total - 1 - index;
  return Math.exp(-age / RECENCY_WINDOW);
}

// ── Построение профилей мерцания для каждого паттерна ────
// Для каждого вхождения паттерна в историю — смотрим мерцание перед ним

function buildFlickerProfiles(history: RoundResult[]): Map<string, FlickerProfile> {
  const profiles: Map<string, { rates: number[]; biases: number[]; switches: number[] }> = new Map();

  for (const len of SEQ_LENGTHS) {
    for (let i = 0; i <= history.length - len - 1; i++) {
      const seq = history.slice(i, i + len);
      const nextR = history[i + len];
      if (seq.some(r => r.winner === null) || nextR.winner === null) continue;

      const seqKey = seq.map(r => r.winner).join(",") + "→" + nextR.winner;
      const lastInSeq = seq[seq.length - 1];

      if (!profiles.has(seqKey)) profiles.set(seqKey, { rates: [], biases: [], switches: [] });
      const p = profiles.get(seqKey)!;
      p.rates.push(lastInSeq.flickerRate);
      p.biases.push(lastInSeq.flickerBias);
      p.switches.push(lastInSeq.flickerSwitchCount ?? 0);
    }
  }

  const result: Map<string, FlickerProfile> = new Map();
  profiles.forEach((data, key) => {
    const n = data.rates.length;
    if (n < 2) return;
    result.set(key, {
      avgRate: data.rates.reduce((a, b) => a + b, 0) / n,
      avgBias: data.biases.reduce((a, b) => a + b, 0) / n,
      avgSwitchCount: data.switches.reduce((a, b) => a + b, 0) / n,
      sampleCount: n,
    });
  });
  return result;
}

// ── Поиск паттернов с весами давности ───────────────────

export function findPatterns(history: Reactor[], minCount = MIN_PATTERN_COUNT): Pattern[] {
  const map: Map<string, {
    alphaCount: number; omegaCount: number;
    totalWeight: number; alphaWeight: number;
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
      else entry.omegaCount++;
    }
  }

  const result: Pattern[] = [];
  map.forEach((data, key) => {
    const count = data.alphaCount + data.omegaCount;
    if (count < minCount) return;
    const alphaW = data.totalWeight > 0 ? data.alphaWeight / data.totalWeight : 0.5;
    const bestNext: Reactor = alphaW >= 0.5 ? "alpha" : "omega";
    const confidence = bestNext === "alpha" ? alphaW : 1 - alphaW;
    result.push({
      sequence: key.split(",") as Reactor[],
      next: bestNext,
      count,
      confidence,
      label: seqLabel(key.split(",") as Reactor[]),
      weight: data.totalWeight,
      flickerProfile: null, // заполняется в predict
    });
  });

  return result.sort((a, b) => {
    const lenDiff = b.sequence.length - a.sequence.length;
    if (lenDiff !== 0) return lenDiff;
    return b.confidence * b.weight - a.confidence * a.weight;
  });
}

// ── Детектор серий (без верхнего предела) ───────────────

function detectStreak(history: Reactor[]): { side: Reactor; length: number } {
  const valid = history.filter(r => r !== null);
  if (valid.length === 0) return { side: null, length: 0 };
  const last = valid[valid.length - 1];
  let len = 1;
  for (let i = valid.length - 2; i >= 0; i--) {
    if (valid[i] === last) len++;
    else break;
  }
  return { side: last, length: len };
}

// ── Базовый сигнал мерцания ──────────────────────────────

function flickerBaseSignal(
  flickerBias: number,
  flickerRate: number,
  history: RoundResult[]
): { hint: Reactor; weight: number; reason: string } {
  if (Math.abs(flickerBias) < 0.05 || flickerRate < 0.3) {
    return { hint: null, weight: 0, reason: "" };
  }

  const baseHint: Reactor = flickerBias > 0 ? "omega" : "alpha";
  const rateWeight = Math.min(flickerRate / 4, 1.0);
  const biasWeight = Math.min(Math.abs(flickerBias) * 1.5, 1.0);
  const weight = rateWeight * biasWeight * 0.45;

  const withFlicker = history.filter(r => r.flickerBias !== 0 && r.flickerRate > 0.3);
  if (withFlicker.length >= 3) {
    const hits = withFlicker.filter(r => {
      const fp: Reactor = r.flickerBias > 0 ? "omega" : "alpha";
      return fp === r.winner;
    }).length;
    const acc = hits / withFlicker.length;
    const multiplier = 0.3 + acc * 1.4;
    const adj = Math.min(weight * multiplier, 0.5);
    return { hint: baseHint, weight: adj, reason: `мерц. ${flickerRate.toFixed(1)}/с (точн. ${Math.round(acc * 100)}%)` };
  }

  return { hint: baseHint, weight, reason: `мерц. ${flickerRate.toFixed(1)}/с` };
}

// ── Сигнал взаимосвязи паттерн↔мерцание ─────────────────
// Если текущее мерцание похоже на профиль паттерна → усиливаем или ослабляем

function flickerPatternBonus(
  pattern: Pattern,
  flickerBias: number,
  flickerRate: number,
  flickerSwitchCount: number,
  flickerProfiles: Map<string, FlickerProfile>
): { bonus: number; reason: string } {
  const key = pattern.sequence.map(r => r).join(",") + "→" + pattern.next;
  const profile = flickerProfiles.get(key);
  if (!profile || profile.sampleCount < 2) return { bonus: 0, reason: "" };

  // Сходство по трём осям: темп, смещение, число переключений
  const rateDiff = Math.abs(flickerRate - profile.avgRate);
  const biasDiff = Math.abs(flickerBias - profile.avgBias);
  const switchDiff = Math.abs(flickerSwitchCount - profile.avgSwitchCount);

  const rateSim = Math.max(0, 1 - rateDiff / 3);
  const biasSim = Math.max(0, 1 - biasDiff / 0.5);
  // switchCount: допуск ±3 переключения
  const switchSim = Math.max(0, 1 - switchDiff / 6);

  // Взвешенное сходство: switchCount даёт 40% веса
  const similarity = rateSim * 0.3 + biasSim * 0.3 + switchSim * 0.4;

  const bonus = (similarity - 0.5) * 0.22;
  const direction = bonus > 0 ? "↑ мерцание совпадает" : "↓ мерцание не типично";
  const reason = profile.sampleCount >= 3
    ? `${direction} (sw≈${profile.avgSwitchCount.toFixed(1)}, sim=${Math.round(similarity * 100)}%)`
    : "";

  return { bonus, reason };
}

// ── Детектор периодичности по номеру шага mod M ──────────
// Для каждого M от 2 до 12 строим таблицу: remainder → {alpha, omega}
// Если на текущем остатке одна сторона побеждает значимо чаще — сигнал

function detectModPeriodicity(history: RoundResult[]): ModSignal | null {
  const n = history.length;
  if (n < 10) return null;

  let bestSignal: ModSignal | null = null;
  let bestScore = 0;

  for (let M = 2; M <= 12; M++) {
    // Для каждого остатка считаем α и ω с весом давности
    const buckets: { alphaW: number; omegaW: number; count: number }[] = Array.from({ length: M }, () => ({ alphaW: 0, omegaW: 0, count: 0 }));

    for (let i = 0; i < n; i++) {
      const r = history[i];
      if (!r.winner) continue;
      const rem = i % M;
      const w = recencyWeight(i, n);
      buckets[rem].count++;
      if (r.winner === "alpha") buckets[rem].alphaW += w;
      else buckets[rem].omegaW += w;
    }

    // Текущий шаг — следующий после последнего
    const nextRem = n % M;
    const bucket = buckets[nextRem];
    const totalW = bucket.alphaW + bucket.omegaW;
    if (bucket.count < 3 || totalW === 0) continue;

    const alphaRate = bucket.alphaW / totalW;
    const dominance = Math.abs(alphaRate - 0.5);

    // Проверяем, что неравномерность не случайна:
    // сравниваем с средней неравномерностью по другим остаткам
    const otherBuckets = buckets.filter((_, idx) => idx !== nextRem);
    const avgOtherDominance = otherBuckets.length > 0
      ? otherBuckets.reduce((s, b) => {
          const tw = b.alphaW + b.omegaW;
          return tw > 0 ? s + Math.abs(b.alphaW / tw - 0.5) : s;
        }, 0) / otherBuckets.length
      : 0;

    // Сигнал значим если текущий остаток заметно выделяется
    const relativeStrength = dominance - avgOtherDominance;
    if (dominance < 0.18 || relativeStrength < 0.08) continue;

    // Уверенность: чем больше образцов и сильнее доминирование — тем лучше
    const confidence = Math.min(dominance * 1.6, 0.9) * Math.min(bucket.count / 8, 1);
    const score = confidence * relativeStrength;

    if (score > bestScore) {
      bestScore = score;
      bestSignal = {
        M,
        remainder: nextRem,
        reactor: alphaRate >= 0.5 ? "alpha" : "omega",
        confidence,
        sampleCount: bucket.count,
      };
    }
  }

  return bestSignal;
}

// ── Детектор зависимости от абсолютного времени (мс) ─────
// Разбиваем timestamp по нескольким периодам T (500мс..5000мс)
// Смотрим: в каком временном bucket сейчас находимся и кто там побеждает

function detectTimePeriodicity(history: RoundResult[], nextTimestamp: number): TimeSignal | null {
  const n = history.length;
  if (n < 10) return null;

  // Проверяем периоды: 500, 750, 1000, 1500, 2000, 3000, 5000 мс
  const periods = [500, 750, 1000, 1500, 2000, 3000, 5000];
  // Делим каждый период на 4 bucket'а → ширина bucket = T/4
  const BUCKETS = 4;

  let bestSignal: TimeSignal | null = null;
  let bestScore = 0;

  for (const T of periods) {
    const bucketSize = T / BUCKETS;
    const table: { alphaW: number; omegaW: number; count: number }[] = Array.from({ length: BUCKETS }, () => ({ alphaW: 0, omegaW: 0, count: 0 }));

    for (let i = 0; i < n; i++) {
      const r = history[i];
      if (!r.winner) continue;
      const bucketIdx = Math.floor((r.timestamp % T) / bucketSize) % BUCKETS;
      const w = recencyWeight(i, n);
      table[bucketIdx].count++;
      if (r.winner === "alpha") table[bucketIdx].alphaW += w;
      else table[bucketIdx].omegaW += w;
    }

    const nextBucket = Math.floor((nextTimestamp % T) / bucketSize) % BUCKETS;
    const b = table[nextBucket];
    const totalW = b.alphaW + b.omegaW;
    if (b.count < 3 || totalW === 0) continue;

    const alphaRate = b.alphaW / totalW;
    const dominance = Math.abs(alphaRate - 0.5);

    // Сравниваем с другими bucket'ами
    const others = table.filter((_, idx) => idx !== nextBucket);
    const avgOther = others.reduce((s, ob) => {
      const tw = ob.alphaW + ob.omegaW;
      return tw > 0 ? s + Math.abs(ob.alphaW / tw - 0.5) : s;
    }, 0) / Math.max(others.length, 1);

    const relativeStrength = dominance - avgOther;
    if (dominance < 0.18 || relativeStrength < 0.08) continue;

    const confidence = Math.min(dominance * 1.5, 0.88) * Math.min(b.count / 8, 1);
    const score = confidence * relativeStrength;

    if (score > bestScore) {
      bestScore = score;
      bestSignal = {
        periodMs: T,
        bucketIdx: nextBucket,
        reactor: alphaRate >= 0.5 ? "alpha" : "omega",
        confidence,
        sampleCount: b.count,
      };
    }
  }

  return bestSignal;
}

// ── Основное предсказание ────────────────────────────────

export function predict(
  history: RoundResult[],
  flickerBias: number,
  flickerRate: number,
  flickerSwitchCount = 0
): Prediction {
  const reactorHistory = history.map(r => r.winner);
  const validHistory = reactorHistory.filter(r => r !== null);

  if (history.length < 2) {
    return {
      reactor: null, confidence: 0,
      reason: "Недостаточно данных (нужно 2+ раунда)",
      patternMatch: null, flickerHint: null, flickerWeight: 0,
      modSignal: null, timeSignal: null,
      signals: { patternScore: 0, flickerScore: 0, flickerPatternScore: 0, balanceScore: 0, streakScore: 0, adaptScore: 0, modScore: 0, timeScore: 0 },
    };
  }

  const patterns = findPatterns(reactorHistory);
  const flickerProfiles = buildFlickerProfiles(history);

  // Прикрепляем профили мерцания к паттернам
  patterns.forEach(p => {
    const key = p.sequence.join(",") + "→" + p.next;
    p.flickerProfile = flickerProfiles.get(key) ?? null;
  });

  // ── 1. Лучший паттерн ────────────────────────────────
  let bestPattern: Pattern | null = null;
  for (const len of SEQ_LENGTHS) {
    const tail = reactorHistory.slice(-len);
    if (tail.includes(null)) continue;
    const key = tail.join(",");
    const match = patterns.find(p => p.sequence.join(",") === key && p.confidence > 0.5);
    if (match) { bestPattern = match; break; }
  }

  // ── Детекторы периодичности ───────────────────────────
  const nextStepIdx = history.length;           // номер следующего раунда (0-based)
  const nextTimestamp = Date.now();
  const modSignal = detectModPeriodicity(history);
  const timeSignal = detectTimePeriodicity(history, nextTimestamp);

  let alphaScore = 0;
  let omegaScore = 0;
  const signals: SignalBreakdown = {
    patternScore: 0, flickerScore: 0, flickerPatternScore: 0,
    balanceScore: 0, streakScore: 0, adaptScore: 0,
    modScore: 0, timeScore: 0,
  };
  const reasonParts: string[] = [];
  void nextStepIdx;

  // Вес паттерна по длине: 5→0.50, 4→0.42, 3→0.34, 2→0.26
  if (bestPattern) {
    const lenW = 0.18 + bestPattern.sequence.length * 0.064;
    const patW = bestPattern.confidence * lenW;
    if (bestPattern.next === "alpha") alphaScore += patW;
    else omegaScore += patW;
    signals.patternScore = patW;
    reasonParts.push(`паттерн ${bestPattern.label} len=${bestPattern.sequence.length} (${Math.round(bestPattern.confidence * 100)}%)`);

    // Бонус взаимосвязи паттерн↔мерцание (включая число переключений)
    const { bonus, reason: bonusReason } = flickerPatternBonus(bestPattern, flickerBias, flickerRate, flickerSwitchCount, flickerProfiles);
    if (bonus !== 0) {
      if (bestPattern.next === "alpha") alphaScore += bonus;
      else omegaScore += bonus;
      signals.flickerPatternScore = Math.abs(bonus);
      if (bonusReason) reasonParts.push(bonusReason);
    }
  }

  // ── 2. Баланс 50/50 ──────────────────────────────────
  const alphaTotal = validHistory.filter(r => r === "alpha").length;
  const alphaPct = validHistory.length > 0 ? alphaTotal / validHistory.length : 0.5;
  const balancePull = (0.5 - alphaPct) * BALANCE_PULL;
  alphaScore += balancePull;
  omegaScore -= balancePull;
  signals.balanceScore = Math.abs(balancePull);
  if (Math.abs(alphaPct - 0.5) > 0.1) {
    reasonParts.push(`баланс ${alphaPct > 0.5 ? "α" : "ω"} +${Math.round(Math.abs(alphaPct - 0.5) * 200)}%`);
  }

  // ── 3. Серии 6+ (без ограничения сверху) ─────────────
  const streak = detectStreak(reactorHistory);
  if (streak.length >= STREAK_THRESHOLD) {
    // Чем длиннее серия — тем сильнее ожидаем смену, но с насыщением
    // 6→0.12, 8→0.18, 10→0.22, 15→0.28, 20→0.32
    const streakW = Math.min(0.12 + Math.log(streak.length - STREAK_THRESHOLD + 1) * 0.08, 0.35);
    if (streak.side === "alpha") omegaScore += streakW;
    else alphaScore += streakW;
    signals.streakScore = streakW;
    reasonParts.push(`серия ${streak.length}×${reactorLabel(streak.side)} → смена`);
  }

  // ── 4. Мерцание ──────────────────────────────────────
  const flicker = flickerBaseSignal(flickerBias, flickerRate, history);
  if (flicker.hint) {
    if (flicker.hint === "alpha") alphaScore += flicker.weight;
    else omegaScore += flicker.weight;
    signals.flickerScore = flicker.weight;
    reasonParts.push(flicker.reason);
  }

  // ── 5. Периодичность по номеру шага mod M ────────────
  if (modSignal) {
    // Вес пропорционален уверенности, но не доминирует над паттерном
    const modW = modSignal.confidence * 0.28;
    if (modSignal.reactor === "alpha") alphaScore += modW;
    else omegaScore += modW;
    signals.modScore = modW;
    reasonParts.push(`шаг%${modSignal.M}=${modSignal.remainder} → ${modSignal.reactor === "alpha" ? "α" : "ω"} (${Math.round(modSignal.confidence * 100)}%, n=${modSignal.sampleCount})`);
  }

  // ── 5b. Периодичность по абсолютному времени ─────────
  if (timeSignal) {
    const timeW = timeSignal.confidence * 0.22;
    if (timeSignal.reactor === "alpha") alphaScore += timeW;
    else omegaScore += timeW;
    signals.timeScore = timeW;
    reasonParts.push(`t%${timeSignal.periodMs}мс bkt=${timeSignal.bucketIdx} → ${timeSignal.reactor === "alpha" ? "α" : "ω"} (${Math.round(timeSignal.confidence * 100)}%, n=${timeSignal.sampleCount})`);
  }

  // ── 6. Адаптивная коррекция (последние 10) ───────────
  const recent = history.slice(-10);
  const recentWithPred = recent.filter(r => r.predictionHit !== null);
  const recentHits = recentWithPred.filter(r => r.predictionHit).length;
  if (recentWithPred.length >= 3) {
    const acc = recentHits / recentWithPred.length;
    if (acc < 0.4) {
      const adaptW = (0.4 - acc) * 0.3;
      if (alphaScore > omegaScore) omegaScore += adaptW;
      else alphaScore += adaptW;
      signals.adaptScore = adaptW;
      reasonParts.push(`адапт. (точн. ${Math.round(acc * 100)}%)`);
    }
  }

  // ── Финал ────────────────────────────────────────────
  if (alphaScore === 0 && omegaScore === 0) { alphaScore = 0.5; omegaScore = 0.5; }
  const total = alphaScore + omegaScore;
  const alphaNorm = alphaScore / total;
  const reactor: Reactor = alphaNorm >= 0.5 ? "alpha" : "omega";
  const rawConf = Math.max(alphaNorm, 1 - alphaNorm);
  const confidence = Math.min(0.92, 0.5 + (rawConf - 0.5) * 1.6);

  return {
    reactor, confidence,
    reason: reasonParts.length > 0 ? reasonParts.join(" · ") : "базовая статистика",
    patternMatch: bestPattern,
    flickerHint: flicker.hint,
    flickerWeight: flicker.weight,
    signals,
    modSignal,
    timeSignal,
  };
}

// ── Топ паттернов ────────────────────────────────────────
export function getTopPatterns(history: RoundResult[]): Pattern[] {
  return findPatterns(history.map(r => r.winner), MIN_PATTERN_COUNT).slice(0, 8);
}
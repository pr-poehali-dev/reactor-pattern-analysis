/**
 * ML-предсказатель v7 — кардинально переработан:
 *
 * Шаг 0: Анализ кадра (screenAnalyzer) — жёлтые пиксели, EMA, доминирование, спайк
 *
 * Сигнал 1: Паттерны последовательностей
 *   — Два специальных структурных паттерна высокого приоритета:
 *     a) Чередование 5 (AOAOA / OAOAO) → три одинаковых противоположных
 *     b) Серия 5 (AAAAA / OOOOO) → ещё один + противоположный
 *   — Общий поиск паттернов длиной 5→4→3→2 с весом давности
 *   — Адаптация: система отслеживает точность каждого типа паттерна
 *
 * Сигнал 2: Анализ мерцания (flicker)
 *   — Скорость переключений и смещение bias сопоставляются с историей
 *
 * Сигнал 3: Баланс 50/50 — регуляризация
 *
 * Сигнал 4: Периодичность по времени
 */

import type { Reactor, RoundResult } from "./screenAnalyzer";

// ── Публичные интерфейсы ──────────────────────────────────

export interface Pattern {
  sequence: Reactor[];
  next: Reactor;
  count: number;
  confidence: number;
  label: string;
  weight: number;
  flickerProfile: FlickerProfile | null;
}

export interface FlickerProfile {
  avgRate: number;
  avgBias: number;
  avgSwitchCount: number;
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
  timeSignal: TimeSignal | null;
  streakFlickerSignal: null; // оставлено для совместимости
}

export interface SignalBreakdown {
  patternScore: number;
  flickerScore: number;
  flickerPatternScore: number;
  balanceScore: number;
  adaptScore: number;
  timeScore: number;
  alternationScore: number;
  streakFlickerScore: number;
  comboScore: number;
}

export interface TimeSignal {
  periodMs: number;
  bucketIdx: number;
  reactor: Reactor;
  confidence: number;
  sampleCount: number;
}

export interface StreakFlickerSignal {
  streakSide: Reactor;
  streakLen: number;
  reactor: Reactor;
  confidence: number;
  similarity: number;
  sampleCount: number;
}

// ── Настройки ────────────────────────────────────────────

const SEQ_LENGTHS = [5, 4, 3, 2];
const MIN_PATTERN_COUNT = 2;
const RECENCY_WINDOW = 25;
const BALANCE_PULL = 0.12;

// ── Адаптивная точность сигналов ─────────────────────────

const MAX_SIGNAL_HISTORY = 80;
const MIN_SIGNAL_SAMPLES = 6;

interface SignalStat { hit: boolean }
const signalStats: Map<string, SignalStat[]> = new Map();

function recordSignalResult(key: string, hit: boolean) {
  if (!signalStats.has(key)) signalStats.set(key, []);
  const arr = signalStats.get(key)!;
  arr.push({ hit });
  if (arr.length > MAX_SIGNAL_HISTORY) arr.shift();
}

function signalAccuracy(key: string): number | null {
  const arr = signalStats.get(key);
  if (!arr || arr.length < MIN_SIGNAL_SAMPLES) return null;
  return arr.filter(x => x.hit).length / arr.length;
}

function signalSampleCount(key: string): number {
  return signalStats.get(key)?.length ?? 0;
}

function signalMultiplier(key: string): number {
  const acc = signalAccuracy(key);
  if (acc === null) return 1.0;
  if (acc < 0.45) return Math.max(0, (acc - 0.35) * 2.0);
  if (acc < 0.50) return 0.2 + (acc - 0.45) * 8.0;
  if (acc < 0.60) return 0.6 + (acc - 0.50) * 6.0;
  if (acc < 0.70) return 1.2 + (acc - 0.60) * 8.0;
  return Math.min(2.5, 2.0 + (acc - 0.70) * 5.0);
}

// ── Снапшот предыдущих сигналов для записи точности ──────

interface SignalSnapshot {
  structuralPattern: Reactor;
  generalPattern: Reactor;
  flicker: Reactor;
  time: Reactor;
}
let prevSnapshot: SignalSnapshot | null = null;

function updateSignalAccuracies(actual: Reactor) {
  if (!prevSnapshot || actual === null) return;
  const snap = prevSnapshot;
  if (snap.structuralPattern !== null) recordSignalResult("structuralPattern", snap.structuralPattern === actual);
  if (snap.generalPattern !== null)    recordSignalResult("generalPattern",    snap.generalPattern === actual);
  if (snap.flicker !== null)           recordSignalResult("flicker",           snap.flicker === actual);
  if (snap.time !== null)              recordSignalResult("time",              snap.time === actual);
}

// ── Публичная диагностика сигналов ───────────────────────

export interface SignalDiagnostic {
  key: string;
  label: string;
  accuracy: number | null;
  samples: number;
  multiplier: number;
  color: string;
}

export function getSignalDiagnostics(): SignalDiagnostic[] {
  return [
    { key: "structuralPattern", label: "Структ.паттерн", color: "#00ffcc" },
    { key: "generalPattern",    label: "Паттерн",        color: "#34d399" },
    { key: "flicker",           label: "Мерцание",       color: "#facc15" },
    { key: "time",              label: "Время",          color: "#e879f9" },
  ].map(s => ({
    ...s,
    accuracy: signalAccuracy(s.key),
    samples: signalSampleCount(s.key),
    multiplier: signalMultiplier(s.key),
  }));
}

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
function opposite(r: Reactor): Reactor {
  if (r === "alpha") return "omega";
  if (r === "omega") return "alpha";
  return null;
}

// ── Паттерны флicker-профилей ─────────────────────────────

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

// ── СИГНАЛ 1a: Структурные паттерны высокого приоритета ──
//
// Паттерн "Чередование-5": последние 5 результатов строго чередуются
//   AOAOA или OAOAO → ожидаем 3 подряд противоположных (OOO или AAA)
//   Гипотеза адаптивна: система проверяет историю и уточняет вес
//
// Паттерн "Серия-5": последние 5 результатов одинаковы
//   AAAAA или OOOOO → ожидаем ещё один тот же, затем противоположный
//   Т.е. шестой = тот же, седьмой = противоположный
//   Сейчас мы на позиции "следующий = шестой" → предсказываем тот же

interface StructuralSignal {
  type: "alternation5" | "series5";
  prediction: Reactor;
  baseConfidence: number;
  historicalAccuracy: number | null;
  historicalSamples: number;
  label: string;
}

function detectStructuralPattern(
  history: RoundResult[]
): StructuralSignal | null {
  const winners = history.map(r => r.winner).filter(r => r !== null) as Reactor[];
  if (winners.length < 5) return null;

  const tail5 = winners.slice(-5);

  // ── Проверка "Чередование-5" ──────────────────────────
  // Строгое чередование: каждый элемент противоположен предыдущему
  const isAlternating = tail5.every((v, i) => i === 0 || v === opposite(tail5[i - 1]));

  if (isAlternating) {
    // Предсказываем: шестой = противоположен пятому (начало серии из трёх)
    const prediction = opposite(tail5[4])!;

    // Проверяем историческую точность этого паттерна
    let hits = 0;
    let total = 0;
    for (let i = 0; i <= winners.length - 6; i++) {
      const seg = winners.slice(i, i + 5);
      const nextVal = winners[i + 5];
      const segAlt = seg.every((v, j) => j === 0 || v === opposite(seg[j - 1]));
      if (segAlt) {
        total++;
        const expected = opposite(seg[4]);
        if (nextVal === expected) hits++;
      }
    }

    const historicalAccuracy = total >= 3 ? hits / total : null;
    const baseConfidence = 0.70; // высокая начальная уверенность

    return {
      type: "alternation5",
      prediction,
      baseConfidence,
      historicalAccuracy,
      historicalSamples: total,
      label: `Чер-5[${seqLabel(tail5)}→${reactorLabel(prediction)}]`,
    };
  }

  // ── Проверка "Серия-5" ────────────────────────────────
  // Пять одинаковых подряд
  const isSeries = tail5.every(v => v === tail5[0]);

  if (isSeries) {
    // Шестой = тот же (продолжение серии перед разворотом)
    const prediction = tail5[0];

    // Проверяем историческую точность: серия 5 → шестой тот же?
    let hits = 0;
    let total = 0;
    for (let i = 0; i <= winners.length - 6; i++) {
      const seg = winners.slice(i, i + 5);
      const nextVal = winners[i + 5];
      const segSeries = seg.every(v => v === seg[0]);
      if (segSeries) {
        total++;
        if (nextVal === seg[0]) hits++;
      }
    }

    const historicalAccuracy = total >= 3 ? hits / total : null;
    const baseConfidence = 0.65;

    return {
      type: "series5",
      prediction,
      baseConfidence,
      historicalAccuracy,
      historicalSamples: total,
      label: `Сер-5[${seqLabel(tail5)}→${reactorLabel(prediction)}]`,
    };
  }

  return null;
}

// ── СИГНАЛ 1b: Общий поиск паттернов (5→4→3→2) ───────────

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
      flickerProfile: null,
    });
  });

  return result.sort((a, b) => {
    const lenDiff = b.sequence.length - a.sequence.length;
    if (lenDiff !== 0) return lenDiff;
    return b.confidence * b.weight - a.confidence * a.weight;
  });
}

// ── Бонус: сравнение флicker с профилем паттерна ─────────

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

  const rateSim  = Math.max(0, 1 - Math.abs(flickerRate - profile.avgRate) / 2);
  const biasSim  = Math.max(0, 1 - Math.abs(flickerBias - profile.avgBias) / 0.35);
  const switchSim = Math.max(0, 1 - Math.abs(flickerSwitchCount - profile.avgSwitchCount) / 4);
  const similarity = rateSim * 0.25 + biasSim * 0.25 + switchSim * 0.5;

  const bonus = (similarity - 0.5) * 0.28;
  const direction = bonus > 0 ? "↑ мерцание совпадает" : "↓ мерцание не типично";
  const reason = profile.sampleCount >= 3
    ? `${direction} (sim=${Math.round(similarity * 100)}%)`
    : "";

  return { bonus, reason };
}

// ── СИГНАЛ 2: Анализ мерцания ────────────────────────────
//
// Гипотеза: проигрывающий реактор мерцает интенсивнее (bias > 0 → alpha мерцала больше → omega побеждает)
// Скорость переключений (rate) используется как мера надёжности сигнала
// Система сверяет с историческими раундами где мерцание было активным

function flickerSignal(
  flickerBias: number,
  flickerRate: number,
  flickerSwitchCount: number,
  history: RoundResult[]
): { hint: Reactor; weight: number; reason: string } {
  if (Math.abs(flickerBias) < 0.03 || flickerRate < 0.15) {
    return { hint: null, weight: 0, reason: "" };
  }

  // Основная гипотеза: кто мерцал больше — тот проигрывает
  const baseHint: Reactor = flickerBias > 0 ? "omega" : "alpha";

  const rateWeight   = Math.min(flickerRate / 2.5, 1.0);
  const biasWeight   = Math.min(Math.abs(flickerBias) * 2.0, 1.0);
  const switchBonus  = Math.min(flickerSwitchCount / 10, 0.4);
  let weight = (rateWeight * biasWeight * 0.55) + switchBonus * 0.15;

  // Адаптация: историческая точность гипотезы мерцания
  const withFlicker = history.filter(r => r.flickerBias !== 0 && r.flickerRate > 0.15);
  if (withFlicker.length >= 3) {
    const hits = withFlicker.filter(r => {
      const fp: Reactor = r.flickerBias > 0 ? "omega" : "alpha";
      return fp === r.winner;
    }).length;
    const acc = hits / withFlicker.length;

    // Линейная поправка: 50% → ×1.0, 70% → ×1.6, 30% → ×0.4
    const multiplier = 0.4 + acc * 1.6;
    weight = Math.min(weight * multiplier, 0.65);

    // Если точность сильно ниже 50% — гипотеза инвертирована
    const hint: Reactor = acc >= 0.4 ? baseHint : opposite(baseHint);
    return {
      hint,
      weight,
      reason: `мерц. ${flickerRate.toFixed(1)}/с bias=${flickerBias > 0 ? "α+" : "ω+"}${Math.round(Math.abs(flickerBias) * 100)}% точн.${Math.round(acc * 100)}%`,
    };
  }

  return {
    hint: baseHint,
    weight,
    reason: `мерц. ${flickerRate.toFixed(1)}/с sw=${flickerSwitchCount}`,
  };
}

// ── СИГНАЛ 4: Периодичность по времени ──────────────────

export function detectTimePeriodicity(history: RoundResult[], nextTimestamp: number): TimeSignal | null {
  const n = history.length;
  if (n < 10) return null;

  const periods = [500, 750, 1000, 1500, 2000, 3000, 5000];
  const BUCKETS = 4;

  let bestSignal: TimeSignal | null = null;
  let bestScore = 0;

  for (const T of periods) {
    const bucketSize = T / BUCKETS;
    const table: { alphaW: number; omegaW: number; count: number }[] =
      Array.from({ length: BUCKETS }, () => ({ alphaW: 0, omegaW: 0, count: 0 }));

    for (let i = 0; i < n; i++) {
      const r = history[i];
      if (!r.winner) continue;
      const bucketIdx = Math.floor((r.timestamp % T) / bucketSize) % BUCKETS;
      const w = recencyWeight(i, n);
      if (r.winner === "alpha") table[bucketIdx].alphaW += w;
      else table[bucketIdx].omegaW += w;
      table[bucketIdx].count++;
    }

    const nextBucket = Math.floor((nextTimestamp % T) / bucketSize) % BUCKETS;
    const b = table[nextBucket];
    const totalW = b.alphaW + b.omegaW;
    if (totalW === 0 || b.count < 3) continue;

    const alphaRate = b.alphaW / totalW;
    const dominance = Math.abs(alphaRate - 0.5);

    // Сравниваем с другими бакетами
    const others = table.filter((_, i) => i !== nextBucket);
    const avgOtherDom = others.reduce((s, o) => {
      const tot = o.alphaW + o.omegaW;
      return s + (tot > 0 ? Math.abs(o.alphaW / tot - 0.5) : 0);
    }, 0) / Math.max(others.length, 1);

    const relativeStrength = dominance - avgOtherDom;
    if (dominance < 0.12 || relativeStrength < 0.05) continue;

    const confidence = Math.min(dominance * 1.8, 0.92) * Math.min(b.count / 6, 1);
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

// ── Основное предсказание ─────────────────────────────────

export function predict(
  history: RoundResult[],
  flickerBias: number,
  flickerRate: number,
  flickerSwitchCount = 0,
  flickerAlphaRedness = 0.5,
  flickerOmegaRedness = 0.5
): Prediction {
  // Записываем точность сигналов предыдущего раунда
  if (history.length >= 1) {
    const last = history[history.length - 1];
    if (last.winner !== null) updateSignalAccuracies(last.winner);
  }

  const reactorHistory = history.map(r => r.winner);
  const validHistory = reactorHistory.filter(r => r !== null);

  if (history.length < 2) {
    return {
      reactor: null, confidence: 0,
      reason: "Недостаточно данных (нужно 2+ раунда)",
      patternMatch: null, flickerHint: null, flickerWeight: 0,
      timeSignal: null, streakFlickerSignal: null,
      signals: {
        patternScore: 0, flickerScore: 0, flickerPatternScore: 0,
        balanceScore: 0, adaptScore: 0, timeScore: 0,
        alternationScore: 0, streakFlickerScore: 0, comboScore: 0,
      },
    };
  }

  // ── Подготовка паттернов ────────────────────────────
  const patterns = findPatterns(reactorHistory);
  const flickerProfiles = buildFlickerProfiles(history);
  patterns.forEach(p => {
    const key = p.sequence.join(",") + "→" + p.next;
    p.flickerProfile = flickerProfiles.get(key) ?? null;
  });

  // Ищем лучший общий паттерн (5→4→3→2)
  let bestGeneralPattern: Pattern | null = null;
  for (const len of SEQ_LENGTHS) {
    const tail = reactorHistory.slice(-len);
    if (tail.includes(null)) continue;
    const key = tail.join(",");
    const match = patterns.find(p => p.sequence.join(",") === key && p.confidence > 0.5);
    if (match) { bestGeneralPattern = match; break; }
  }

  // ── Структурный паттерн ─────────────────────────────
  const structural = detectStructuralPattern(history);

  // ── Остальные сигналы ───────────────────────────────
  const flicker = flickerSignal(flickerBias, flickerRate, flickerSwitchCount, history);
  const nextTimestamp = Date.now();
  const timeSignal = detectTimePeriodicity(history, nextTimestamp);

  // ── Суммирование счётов ─────────────────────────────
  let alphaScore = 0;
  let omegaScore = 0;
  const signals: SignalBreakdown = {
    patternScore: 0, flickerScore: 0, flickerPatternScore: 0,
    balanceScore: 0, adaptScore: 0, timeScore: 0,
    alternationScore: 0, streakFlickerScore: 0, comboScore: 0,
  };
  const reasonParts: string[] = [];

  // ── СИГНАЛ 1a: Структурный паттерн (наивысший приоритет) ──
  let structuralPrediction: Reactor = null;
  if (structural) {
    const mult = signalMultiplier("structuralPattern");

    // Финальная уверенность: используем историческую если есть, иначе базовую
    let effectiveConf: number;
    if (structural.historicalAccuracy !== null && structural.historicalSamples >= 3) {
      // Взвешиваем базовую и историческую: чем больше сэмплов — тем больше доверяем истории
      const histWeight = Math.min(structural.historicalSamples / 15, 0.8);
      effectiveConf = structural.baseConfidence * (1 - histWeight) + structural.historicalAccuracy * histWeight;
    } else {
      effectiveConf = structural.baseConfidence;
    }

    // Если историческая точность ниже 40% — инвертируем предсказание (адаптация)
    let finalPrediction = structural.prediction;
    if (structural.historicalAccuracy !== null && structural.historicalAccuracy < 0.40) {
      finalPrediction = opposite(structural.prediction)!;
      effectiveConf = 1 - effectiveConf;
    }

    const score = effectiveConf * 0.55 * mult; // высокий вес структурного паттерна
    if (finalPrediction === "alpha") alphaScore += score;
    else omegaScore += score;

    signals.alternationScore = score; // используем поле alternationScore для структурного
    structuralPrediction = finalPrediction;

    const accStr = structural.historicalAccuracy !== null
      ? ` ист.${Math.round(structural.historicalAccuracy * 100)}%[${structural.historicalSamples}]`
      : "";
    reasonParts.push(`${structural.label}${accStr}×${mult.toFixed(1)}`);
  }

  // ── СИГНАЛ 1b: Общий паттерн ──────────────────────────
  let generalPatternPrediction: Reactor = null;
  if (bestGeneralPattern) {
    const lenW = 0.18 + bestGeneralPattern.sequence.length * 0.064;
    const mult = signalMultiplier("generalPattern");
    const patW = bestGeneralPattern.confidence * lenW * mult;

    if (bestGeneralPattern.next === "alpha") alphaScore += patW;
    else omegaScore += patW;
    signals.patternScore = patW;
    generalPatternPrediction = bestGeneralPattern.next;
    reasonParts.push(`пат ${bestGeneralPattern.label}(${Math.round(bestGeneralPattern.confidence * 100)}%)×${mult.toFixed(1)}`);

    // Бонус флicker←→паттерн
    const { bonus, reason: bonusReason } = flickerPatternBonus(
      bestGeneralPattern, flickerBias, flickerRate, flickerSwitchCount, flickerProfiles
    );
    if (bonus !== 0) {
      if (bestGeneralPattern.next === "alpha") alphaScore += bonus;
      else omegaScore += bonus;
      signals.flickerPatternScore = Math.abs(bonus);
      if (bonusReason) reasonParts.push(bonusReason);
    }
  }

  // ── СИГНАЛ 2: Мерцание ─────────────────────────────────
  let flickerPrediction: Reactor = null;
  if (flicker.hint) {
    const flickMult = signalMultiplier("flicker");
    const flickW = flicker.weight * flickMult;
    if (flicker.hint === "alpha") alphaScore += flickW;
    else omegaScore += flickW;
    signals.flickerScore = flickW;
    flickerPrediction = flicker.hint;
    reasonParts.push(flicker.reason + `×${flickMult.toFixed(1)}`);
  }

  // ── СИГНАЛ 3: Баланс 50/50 ─────────────────────────────
  const alphaTotal = validHistory.filter(r => r === "alpha").length;
  const alphaPct = validHistory.length > 0 ? alphaTotal / validHistory.length : 0.5;
  const balancePull = (0.5 - alphaPct) * BALANCE_PULL;
  alphaScore += balancePull;
  omegaScore -= balancePull;
  signals.balanceScore = Math.abs(balancePull);
  if (Math.abs(alphaPct - 0.5) > 0.1) {
    reasonParts.push(`баланс ${alphaPct > 0.5 ? "α" : "ω"} +${Math.round(Math.abs(alphaPct - 0.5) * 200)}%`);
  }

  // ── СИГНАЛ 4: Временная периодичность ──────────────────
  let timePrediction: Reactor = null;
  if (timeSignal) {
    const timeMult = signalMultiplier("time");
    const tW = timeSignal.confidence * 0.55 * timeMult;
    if (timeSignal.reactor === "alpha") alphaScore += tW;
    else omegaScore += tW;
    signals.timeScore = tW;
    timePrediction = timeSignal.reactor;
    reasonParts.push(`время T=${timeSignal.periodMs}мс(${Math.round(timeSignal.confidence * 100)}%)×${timeMult.toFixed(1)}`);
  }

  // ── Финальный расчёт ───────────────────────────────────
  const total = alphaScore + omegaScore;
  if (total === 0) {
    return {
      reactor: "alpha", confidence: 0.5,
      reason: "недостаточно сигналов",
      patternMatch: bestGeneralPattern,
      flickerHint: null, flickerWeight: 0,
      timeSignal, streakFlickerSignal: null,
      signals,
    };
  }

  const alphaNorm = alphaScore / total;
  const reactor: Reactor = alphaNorm >= 0.5 ? "alpha" : "omega";
  const rawConf = Math.max(alphaNorm, 1 - alphaNorm);
  const confidence = Math.min(0.92, 0.5 + (rawConf - 0.5) * 1.6);

  // Сохраняем снапшот для следующего раунда
  prevSnapshot = {
    structuralPattern: structuralPrediction,
    generalPattern:    generalPatternPrediction,
    flicker:           flickerPrediction,
    time:              timePrediction,
  };

  return {
    reactor,
    confidence,
    reason: reasonParts.length > 0 ? reasonParts.join(" · ") : "базовая статистика",
    patternMatch: bestGeneralPattern,
    flickerHint: flicker.hint,
    flickerWeight: flicker.weight,
    signals,
    timeSignal,
    streakFlickerSignal: null,
  };
}

// ── Топ паттернов (публичный) ─────────────────────────────
export function getTopPatterns(history: RoundResult[]): Pattern[] {
  return findPatterns(history.map(r => r.winner), MIN_PATTERN_COUNT).slice(0, 8);
}
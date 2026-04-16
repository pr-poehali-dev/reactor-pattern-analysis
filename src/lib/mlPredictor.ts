/**
 * ML-предсказатель v6:
 * 1.  Паттерны длиной 5→4→3→2 с весом давности
 * 2.  Баланс 50/50 (регуляризация)
 * 3.  Взаимосвязь паттерн↔мерцание (flickerProfile)
 * 4.  Адаптивные веса сигналов: нелинейный multiplier — сильный штраф <50%, бонус >65%
 * 5.  Детектор зависимости от абсолютного времени (мс % T)
 * 6.  Детектор чередования: насколько часто результат меняется vs повторяется
 * 7.  Детектор профиля мерцания серий
 * 8.  Комбинированный анализ: история совместных предсказаний сигналов → корреляционная поправка
 * 9.  Адаптация: скользящий вес по последним 20 раундам
 * УДАЛЕНЫ (неэффективны <50%): серии (41%), шаг mod (46%), lag-корреляция (49%)
 */
import type { Reactor, RoundResult } from "./screenAnalyzer";

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
  streakFlickerSignal: StreakFlickerSignal | null;
}

export interface SignalBreakdown {
  patternScore: number;
  flickerScore: number;
  flickerPatternScore: number;
  balanceScore: number;
  adaptScore: number;
  timeScore: number;
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

// ── Хранилище точности сигналов (адаптивные веса) ────────
// Ключ → массив результатов за последние MAX_SIGNAL_HISTORY раундов

const MAX_SIGNAL_HISTORY = 80;
const MIN_SIGNAL_SAMPLES = 6; // минимум для показа точности

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

// Нелинейный вес сигнала на основе его исторической точности:
// <45% → штраф до 0 (сигнал не используется)
// 45-50% → слабый штраф (0.1..0.6)
// 50-60% → умеренный вес (0.6..1.2)
// 60-70% → сильный вес (1.2..2.0)  ← чередование 71% получит ~2.1
// >70%  → максимум 2.5
function signalMultiplier(key: string): number {
  const acc = signalAccuracy(key);
  if (acc === null) return 1.0;
  if (acc < 0.45) return Math.max(0, (acc - 0.35) * 2.0);  // 35%→0, 45%→0.2
  if (acc < 0.50) return 0.2 + (acc - 0.45) * 8.0;         // 45%→0.2, 50%→0.6
  if (acc < 0.60) return 0.6 + (acc - 0.50) * 6.0;         // 50%→0.6, 60%→1.2
  if (acc < 0.70) return 1.2 + (acc - 0.60) * 8.0;         // 60%→1.2, 70%→2.0
  return Math.min(2.5, 2.0 + (acc - 0.70) * 5.0);          // 70%→2.0, 80%→2.5
}

// ── Снапшот предыдущих сигналов для записи точности ──────
interface SignalSnapshot {
  pattern: Reactor;
  flicker: Reactor;
  time: Reactor;
}
let prevSnapshot: SignalSnapshot | null = null;

function updateSignalAccuracies(actual: Reactor) {
  if (!prevSnapshot || actual === null) return;
  const snap = prevSnapshot;
  if (snap.pattern !== null) recordSignalResult("pattern", snap.pattern === actual);
  if (snap.flicker !== null) recordSignalResult("flicker", snap.flicker === actual);
  if (snap.time !== null)    recordSignalResult("time",    snap.time === actual);
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
    { key: "pattern", label: "Паттерн",  color: "#00ffcc" },
    { key: "flicker", label: "Мерцание", color: "#facc15" },
    { key: "time",    label: "Время",    color: "#e879f9" },
    { key: "combo",   label: "Комбо",    color: "#818cf8" },
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

// ── Профили мерцания для каждого паттерна ────────────────

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

// ── Профили мерцания серий ────────────────────────────────
// Для каждой комбинации (side, length, outcome) запоминаем средний профиль мерцания
// "Когда идёт серия из 4 омег и реактор мерцает вот так → следующий обычно X"

interface StreakFlickerEntry {
  rates: number[];
  biases: number[];
  switches: number[];
  outcome: Reactor;
}

function buildStreakFlickerProfiles(
  history: RoundResult[]
): Map<string, StreakFlickerEntry[]> {
  // key = `${side}_${len}` → массив исходов с профилями мерцания
  const map: Map<string, StreakFlickerEntry[]> = new Map();

  for (let i = 1; i < history.length; i++) {
    const cur = history[i - 1];
    const next = history[i];
    if (!cur.winner || !next.winner) continue;

    // Считаем длину серии, заканчивающейся на i-1
    let len = 1;
    for (let j = i - 2; j >= 0; j--) {
      if (history[j].winner === cur.winner) len++;
      else break;
    }
    if (len < 2) continue;

    const key = `${cur.winner}_${len}`;
    if (!map.has(key)) {
      map.set(key, [
        { rates: [], biases: [], switches: [], outcome: "alpha" },
        { rates: [], biases: [], switches: [], outcome: "omega" },
      ]);
    }
    const entries = map.get(key)!;
    const entry = entries.find(e => e.outcome === next.winner)!;
    entry.rates.push(cur.flickerRate);
    entry.biases.push(cur.flickerBias);
    entry.switches.push(cur.flickerSwitchCount ?? 0);
  }

  return map;
}

function detectStreakFlickerSignal(
  history: RoundResult[],
  currentFlickerRate: number,
  currentFlickerBias: number,
  currentFlickerSwitchCount: number
): StreakFlickerSignal | null {
  const n = history.length;
  if (n < 8) return null;

  // Определяем текущую серию
  const valid = history.filter(r => r.winner !== null);
  if (valid.length < 3) return null;
  const last = valid[valid.length - 1];
  let streakLen = 1;
  for (let i = valid.length - 2; i >= 0; i--) {
    if (valid[i].winner === last.winner) streakLen++;
    else break;
  }
  if (streakLen < 2) return null;

  const profiles = buildStreakFlickerProfiles(history.slice(0, -1));

  // Ищем профиль для текущей серии (и более коротких, если нет данных)
  for (let tryLen = streakLen; tryLen >= 2; tryLen--) {
    const key = `${last.winner}_${tryLen}`;
    const entries = profiles.get(key);
    if (!entries) continue;

    const avg = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

    const scored: { outcome: Reactor; similarity: number; sampleCount: number }[] = [];
    for (const entry of entries) {
      const n = entry.rates.length;
      if (n < 2) continue;
      const avgRate = avg(entry.rates);
      const avgBias = avg(entry.biases);
      const avgSwitch = avg(entry.switches);

      const rateSim = Math.max(0, 1 - Math.abs(currentFlickerRate - avgRate) / 2);
      const biasSim = Math.max(0, 1 - Math.abs(currentFlickerBias - avgBias) / 0.35);
      const switchSim = Math.max(0, 1 - Math.abs(currentFlickerSwitchCount - avgSwitch) / 4);
      const similarity = rateSim * 0.25 + biasSim * 0.25 + switchSim * 0.5;

      scored.push({ outcome: entry.outcome, similarity, sampleCount: n });
    }

    if (scored.length < 2) continue;

    const [a, b] = scored.sort((x, y) => y.similarity - x.similarity);
    const diff = a.similarity - b.similarity;
    if (diff < 0.12) continue;

    const totalSamples = scored.reduce((s, e) => s + e.sampleCount, 0);
    const confidence = Math.min(diff * 2.2, 0.82) * Math.min(totalSamples / 10, 1);
    if (confidence < 0.12) continue;

    return {
      streakSide: last.winner,
      streakLen,
      reactor: a.outcome,
      confidence,
      similarity: a.similarity,
      sampleCount: totalSamples,
    };
  }

  return null;
}

// ── Поиск паттернов с весами давности ────────────────────

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



// ── Базовый сигнал мерцания ───────────────────────────────

// redness: R/(R+G) жёлтого сигнала. Чистый жёлтый ≈ 0.5, красноватый > 0.5, зеленоватый < 0.5
// Гипотеза: проигрывающий реактор светит более красноватым жёлтым → redness > 0.54 намекает на omega
// Исторически подтверждаем: собираем средний redness победителя и проигравшего и сравниваем
function flickerBaseSignal(
  flickerBias: number,
  flickerRate: number,
  history: RoundResult[],
  flickerSwitchCount = 0,
  flickerAlphaRedness = 0.5,
  flickerOmegaRedness = 0.5
): { hint: Reactor; weight: number; reason: string } {
  if (Math.abs(flickerBias) < 0.03 || flickerRate < 0.15) {
    return { hint: null, weight: 0, reason: "" };
  }

  const baseHint: Reactor = flickerBias > 0 ? "omega" : "alpha";
  const rateWeight = Math.min(flickerRate / 2.5, 1.0);
  const biasWeight = Math.min(Math.abs(flickerBias) * 2.0, 1.0);
  const switchBonus = Math.min(flickerSwitchCount / 10, 0.4);
  const weight = (rateWeight * biasWeight * 0.55) + switchBonus * 0.15;

  // ── Сигнал красноватости: кто из реакторов светит теплее ──
  // Разница redness между alpha и omega — дополнительный намёк
  const rednessDiff = flickerAlphaRedness - flickerOmegaRedness;
  // rednessDiff > 0 → alpha краснее → alpha вероятно проигрывает → omega
  // rednessDiff < 0 → omega краснее → omega вероятно проигрывает → alpha
  let rednessHint: Reactor = null;
  let rednessW = 0;
  if (Math.abs(rednessDiff) > 0.015) {
    // Историческое подтверждение: собираем средний redness победителя
    const withRedness = history.filter(r =>
      r.flickerRate > 0.15 &&
      (r.flickerAlphaRedness ?? 0.5) !== 0.5 &&
      (r.flickerOmegaRedness ?? 0.5) !== 0.5
    );
    let rednessAccConf = 0.5;
    if (withRedness.length >= 4) {
      // Проверяем: когда alphaRedness > omegaRedness — кто побеждал?
      const redAlphaHigher = withRedness.filter(r => (r.flickerAlphaRedness ?? 0.5) > (r.flickerOmegaRedness ?? 0.5));
      const redAlphaHigherOmegaWon = redAlphaHigher.filter(r => r.winner === "omega").length;
      if (redAlphaHigher.length >= 3) {
        rednessAccConf = redAlphaHigherOmegaWon / redAlphaHigher.length;
      }
    }
    // rednessAccConf: насколько часто "alpha краснее → omega выигрывает"
    // Если подтверждения нет (≈0.5) — используем слабый базовый сигнал
    const redConf = Math.abs(rednessAccConf - 0.5);
    if (redConf > 0.05 || withRedness.length < 4) {
      rednessHint = rednessDiff > 0
        ? (rednessAccConf >= 0.5 ? "omega" : "alpha")
        : (rednessAccConf >= 0.5 ? "alpha" : "omega");
      rednessW = Math.min(Math.abs(rednessDiff) * 3.5, 0.3) * Math.max(redConf * 2, 0.3);
    }
  }

  const withFlicker = history.filter(r => r.flickerBias !== 0 && r.flickerRate > 0.15);
  let finalHint = baseHint;
  let finalWeight = weight;
  let reasonExtra = `sw=${flickerSwitchCount}`;

  if (withFlicker.length >= 3) {
    const hits = withFlicker.filter(r => {
      const fp: Reactor = r.flickerBias > 0 ? "omega" : "alpha";
      return fp === r.winner;
    }).length;
    const acc = hits / withFlicker.length;
    const multiplier = 0.4 + acc * 1.6;
    finalWeight = Math.min(weight * multiplier, 0.62);
    reasonExtra += ` точн.${Math.round(acc * 100)}%`;
  }

  // Добавляем redness сигнал поверх
  if (rednessHint !== null) {
    if (rednessHint === finalHint) {
      finalWeight = Math.min(finalWeight + rednessW, 0.72);
    } else {
      finalWeight = Math.max(finalWeight - rednessW * 0.5, 0);
      if (rednessW > finalWeight) { finalHint = rednessHint; finalWeight = rednessW; }
    }
    const rLabel = Math.abs(rednessDiff) > 0 ? ` red=${rednessDiff > 0 ? "α+" : "ω+"}${Math.round(Math.abs(rednessDiff) * 100)}` : "";
    reasonExtra += rLabel;
  }

  return { hint: finalHint, weight: finalWeight, reason: `мерц. ${flickerRate.toFixed(1)}/с ${reasonExtra}` };
}

// ── Сигнал взаимосвязи паттерн↔мерцание ─────────────────

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

  const rateDiff = Math.abs(flickerRate - profile.avgRate);
  const biasDiff = Math.abs(flickerBias - profile.avgBias);
  const switchDiff = Math.abs(flickerSwitchCount - profile.avgSwitchCount);

  const rateSim = Math.max(0, 1 - rateDiff / 2);
  const biasSim = Math.max(0, 1 - biasDiff / 0.35);
  const switchSim = Math.max(0, 1 - switchDiff / 4);

  const similarity = rateSim * 0.25 + biasSim * 0.25 + switchSim * 0.5;

  const bonus = (similarity - 0.5) * 0.34;
  const direction = bonus > 0 ? "↑ мерцание совпадает" : "↓ мерцание не типично";
  const reason = profile.sampleCount >= 3
    ? `${direction} (sim=${Math.round(similarity * 100)}%)`
    : "";

  return { bonus, reason };
}



// ── Детектор периодичности по времени ────────────────────

function detectTimePeriodicity(history: RoundResult[], nextTimestamp: number): TimeSignal | null {
  const n = history.length;
  if (n < 10) return null;

  const periods = [500, 750, 1000, 1500, 2000, 3000, 5000];
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

    const others = table.filter((_, idx) => idx !== nextBucket);
    const avgOther = others.reduce((s, ob) => {
      const tw = ob.alphaW + ob.omegaW;
      return tw > 0 ? s + Math.abs(ob.alphaW / tw - 0.5) : s;
    }, 0) / Math.max(others.length, 1);

    const relativeStrength = dominance - avgOther;
    if (dominance < 0.12 || relativeStrength < 0.05) continue;

    const confidence = Math.min(dominance * 1.8, 0.92) * Math.min(b.count / 6, 1);
    const score = confidence * relativeStrength;

    if (score > bestScore) {
      bestScore = score;
      bestSignal = { periodMs: T, bucketIdx: nextBucket, reactor: alphaRate >= 0.5 ? "alpha" : "omega", confidence, sampleCount: b.count };
    }
  }

  return bestSignal;
}



// ── Детектор чередования ─────────────────────────────────
// Считает: как часто результат меняется (α→ω или ω→α) в последних N раундах
// Если чередование >70% → предсказываем смену; если <30% → продолжение

function detectAlternation(history: Reactor[], window = 20): { signal: Reactor; strength: number } | null {
  const valid = history.filter(r => r !== null);
  if (valid.length < 6) return null;

  const recent = valid.slice(-window);
  if (recent.length < 4) return null;

  let changes = 0;
  for (let i = 1; i < recent.length; i++) {
    if (recent[i] !== recent[i - 1]) changes++;
  }

  const changeRate = changes / (recent.length - 1);
  const last: Reactor = recent[recent.length - 1];

  if (changeRate >= 0.68) {
    // Высокое чередование → ожидаем смену
    const next: Reactor = last === "alpha" ? "omega" : "alpha";
    return { signal: next, strength: (changeRate - 0.68) * 2.5 };
  }
  if (changeRate <= 0.32) {
    // Низкое чередование → ожидаем продолжение
    return { signal: last, strength: (0.32 - changeRate) * 2.5 };
  }

  return null;
}

// ── Комбинированный анализ сигналов ──────────────────────
// Идея: когда несколько сигналов говорят одно и то же — это не просто сумма,
// а структурный паттерн. Смотрим историю: когда сигналы X и Y оба давали одно
// направление — насколько часто итог совпадал?
// Это позволяет системе самой находить синергии между сигналами.

interface SignalVote { pattern: Reactor; flicker: Reactor; time: Reactor }

// Сохраняем последние голоса каждого сигнала для обучения combo
const comboHistory: { votes: SignalVote; actual: Reactor }[] = [];
const MAX_COMBO_HISTORY = 120;

function recordComboVote(votes: SignalVote, actual: Reactor) {
  comboHistory.push({ votes, actual });
  if (comboHistory.length > MAX_COMBO_HISTORY) comboHistory.shift();
}

// Строит ключ из голосов (только активных сигналов): "P:alpha|F:omega|A:alpha"
function comboKey(votes: Partial<SignalVote>): string {
  return Object.entries(votes)
    .filter(([, v]) => v !== null)
    .map(([k, v]) => `${k[0]}:${v}`)
    .sort()
    .join("|");
}

// Ищет исходы для точного совпадения голосов, затем для подмножеств (fuzzy)
function detectComboSignal(
  currentVotes: SignalVote
): { reactor: Reactor; confidence: number; matchKey: string; sampleCount: number } | null {
  if (comboHistory.length < 8) return null;

  // Активные голоса (не null)
  const activeEntries = Object.entries(currentVotes).filter(([, v]) => v !== null) as [string, Reactor][];
  if (activeEntries.length < 2) return null;

  // Ищем от точного совпадения к менее точному
  const subsets: [string, Reactor][][] = [];
  // Все подмножества размером >= 2
  for (let mask = (1 << activeEntries.length) - 1; mask > 0; mask--) {
    const subset = activeEntries.filter((_, i) => (mask >> i) & 1);
    if (subset.length >= 2) subsets.push(subset);
  }
  // Сортируем по убыванию размера (сначала точные совпадения)
  subsets.sort((a, b) => b.length - a.length);

  for (const subset of subsets) {
    // Проверяем: все в подмножестве голосуют одинаково?
    const directions = subset.map(([, v]) => v);
    const allSame = directions.every(d => d === directions[0]);
    if (!allSame) continue; // смешанный консенсус — пропускаем

    const direction = directions[0];
    const key = comboKey(Object.fromEntries(subset));

    // Ищем совпадения в истории
    const matches = comboHistory.filter(entry => {
      return subset.every(([k, v]) => (entry.votes as Record<string, Reactor>)[k] === v);
    });

    if (matches.length < 4) continue;

    const hits = matches.filter(m => m.actual === direction).length;
    const acc = hits / matches.length;
    if (acc < 0.52) continue; // ниже 52% — не даём сигнал

    const sampleWeight = Math.min(matches.length / 15, 1);
    const confidence = (acc - 0.5) * 2 * sampleWeight; // 0.52→0.04, 0.70→0.40, 1.0→1.0
    if (confidence < 0.04) continue;

    return { reactor: direction, confidence, matchKey: key, sampleCount: matches.length };
  }

  return null;
}

// ── Обновление точности сигналов на основе последнего раунда ──

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
      signals: { patternScore: 0, flickerScore: 0, flickerPatternScore: 0, balanceScore: 0, adaptScore: 0, timeScore: 0, alternationScore: 0, streakFlickerScore: 0, comboScore: 0 },
    };
  }

  const patterns = findPatterns(reactorHistory);
  const flickerProfiles = buildFlickerProfiles(history);

  patterns.forEach(p => {
    const key = p.sequence.join(",") + "→" + p.next;
    p.flickerProfile = flickerProfiles.get(key) ?? null;
  });

  let bestPattern: Pattern | null = null;
  for (const len of SEQ_LENGTHS) {
    const tail = reactorHistory.slice(-len);
    if (tail.includes(null)) continue;
    const key = tail.join(",");
    const match = patterns.find(p => p.sequence.join(",") === key && p.confidence > 0.5);
    if (match) { bestPattern = match; break; }
  }

  const nextTimestamp = Date.now();
  const timeSignal = detectTimePeriodicity(history, nextTimestamp);
  const flicker = flickerBaseSignal(flickerBias, flickerRate, history, flickerSwitchCount, flickerAlphaRedness, flickerOmegaRedness);

  let alphaScore = 0;
  let omegaScore = 0;
  const signals: SignalBreakdown = {
    patternScore: 0, flickerScore: 0, flickerPatternScore: 0,
    balanceScore: 0, adaptScore: 0,
    timeScore: 0, comboScore: 0,
  };
  const reasonParts: string[] = [];

  // ── 1. Паттерн ────────────────────────────────────────
  if (bestPattern) {
    const lenW = 0.18 + bestPattern.sequence.length * 0.064;
    const mult = signalMultiplier("pattern");
    const patW = bestPattern.confidence * lenW * mult;
    if (bestPattern.next === "alpha") alphaScore += patW;
    else omegaScore += patW;
    signals.patternScore = patW;
    reasonParts.push(`паттерн ${bestPattern.label}(${Math.round(bestPattern.confidence * 100)}%)×${mult.toFixed(1)}`);

    const { bonus, reason: bonusReason } = flickerPatternBonus(bestPattern, flickerBias, flickerRate, flickerSwitchCount, flickerProfiles);
    if (bonus !== 0) {
      if (bestPattern.next === "alpha") alphaScore += bonus;
      else omegaScore += bonus;
      signals.flickerPatternScore = Math.abs(bonus);
      if (bonusReason) reasonParts.push(bonusReason);
    }
  }

  // ── 2. Баланс 50/50 ────────────────────────────────────
  const alphaTotal = validHistory.filter(r => r === "alpha").length;
  const alphaPct = validHistory.length > 0 ? alphaTotal / validHistory.length : 0.5;
  const balancePull = (0.5 - alphaPct) * BALANCE_PULL;
  alphaScore += balancePull;
  omegaScore -= balancePull;
  signals.balanceScore = Math.abs(balancePull);
  if (Math.abs(alphaPct - 0.5) > 0.1) {
    reasonParts.push(`баланс ${alphaPct > 0.5 ? "α" : "ω"} +${Math.round(Math.abs(alphaPct - 0.5) * 200)}%`);
  }

  // ── 3. Мерцание ────────────────────────────────────────
  if (flicker.hint) {
    const flickMult = signalMultiplier("flicker");
    const flickW = flicker.weight * flickMult;
    if (flicker.hint === "alpha") alphaScore += flickW;
    else omegaScore += flickW;
    signals.flickerScore = flickW;
    reasonParts.push(flicker.reason + (flickMult !== 1 ? ` ×${flickMult.toFixed(1)}` : ""));
  }

  // ── 4. Периодичность по времени ────────────────────────
  if (timeSignal) {
    const timeMult = signalMultiplier("time");
    // Усиленный базовый вес: 0.55 вместо 0.34 (сигнал показал 66% точность)
    const timeW = timeSignal.confidence * 0.55 * timeMult;
    if (timeSignal.reactor === "alpha") alphaScore += timeW;
    else omegaScore += timeW;
    signals.timeScore = timeW;
    reasonParts.push(`t%${timeSignal.periodMs}ms→${timeSignal.reactor === "alpha" ? "α" : "ω"}(${Math.round(timeSignal.confidence * 100)}%)×${timeMult.toFixed(1)}`);
  }

  // ── 5. Комбо: история совместных предсказаний ──────────
  const currentVotes: SignalVote = {
    pattern: bestPattern ? bestPattern.next : null,
    flicker: flicker.hint,
    time:    timeSignal ? timeSignal.reactor : null,
  };
  const combo = detectComboSignal(currentVotes);
  if (combo) {
    const comboMult = signalMultiplier("combo");
    const comboW = combo.confidence * 0.55 * comboMult;
    if (combo.reactor === "alpha") alphaScore += comboW;
    else omegaScore += comboW;
    signals.comboScore = comboW;
    reasonParts.push(`комбо[${combo.matchKey}]→${combo.reactor === "alpha" ? "α" : "ω"}(n=${combo.sampleCount},conf=${Math.round(combo.confidence * 100)}%)×${comboMult.toFixed(1)}`);
  }

  // ── 8. Адаптация: скользящий вес по последним 20 ──────
  const recent = history.slice(-20);
  const recentWithPred = recent.filter(r => r.predictionHit !== null);
  if (recentWithPred.length >= 5) {
    const recentHits = recentWithPred.filter(r => r.predictionHit).length;
    const acc = recentHits / recentWithPred.length;
    if (acc < 0.50) {
      const adaptW = (0.50 - acc) * 0.45;
      if (alphaScore > omegaScore) omegaScore += adaptW;
      else alphaScore += adaptW;
      signals.adaptScore = adaptW;
      reasonParts.push(`адапт.↔ (${Math.round(acc * 100)}% за 20)`);
    } else if (acc > 0.60) {
      const adaptW = (acc - 0.60) * 0.25;
      if (alphaScore > omegaScore) alphaScore += adaptW;
      else omegaScore += adaptW;
      signals.adaptScore = adaptW;
    }
  }

  // ── Финал ─────────────────────────────────────────────
  if (alphaScore === 0 && omegaScore === 0) { alphaScore = 0.5; omegaScore = 0.5; }
  const total = alphaScore + omegaScore;
  const alphaNorm = alphaScore / total;
  const reactor: Reactor = alphaNorm >= 0.5 ? "alpha" : "omega";
  const rawConf = Math.max(alphaNorm, 1 - alphaNorm);
  const confidence = Math.min(0.92, 0.5 + (rawConf - 0.5) * 1.6);

  // Записываем голоса раунда в историю combo (обучение)
  const lastResult = history[history.length - 1];
  if (lastResult?.winner) recordComboVote(currentVotes, lastResult.winner);

  // Сохраняем снапшот для оценки точности сигналов
  prevSnapshot = {
    pattern: bestPattern ? bestPattern.next : null,
    flicker: flicker.hint,
    time:    timeSignal ? timeSignal.reactor : null,
  };

  return {
    reactor, confidence,
    reason: reasonParts.length > 0 ? reasonParts.join(" · ") : "базовая статистика",
    patternMatch: bestPattern,
    flickerHint: flicker.hint,
    flickerWeight: flicker.weight,
    signals,
    timeSignal,
    streakFlickerSignal: null,
  };
}

// ── Топ паттернов ─────────────────────────────────────────
export function getTopPatterns(history: RoundResult[]): Pattern[] {
  return findPatterns(history.map(r => r.winner), MIN_PATTERN_COUNT).slice(0, 8);
}
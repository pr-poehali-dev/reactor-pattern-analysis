/**
 * NeuralPredictor v1 — самообучающийся ИИ-предсказатель
 *
 * Архитектура:
 * - Многослойная таблица Q-значений (state → action) с ε-greedy исследованием
 * - Автоматическое обнаружение признаков из истории побед + мерцания
 * - Рефлексия: ИИ анализирует свои ошибки и корректирует гипотезы
 * - "Мысли" — внутренний монолог, объясняющий ход рассуждений
 */

import type { Reactor, RoundResult } from "./screenAnalyzer";

// ── Типы ──────────────────────────────────────────────────

export interface AIThought {
  id: number;
  type: "observe" | "hypothesis" | "correct" | "doubt" | "confirm" | "discover";
  text: string;
  timestamp: number;
}

export interface AIPrediction {
  reactor: Reactor;
  confidence: number;
  thoughts: AIThought[];
  activeHypotheses: string[];
  dominantFeature: string;
  learningProgress: number;   // 0..1
}

// ── Типы признаков ────────────────────────────────────────

interface Feature {
  name: string;
  value: number;    // числовое значение
  label: string;    // читаемое описание
}

interface Hypothesis {
  id: string;
  description: string;
  // Предсказывает: если признак X, то победитель Y
  featureName: string;
  featureThreshold: number;
  featureDirection: "above" | "below";  // выше/ниже порога
  predictedWinner: Reactor;
  // Статистика
  hits: number;
  misses: number;
  weight: number;       // текущий вес гипотезы (обновляется онлайн)
  lastUsed: number;     // timestamp
  confirmed: boolean;
}

// ── Хранилище состояния ИИ (персистентное между вызовами) ─

interface AIState {
  hypotheses: Map<string, Hypothesis>;
  thoughtLog: AIThought[];
  thoughtIdCounter: number;
  epsilon: number;
  totalRounds: number;
  correctPredictions: number;
  featureHistory: Feature[][];
  lastPrediction: Reactor;
}

let aiState: AIState = createInitialState();

function createInitialState(): AIState {
  return {
    hypotheses: new Map(),
    thoughtLog: [],
    thoughtIdCounter: 0,
    epsilon: 0.3,
    totalRounds: 0,
    correctPredictions: 0,
    featureHistory: [],
    lastPrediction: null,
  };
}

export function resetAI() {
  aiState = createInitialState();
}

// ── Ключ localStorage ─────────────────────────────────────
const MEMORY_KEY = "reactoros_ai_memory";

interface PersistedMemory {
  hypotheses: Array<[string, Hypothesis]>;
  epsilon: number;
  totalRounds: number;
  correctPredictions: number;
  savedAt: number;
}

export function saveMemory(): void {
  const mem: PersistedMemory = {
    hypotheses: [...aiState.hypotheses.entries()],
    epsilon: aiState.epsilon,
    totalRounds: aiState.totalRounds,
    correctPredictions: aiState.correctPredictions,
    savedAt: Date.now(),
  };
  localStorage.setItem(MEMORY_KEY, JSON.stringify(mem));
}

export function loadMemory(): { ok: boolean; rounds: number; hypothesesCount: number; savedAt: number | null } {
  const raw = localStorage.getItem(MEMORY_KEY);
  if (!raw) return { ok: false, rounds: 0, hypothesesCount: 0, savedAt: null };
  try {
    const mem: PersistedMemory = JSON.parse(raw);
    aiState.hypotheses = new Map(mem.hypotheses);
    aiState.epsilon = mem.epsilon ?? 0.1;
    aiState.totalRounds = mem.totalRounds ?? 0;
    aiState.correctPredictions = mem.correctPredictions ?? 0;
    return { ok: true, rounds: mem.totalRounds, hypothesesCount: mem.hypotheses.length, savedAt: mem.savedAt };
  } catch {
    return { ok: false, rounds: 0, hypothesesCount: 0, savedAt: null };
  }
}

export function clearMemory(): void {
  localStorage.removeItem(MEMORY_KEY);
  aiState = createInitialState();
}

export function hasSavedMemory(): boolean {
  return !!localStorage.getItem(MEMORY_KEY);
}

export function getSavedMemoryMeta(): { rounds: number; hypothesesCount: number; savedAt: number } | null {
  const raw = localStorage.getItem(MEMORY_KEY);
  if (!raw) return null;
  try {
    const mem: PersistedMemory = JSON.parse(raw);
    return { rounds: mem.totalRounds ?? 0, hypothesesCount: mem.hypotheses?.length ?? 0, savedAt: mem.savedAt };
  } catch { return null; }
}

// ── Генератор мыслей ──────────────────────────────────────

function think(type: AIThought["type"], text: string): AIThought {
  const thought: AIThought = {
    id: aiState.thoughtIdCounter++,
    type,
    text,
    timestamp: Date.now(),
  };
  aiState.thoughtLog.push(thought);
  // Держим последние 60 мыслей
  if (aiState.thoughtLog.length > 60) aiState.thoughtLog = aiState.thoughtLog.slice(-60);
  return thought;
}

// ── Извлечение признаков из состояния ────────────────────

function extractFeatures(history: RoundResult[], flickerBias: number, flickerRate: number, flickerSwitchCount: number): Feature[] {
  const n = history.length;
  const features: Feature[] = [];

  // 1. Последние победители (hot-encoding)
  const last1 = n >= 1 ? history[n - 1].winner : null;
  const last2 = n >= 2 ? history[n - 2].winner : null;
  const last3 = n >= 3 ? history[n - 3].winner : null;
  const last4 = n >= 4 ? history[n - 4].winner : null;

  features.push({ name: "last1_alpha", value: last1 === "alpha" ? 1 : 0, label: "прошлый = α" });
  features.push({ name: "last2_alpha", value: last2 === "alpha" ? 1 : 0, label: "позапрошлый = α" });
  features.push({ name: "last3_alpha", value: last3 === "alpha" ? 1 : 0, label: "3й назад = α" });
  features.push({ name: "last4_alpha", value: last4 === "alpha" ? 1 : 0, label: "4й назад = α" });

  // 2. Серия одной стороны
  let streak = 0;
  let streakSide: Reactor = null;
  for (let i = n - 1; i >= 0; i--) {
    if (i === n - 1) { streakSide = history[i].winner; streak = 1; }
    else if (history[i].winner === streakSide) streak++;
    else break;
  }
  features.push({ name: "streak_len", value: streak, label: `серия ${streak}` });
  features.push({ name: "streak_alpha", value: streakSide === "alpha" ? 1 : 0, label: streakSide === "alpha" ? "серия у α" : "серия у ω" });

  // 3. Баланс за последние 10 раундов
  const last10 = history.slice(-10);
  const alphaRatio10 = last10.length > 0 ? last10.filter(r => r.winner === "alpha").length / last10.length : 0.5;
  features.push({ name: "alpha_ratio_10", value: alphaRatio10, label: `α доля за 10: ${Math.round(alphaRatio10 * 100)}%` });

  // 4. Чередование (насколько часто меняется победитель)
  let alternations = 0;
  for (let i = 1; i < Math.min(n, 8); i++) {
    if (history[n - i].winner !== history[n - i - 1].winner) alternations++;
  }
  const altRate = Math.min(n - 1, 7) > 0 ? alternations / Math.min(n - 1, 7) : 0;
  features.push({ name: "alt_rate", value: altRate, label: `чередование: ${Math.round(altRate * 100)}%` });

  // 5. Мерцание
  features.push({ name: "flicker_bias", value: flickerBias, label: `смещение: ${flickerBias > 0 ? "α чаще" : "ω чаще"} (${Math.abs(flickerBias * 100).toFixed(0)}%)` });
  features.push({ name: "flicker_rate", value: flickerRate, label: `темп: ${flickerRate.toFixed(1)}/сек` });
  features.push({ name: "flicker_switches", value: flickerSwitchCount, label: `переключений: ${flickerSwitchCount}` });

  // 6. Мерцание последних раундов (было ли сильное)
  const recentFlickerBias = n >= 3 ? history.slice(-3).reduce((s, r) => s + Math.abs(r.flickerBias), 0) / 3 : 0;
  features.push({ name: "recent_flicker_strength", value: recentFlickerBias, label: `сила мерцания (3 ранд): ${recentFlickerBias.toFixed(2)}` });

  // 7. Корреляция мерцания с победителем (исторически)
  if (n >= 4) {
    const withFlicker = history.slice(-10).filter(r => Math.abs(r.flickerBias) > 0.05);
    const biasAlpha = withFlicker.filter(r => r.flickerBias < 0 && r.winner === "alpha").length;
    const biasTotal = withFlicker.filter(r => r.flickerBias < 0).length;
    const flickerToAlphaAcc = biasTotal > 0 ? biasAlpha / biasTotal : 0.5;
    features.push({ name: "flicker_to_alpha_acc", value: flickerToAlphaAcc, label: `точность мерц→α: ${Math.round(flickerToAlphaAcc * 100)}%` });
  }

  return features;
}

// ── Генерация гипотез на основе данных ───────────────────

function generateHypotheses(history: RoundResult[], features: Feature[]): string[] {
  const generated: string[] = [];
  const n = history.length;
  if (n < 4) return generated;

  // Для каждого числового признака — проверяем корреляцию с победителем
  const featureNames = features.map(f => f.name);

  for (const feat of features) {
    if (aiState.hypotheses.has(`h_${feat.name}_high_alpha`) || aiState.hypotheses.has(`h_${feat.name}_low_alpha`)) continue;

    // Ищем примеры в сохранённой истории признаков
    if (aiState.featureHistory.length < 4) continue;

    const paired = aiState.featureHistory.slice(-20).map((fList, idx) => {
      const histIdx = (aiState.featureHistory.length - 20 > 0 ? aiState.featureHistory.length - 20 : 0) + idx;
      const round = history[histIdx];
      const f = fList.find(f2 => f2.name === feat.name);
      return round && f ? { value: f.value, winner: round.winner } : null;
    }).filter(Boolean) as { value: number; winner: Reactor }[];

    if (paired.length < 4) continue;

    const median = paired.map(p => p.value).sort((a, b) => a - b)[Math.floor(paired.length / 2)];

    // Высокое значение → alpha?
    const highSamples = paired.filter(p => p.value >= median);
    const lowSamples = paired.filter(p => p.value < median);
    const highAlphaRate = highSamples.length > 0 ? highSamples.filter(p => p.winner === "alpha").length / highSamples.length : 0.5;
    const lowAlphaRate = lowSamples.length > 0 ? lowSamples.filter(p => p.winner === "alpha").length / lowSamples.length : 0.5;

    if (Math.abs(highAlphaRate - 0.5) >= 0.2 && highSamples.length >= 3) {
      const winner: Reactor = highAlphaRate > 0.5 ? "alpha" : "omega";
      const id = `h_${feat.name}_high_${winner}`;
      if (!aiState.hypotheses.has(id)) {
        aiState.hypotheses.set(id, {
          id,
          description: `${feat.label} (выс.) → ${winner === "alpha" ? "α" : "ω"}`,
          featureName: feat.name,
          featureThreshold: median,
          featureDirection: "above",
          predictedWinner: winner,
          hits: 0, misses: 0, weight: 0.5, lastUsed: 0, confirmed: false,
        });
        generated.push(id);
        think("discover", `Новая гипотеза: если «${feat.label}» высокое — вероятно победит ${winner === "alpha" ? "α Альфа" : "ω Омега"}`);
      }
    }

    if (Math.abs(lowAlphaRate - 0.5) >= 0.2 && lowSamples.length >= 3) {
      const winner: Reactor = lowAlphaRate > 0.5 ? "alpha" : "omega";
      const id = `h_${feat.name}_low_${winner}`;
      if (!aiState.hypotheses.has(id)) {
        aiState.hypotheses.set(id, {
          id,
          description: `${feat.label} (низк.) → ${winner === "alpha" ? "α" : "ω"}`,
          featureName: feat.name,
          featureThreshold: median,
          featureDirection: "below",
          predictedWinner: winner,
          hits: 0, misses: 0, weight: 0.5, lastUsed: 0, confirmed: false,
        });
        generated.push(id);
        think("discover", `Новая гипотеза: если «${feat.label}» низкое — вероятно победит ${winner === "alpha" ? "α Альфа" : "ω Омега"}`);
      }
    }
  }

  return generated;
}

// ── Обновление весов гипотез после известного результата ─

function updateHypotheses(features: Feature[], winner: Reactor) {
  let bestHit: Hypothesis | null = null;
  let worstMiss: Hypothesis | null = null;

  aiState.hypotheses.forEach(h => {
    const feat = features.find(f => f.name === h.featureName);
    if (!feat) return;

    const triggered =
      (h.featureDirection === "above" && feat.value >= h.featureThreshold) ||
      (h.featureDirection === "below" && feat.value < h.featureThreshold);

    if (!triggered) return;

    if (h.predictedWinner === winner) {
      h.hits++;
      h.weight = Math.min(h.weight + 0.07, 0.95);
      if (!bestHit || h.weight > bestHit.weight) bestHit = h;
    } else {
      h.misses++;
      h.weight = Math.max(h.weight - 0.1, 0.05);
      if (!worstMiss || h.weight < worstMiss.weight) worstMiss = h;
    }

    const total = h.hits + h.misses;
    h.confirmed = total >= 5 && h.hits / total >= 0.65;
  });

  // Рефлексия
  if (bestHit) {
    const acc = bestHit.hits / (bestHit.hits + bestHit.misses);
    if (acc >= 0.7 && !bestHit.confirmed) {
      think("confirm", `Гипотеза «${bestHit.description}» подтверждается — точность ${Math.round(acc * 100)}%`);
    }
  }
  if (worstMiss && worstMiss.weight < 0.2) {
    think("doubt", `Гипотеза «${worstMiss.description}» не работает — отключаю (вес ${worstMiss.weight.toFixed(2)})`);
    aiState.hypotheses.delete(worstMiss.id);
  }
}

// ── Голосование гипотезами ────────────────────────────────

function voteHypotheses(features: Feature[]): { reactor: Reactor; score: number; active: Hypothesis[] } {
  let alphaScore = 0;
  let omegaScore = 0;
  const active: Hypothesis[] = [];

  aiState.hypotheses.forEach(h => {
    if (h.weight < 0.15) return;  // мёртвые гипотезы

    const feat = features.find(f => f.name === h.featureName);
    if (!feat) return;

    const triggered =
      (h.featureDirection === "above" && feat.value >= h.featureThreshold) ||
      (h.featureDirection === "below" && feat.value < h.featureThreshold);

    if (!triggered) return;

    const vote = h.weight;
    if (h.predictedWinner === "alpha") alphaScore += vote;
    else omegaScore += vote;

    h.lastUsed = Date.now();
    active.push(h);
  });

  const total = alphaScore + omegaScore;
  if (total === 0) return { reactor: null, score: 0.5, active: [] };

  const reactor: Reactor = alphaScore >= omegaScore ? "alpha" : "omega";
  const score = Math.max(alphaScore, omegaScore) / total;
  return { reactor, score, active };
}

// ── Прямой анализ паттернов (вспомогательный) ────────────

function directPatternAnalysis(history: RoundResult[]): { reactor: Reactor; confidence: number; note: string } {
  const n = history.length;
  if (n < 3) return { reactor: null, confidence: 0, note: "" };

  const wins = history.map(r => r.winner);

  // Проверяем паттерны длиной 4→3→2 с учётом давности
  for (const len of [4, 3, 2]) {
    if (n < len + 1) continue;
    const tail = wins.slice(-len);
    if (tail.includes(null)) continue;

    let alphaW = 0, omegaW = 0;
    for (let i = 0; i <= n - len - 1; i++) {
      const seq = wins.slice(i, i + len);
      if (seq.some(v => v === null)) continue;
      if (seq.every((v, j) => v === tail[j])) {
        const next = wins[i + len];
        const age = n - 1 - (i + len);
        const w = Math.exp(-age / 15);
        if (next === "alpha") alphaW += w;
        else if (next === "omega") omegaW += w;
      }
    }

    const total = alphaW + omegaW;
    if (total === 0) continue;

    const reactor: Reactor = alphaW >= omegaW ? "alpha" : "omega";
    const conf = Math.max(alphaW, omegaW) / total;
    if (conf > 0.55) {
      return { reactor, confidence: conf, note: `паттерн len=${len} (${Math.round(conf * 100)}%)` };
    }
  }

  return { reactor: null, confidence: 0, note: "" };
}

// ── Главная функция предсказания ──────────────────────────

export function aiPredict(
  history: RoundResult[],
  flickerBias: number,
  flickerRate: number,
  flickerSwitchCount = 0,
  actualWinner?: Reactor   // передаётся после раунда для обучения
): AIPrediction {
  const roundThoughts: AIThought[] = [];
  const n = history.length;

  // ── 1. Обучение на предыдущем результате ─────────────────
  if (actualWinner && aiState.featureHistory.length > 0) {
    const prevFeatures = aiState.featureHistory[aiState.featureHistory.length - 1];
    updateHypotheses(prevFeatures, actualWinner);
    aiState.totalRounds++;
    const lastPred = aiState.lastPrediction;
    if (lastPred && lastPred === actualWinner) {
      aiState.correctPredictions++;
      if (aiState.totalRounds % 3 === 0) {
        const acc = aiState.correctPredictions / aiState.totalRounds;
        think("confirm", `Точность за ${aiState.totalRounds} раундов: ${Math.round(acc * 100)}% — продолжаю обучение`);
      }
    } else if (lastPred) {
      think("correct", `Ошибся: предсказал ${lastPred === "alpha" ? "α" : "ω"}, победил ${actualWinner === "alpha" ? "α" : "ω"} — корректирую гипотезы`);
      // Уменьшаем epsilon медленнее при ошибках
      aiState.epsilon = Math.min(aiState.epsilon + 0.03, 0.35);
    }
  }

  // ── 2. Извлечение признаков ───────────────────────────────
  const features = extractFeatures(history, flickerBias, flickerRate, flickerSwitchCount);
  aiState.featureHistory.push(features);
  if (aiState.featureHistory.length > 100) aiState.featureHistory = aiState.featureHistory.slice(-100);

  // ── 3. Генерация новых гипотез ────────────────────────────
  if (n >= 4) generateHypotheses(history, features);

  // ── 4. Голосование ───────────────────────────────────────
  const { reactor: voteReactor, score: voteScore, active } = voteHypotheses(features);

  // ── 5. Прямой анализ паттернов ────────────────────────────
  const { reactor: patReactor, confidence: patConf, note: patNote } = directPatternAnalysis(history);

  // ── 6. Сигнал мерцания ───────────────────────────────────
  let flickerReactor: Reactor = null;
  let flickerWeight = 0;
  if (Math.abs(flickerBias) > 0.08 && flickerRate > 0.4) {
    // Проверяем историческую точность этого правила
    const withFlicker = history.slice(-15).filter(r => Math.abs(r.flickerBias) > 0.05 && r.flickerRate > 0.4);
    const rule: Reactor = flickerBias > 0 ? "omega" : "alpha"; // антикорреляция
    if (withFlicker.length >= 3) {
      const acc = withFlicker.filter(r => {
        const pred: Reactor = r.flickerBias > 0 ? "omega" : "alpha";
        return pred === r.winner;
      }).length / withFlicker.length;
      flickerWeight = (acc - 0.4) * 1.5 * Math.min(flickerRate / 3, 1);
      flickerReactor = flickerWeight > 0 ? rule : (rule === "alpha" ? "omega" : "alpha");
      flickerWeight = Math.abs(flickerWeight);
    } else {
      flickerReactor = rule;
      flickerWeight = 0.15;
    }
  }

  // ── 7. Ансамбль ──────────────────────────────────────────
  let alphaTotal = 0;
  let omegaTotal = 0;

  // Голосование гипотезами (вес зависит от числа активных)
  const hypWeight = Math.min(0.5, active.length * 0.08 + 0.1);
  if (voteReactor === "alpha") alphaTotal += voteScore * hypWeight;
  else if (voteReactor === "omega") omegaTotal += voteScore * hypWeight;

  // Паттерн
  if (patReactor === "alpha") alphaTotal += patConf * 0.4;
  else if (patReactor === "omega") omegaTotal += patConf * 0.4;

  // Мерцание
  if (flickerReactor === "alpha") alphaTotal += flickerWeight * 0.3;
  else if (flickerReactor === "omega") omegaTotal += flickerWeight * 0.3;

  // ── 8. Итог ───────────────────────────────────────────────
  let finalReactor: Reactor = null;
  let rawConf = 0.5;

  if (n >= 2) {
    const totalScore = alphaTotal + omegaTotal;
    if (totalScore > 0) {
      finalReactor = alphaTotal >= omegaTotal ? "alpha" : "omega";
      rawConf = Math.max(alphaTotal, omegaTotal) / totalScore;
    } else if (patReactor) {
      finalReactor = patReactor;
      rawConf = patConf;
    }
  }

  // ── 9. ε-greedy: иногда исследуем ────────────────────────
  const exploring = Math.random() < aiState.epsilon && n >= 3;
  if (exploring && finalReactor) {
    const was = finalReactor;
    finalReactor = finalReactor === "alpha" ? "omega" : "alpha";
    rawConf = 1 - rawConf;
    think("observe", `Исследую противоположную гипотезу (ε=${aiState.epsilon.toFixed(2)})`);
  }

  // Уменьшаем ε со временем
  aiState.epsilon = Math.max(0.05, aiState.epsilon * 0.995);

  // Нормализуем уверенность
  const confidence = finalReactor ? Math.min(0.91, 0.5 + (rawConf - 0.5) * 1.4) : 0;

  // ── 10. Генерируем мысли для этого раунда ────────────────
  const sessionThoughts: AIThought[] = [];

  if (n < 2) {
    sessionThoughts.push(think("observe", "Накапливаю данные. Нужно минимум 2 раунда для гипотез."));
  } else {
    // Наблюдение о текущем мерцании
    if (flickerRate > 1.5) {
      sessionThoughts.push(think("observe",
        `Интенсивное мерцание: ${flickerRate.toFixed(1)} переключений/сек. ${flickerBias > 0.1 ? "α доминирует" : flickerBias < -0.1 ? "ω доминирует" : "стороны равны"}.`
      ));
    }

    // Сообщаем о активных гипотезах
    const confirmed = [...aiState.hypotheses.values()].filter(h => h.confirmed);
    if (confirmed.length > 0) {
      const best = confirmed.sort((a, b) => b.weight - a.weight)[0];
      sessionThoughts.push(think("hypothesis",
        `Сильная гипотеза: «${best.description}» — вес ${best.weight.toFixed(2)}, точность ${Math.round(best.hits / (best.hits + best.misses) * 100)}%`
      ));
    }

    // Паттерн
    if (patReactor && patConf > 0.6) {
      sessionThoughts.push(think("observe",
        `Паттерн: ${patNote}. Голосует за ${patReactor === "alpha" ? "α Альфа" : "ω Омега"}.`
      ));
    }

    // Итоговое решение
    if (finalReactor) {
      const accSoFar = aiState.totalRounds > 0 ? Math.round(aiState.correctPredictions / aiState.totalRounds * 100) : "—";
      sessionThoughts.push(think("hypothesis",
        `Прогноз: ${finalReactor === "alpha" ? "α Альфа" : "ω Омега"} (уверен на ${Math.round(confidence * 100)}%). Общая точность: ${accSoFar}%.`
      ));
    }
  }

  // Сохраняем предсказание для обучения в следующем раунде
  aiState.lastPrediction = finalReactor;

  // Активные гипотезы (описания)
  const activeHypotheses = active.slice(0, 4).map(h =>
    `${h.description} [${h.weight.toFixed(2)}]`
  );

  // Определяем доминирующий признак
  let dominantFeature = "нет данных";
  if (active.length > 0) {
    const top = active.sort((a, b) => b.weight - a.weight)[0];
    dominantFeature = top.description;
  } else if (patReactor) {
    dominantFeature = patNote || "паттерн последовательности";
  }

  const learningProgress = Math.min(n / 20, 1);

  return {
    reactor: finalReactor,
    confidence,
    thoughts: [...aiState.thoughtLog.slice(-8)],
    activeHypotheses,
    dominantFeature,
    learningProgress,
  };
}

// ── Экспорт истории мыслей ────────────────────────────────

export function getThoughtLog(): AIThought[] {
  return [...aiState.thoughtLog];
}

export function getHypotheses(): Hypothesis[] {
  return [...aiState.hypotheses.values()].sort((a, b) => b.weight - a.weight);
}

// ML-предсказатель на основе паттернов последовательностей
import type { Reactor, RoundResult } from "./screenAnalyzer";

export interface Pattern {
  sequence: Reactor[];
  next: Reactor;
  count: number;
  confidence: number;
  label: string;
}

export interface Prediction {
  reactor: Reactor;
  confidence: number;
  reason: string;
  patternMatch: Pattern | null;
  flickerHint: Reactor | null;
  flickerWeight: number;
}

const SEQ_LENGTHS = [5, 4, 3, 2];

// Ищет совпадающие паттерны в истории
export function findPatterns(history: Reactor[], minCount = 2): Pattern[] {
  const patterns: Map<string, { next: Record<string, number>; total: number }> = new Map();

  for (const len of SEQ_LENGTHS) {
    for (let i = 0; i <= history.length - len - 1; i++) {
      const seq = history.slice(i, i + len);
      const next = history[i + len];
      if (seq.includes(null) || next === null) continue;

      const key = seq.join(",");
      if (!patterns.has(key)) patterns.set(key, { next: {}, total: 0 });
      const p = patterns.get(key)!;
      p.next[next as string] = (p.next[next as string] ?? 0) + 1;
      p.total++;
    }
  }

  const result: Pattern[] = [];

  patterns.forEach((data, key) => {
    if (data.total < minCount) return;
    const seq = key.split(",") as Reactor[];
    const entries = Object.entries(data.next).sort((a, b) => b[1] - a[1]);
    const [bestNext, bestCount] = entries[0];
    const confidence = bestCount / data.total;

    const labels: Record<string, string> = {
      "alpha,alpha,alpha,alpha,alpha": "AAAAA",
      "omega,omega,omega,omega,omega": "OOOOO",
      "alpha,omega,alpha,omega,alpha": "AOAOA",
      "omega,alpha,omega,alpha,omega": "OAOAO",
    };

    result.push({
      sequence: seq,
      next: bestNext as Reactor,
      count: data.total,
      confidence,
      label: labels[key] ?? seq.map(r => r === "alpha" ? "A" : "O").join(""),
    });
  });

  return result.sort((a, b) => b.confidence * b.count - a.confidence * a.count);
}

// Основное предсказание
export function predict(
  history: RoundResult[],
  flickerBias: number,
  flickerRate: number
): Prediction {
  const reactorHistory = history.map(r => r.winner);

  if (history.length < 2) {
    return {
      reactor: null,
      confidence: 0,
      reason: "Недостаточно данных",
      patternMatch: null,
      flickerHint: null,
      flickerWeight: 0,
    };
  }

  const patterns = findPatterns(reactorHistory);

  // Ищем паттерн, соответствующий концу истории
  let bestPattern: Pattern | null = null;
  for (const len of SEQ_LENGTHS) {
    const tail = reactorHistory.slice(-len);
    if (tail.includes(null)) continue;
    const key = tail.join(",");
    const match = patterns.find(p => p.sequence.join(",") === key);
    if (match && match.confidence >= 0.5) {
      bestPattern = match;
      break;
    }
  }

  // Hint от мерцания: кто мерцал меньше — тот победит (последний жёлтый → победитель)
  // bias > 0 = alpha мерцала чаще → omega победит (и наоборот)
  let flickerHint: Reactor = null;
  let flickerWeight = 0;

  if (Math.abs(flickerBias) > 0.1 && flickerRate > 0.5) {
    flickerHint = flickerBias > 0 ? "omega" : "alpha";
    flickerWeight = Math.min(Math.abs(flickerBias) * 0.6, 0.4);
  }

  // Финальный расчёт
  let alphaScore = 0;
  let omegaScore = 0;

  if (bestPattern) {
    if (bestPattern.next === "alpha") alphaScore += bestPattern.confidence * (1 - flickerWeight);
    else omegaScore += bestPattern.confidence * (1 - flickerWeight);
  } else {
    // Базовая статистика
    const alphaWins = reactorHistory.filter(r => r === "alpha").length;
    const total = reactorHistory.filter(r => r !== null).length;
    const alphaPct = total > 0 ? alphaWins / total : 0.5;
    alphaScore += alphaPct * 0.4;
    omegaScore += (1 - alphaPct) * 0.4;
  }

  if (flickerHint === "alpha") alphaScore += flickerWeight;
  else if (flickerHint === "omega") omegaScore += flickerWeight;

  const reactor: Reactor = alphaScore >= omegaScore ? "alpha" : "omega";
  const rawConf = Math.max(alphaScore, omegaScore);
  const confidence = Math.min(0.95, 0.5 + rawConf * 0.5);

  let reason = "";
  if (bestPattern) {
    reason = `Паттерн «${bestPattern.label}» (${Math.round(bestPattern.confidence * 100)}%)`;
    if (flickerHint) reason += ` + мерцание → ${flickerHint === "alpha" ? "Альфа" : "Омега"}`;
  } else if (flickerHint) {
    reason = `Мерцание: ${flickerHint === "alpha" ? "Альфа" : "Омега"} мерцала чаще`;
  } else {
    reason = "Базовая статистика";
  }

  return { reactor, confidence, reason, patternMatch: bestPattern, flickerHint, flickerWeight };
}

// Все найденные паттерны с частотой
export function getTopPatterns(history: RoundResult[]): Pattern[] {
  return findPatterns(history.map(r => r.winner), 2).slice(0, 8);
}

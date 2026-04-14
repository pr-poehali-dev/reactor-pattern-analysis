/**
 * Метапредиктор: анализирует историю консенсусов ML и ИИ,
 * учится на их совместных ошибках и исправляет прогноз.
 *
 * Логика:
 * 1. Смотрим на каждый прошлый раунд: были ли ML и ИИ согласны (консенсус)?
 * 2. Если консенсус — чем это заканчивалось? Победой консенсуса или ошибкой?
 * 3. Текущее состояние: ML и ИИ сейчас согласны?
 *    - Если да → применяем накопленный коэффициент доверия к консенсусу
 *    - Если расходятся → выбираем того, у кого выше точность при расхождениях
 */

import type { RoundResult } from "./screenAnalyzer";

export type Reactor = "alpha" | "omega" | null;

export interface MetaPrediction {
  reactor: Reactor;
  confidence: number;
  mode: "consensus_trusted" | "consensus_inverted" | "divergence" | "insufficient";
  consensusAccuracy: number | null;   // точность консенсуса исторически
  divergenceWinner: "ml" | "ai" | null; // кто точнее при расхождениях
  sampleCount: number;                // на скольких раундах обучен
  explanation: string;
}

interface ConsensusRecord {
  agreed: boolean;
  agreeOn: Reactor;      // на чём сошлись (если agreed)
  mlPred: Reactor;
  aiPred: Reactor;
  actual: Reactor;
  mlWon: boolean | null;
  aiWon: boolean | null;
}

function buildRecords(history: RoundResult[]): ConsensusRecord[] {
  return history
    .filter(r => r.predictedBefore !== null || r.aiPredictedBefore !== null)
    .map(r => {
      const ml = r.predictedBefore;
      const ai = r.aiPredictedBefore;
      const actual = r.winner;
      const agreed = ml !== null && ai !== null && ml === ai;
      return {
        agreed,
        agreeOn: agreed ? ml : null,
        mlPred: ml,
        aiPred: ai,
        actual,
        mlWon: ml !== null ? ml === actual : null,
        aiWon: ai !== null ? ai === actual : null,
      };
    });
}

export function metaPredict(
  history: RoundResult[],
  mlReactor: Reactor,
  aiReactor: Reactor
): MetaPrediction {
  const records = buildRecords(history);

  const MIN_SAMPLES = 4;

  // --- Статистика консенсусов ---
  const consensusRounds = records.filter(r => r.agreed);
  const consensusHits = consensusRounds.filter(r => r.agreeOn === r.actual).length;
  const consensusAcc = consensusRounds.length >= MIN_SAMPLES
    ? consensusHits / consensusRounds.length
    : null;

  // --- Статистика расхождений ---
  const divergeRounds = records.filter(r => !r.agreed && r.mlPred !== null && r.aiPred !== null);
  const mlDivergeHits = divergeRounds.filter(r => r.mlWon).length;
  const aiDivergeHits = divergeRounds.filter(r => r.aiWon).length;
  const divergenceWinner: "ml" | "ai" | null =
    divergeRounds.length >= MIN_SAMPLES
      ? mlDivergeHits >= aiDivergeHits ? "ml" : "ai"
      : null;

  const sampleCount = records.length;

  // --- Текущее состояние ---
  const currentlyAgree = mlReactor !== null && aiReactor !== null && mlReactor === aiReactor;

  // Недостаточно данных
  if (sampleCount < MIN_SAMPLES) {
    return {
      reactor: mlReactor ?? aiReactor,
      confidence: 0,
      mode: "insufficient",
      consensusAccuracy: null,
      divergenceWinner: null,
      sampleCount,
      explanation: `Мало данных (${sampleCount}/${MIN_SAMPLES} раундов)`,
    };
  }

  // --- Режим: консенсус ---
  if (currentlyAgree && consensusAcc !== null) {
    if (consensusAcc >= 0.55) {
      // Консенсус исторически надёжен — доверяем
      return {
        reactor: mlReactor,
        confidence: consensusAcc,
        mode: "consensus_trusted",
        consensusAccuracy: consensusAcc,
        divergenceWinner,
        sampleCount,
        explanation: `Оба согласны. Исторически консенсус верен в ${Math.round(consensusAcc * 100)}% — доверяем`,
      };
    } else {
      // Консенсус чаще ошибается — инвертируем
      const inverted: Reactor = mlReactor === "alpha" ? "omega" : "alpha";
      return {
        reactor: inverted,
        confidence: 1 - consensusAcc,
        mode: "consensus_inverted",
        consensusAccuracy: consensusAcc,
        divergenceWinner,
        sampleCount,
        explanation: `Оба согласны, но исторически консенсус верен лишь в ${Math.round(consensusAcc * 100)}% — инвертируем`,
      };
    }
  }

  // --- Режим: расхождение ---
  if (!currentlyAgree && mlReactor !== null && aiReactor !== null) {
    const pick = divergenceWinner === "ai" ? aiReactor : mlReactor;
    const pickAcc = divergenceWinner === "ai"
      ? (divergeRounds.length > 0 ? aiDivergeHits / divergeRounds.length : 0.5)
      : (divergeRounds.length > 0 ? mlDivergeHits / divergeRounds.length : 0.5);

    return {
      reactor: pick,
      confidence: pickAcc,
      mode: "divergence",
      consensusAccuracy: consensusAcc,
      divergenceWinner,
      sampleCount,
      explanation: `Мнения разошлись. При расхождениях точнее ${divergenceWinner === "ai" ? "ИИ" : "ML"} (${Math.round(pickAcc * 100)}%)`,
    };
  }

  // Fallback
  return {
    reactor: mlReactor ?? aiReactor,
    confidence: 0.5,
    mode: "insufficient",
    consensusAccuracy: consensusAcc,
    divergenceWinner,
    sampleCount,
    explanation: "Недостаточно данных для метапрогноза",
  };
}

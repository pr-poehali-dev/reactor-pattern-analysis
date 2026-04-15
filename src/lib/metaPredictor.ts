/**
 * Метапредиктор: анализирует историю пар (ML, ИИ) с учётом уровней уверенности.
 *
 * Вместо бинарного «согласны / расходятся» — разбивает историю на профили:
 *   • Профиль = (mlBucket, aiBucket, agreed)
 *   • Bucket: "high" (>70%), "mid" (50–70%), "low" (<50%)
 *
 * Для каждого профиля считается своя точность и принимается своё решение.
 * Это позволяет учесть разницу между «ML 92% + ИИ 48%» и «ML 60% + ИИ 55%».
 */

import type { RoundResult } from "./screenAnalyzer";

export type Reactor = "alpha" | "omega" | null;
export type ConfBucket = "high" | "mid" | "low";

export type MetaMode =
  | "profile_trusted"    // профиль исторически надёжен — доверяем лидеру
  | "profile_inverted"   // профиль исторически ошибается — инвертируем
  | "profile_pick_ml"    // профиль-расхождение: исторически точнее ML
  | "profile_pick_ai"    // профиль-расхождение: исторически точнее ИИ
  | "insufficient";      // мало данных

export interface MetaPrediction {
  reactor: Reactor;
  confidence: number;
  mode: MetaMode;
  profileKey: string;           // описание текущего профиля, напр. "ML высокий · ИИ низкий · согласны"
  profileAccuracy: number | null; // точность этого профиля в истории
  profileSamples: number;       // сколько раундов с таким профилем в истории
  sampleCount: number;          // всего раундов в истории
  explanation: string;
}

function bucket(conf: number): ConfBucket {
  if (conf >= 0.70) return "high";
  if (conf >= 0.50) return "mid";
  return "low";
}

function bucketLabel(b: ConfBucket): string {
  return b === "high" ? "высокий" : b === "mid" ? "средний" : "низкий";
}

function profileKey(mlB: ConfBucket, aiB: ConfBucket, agreed: boolean): string {
  return `${mlB}|${aiB}|${agreed ? "agree" : "disagree"}`;
}

function profileLabel(mlB: ConfBucket, aiB: ConfBucket, agreed: boolean): string {
  return `ML ${bucketLabel(mlB)} · ИИ ${bucketLabel(aiB)} · ${agreed ? "согласны" : "расходятся"}`;
}

interface ProfileStat {
  total: number;
  // для согласных: побед консенсуса
  consensusHits: number;
  // для расходящихся: побед ML и ИИ
  mlHits: number;
  aiHits: number;
  // на кого ставить при данном профиле (лидер по conf)
  dominantIsML: boolean; // true = ML уверенней, false = ИИ уверенней
}

const MIN_PROFILE_SAMPLES = 3;
const MIN_TOTAL_SAMPLES = 4;

export function metaPredict(
  history: RoundResult[],
  mlReactor: Reactor,
  mlConfidence: number,
  aiReactor: Reactor,
  aiConfidence: number,
): MetaPrediction {
  const records = history.filter(
    r => r.predictedBefore !== null && r.aiPredictedBefore !== null
  );

  const sampleCount = records.length;

  const mlB = bucket(mlConfidence);
  const aiB = bucket(aiConfidence);
  const agreed = mlReactor !== null && aiReactor !== null && mlReactor === aiReactor;
  const curKey = profileKey(mlB, aiB, agreed);
  const curLabel = profileLabel(mlB, aiB, agreed);

  // Недостаточно данных
  if (sampleCount < MIN_TOTAL_SAMPLES) {
    return {
      reactor: mlReactor ?? aiReactor,
      confidence: 0,
      mode: "insufficient",
      profileKey: curLabel,
      profileAccuracy: null,
      profileSamples: 0,
      sampleCount,
      explanation: `Мало данных (${sampleCount}/${MIN_TOTAL_SAMPLES} раундов)`,
    };
  }

  // Строим статистику по профилям
  const stats = new Map<string, ProfileStat>();

  for (const r of records) {
    const rMlB = bucket(r.mlConfidenceBefore);
    const rAiB = bucket(r.aiConfidenceBefore);
    const rAgreed = r.predictedBefore === r.aiPredictedBefore;
    const key = profileKey(rMlB, rAiB, rAgreed);

    if (!stats.has(key)) {
      stats.set(key, { total: 0, consensusHits: 0, mlHits: 0, aiHits: 0, dominantIsML: r.mlConfidenceBefore >= r.aiConfidenceBefore });
    }
    const s = stats.get(key)!;
    s.total++;

    if (rAgreed) {
      if (r.predictedBefore === r.winner) s.consensusHits++;
    } else {
      if (r.predictionHit) s.mlHits++;
      if (r.aiPredictionHit) s.aiHits++;
    }
  }

  const stat = stats.get(curKey);

  // Нет наблюдений для текущего профиля — fallback на лидера по уверенности
  if (!stat || stat.total < MIN_PROFILE_SAMPLES) {
    const fallback = mlConfidence >= aiConfidence ? mlReactor : aiReactor;
    const fallbackConf = Math.max(mlConfidence, aiConfidence);
    return {
      reactor: fallback,
      confidence: fallbackConf,
      mode: "insufficient",
      profileKey: curLabel,
      profileAccuracy: stat ? stat.total > 0 ? (agreed ? stat.consensusHits / stat.total : Math.max(stat.mlHits, stat.aiHits) / stat.total) : null : null,
      profileSamples: stat?.total ?? 0,
      sampleCount,
      explanation: `Профиль «${curLabel}» встречался ${stat?.total ?? 0} раз — мало для вывода. Ставлю на лидера по уверенности`,
    };
  }

  // --- Профиль найден, данных достаточно ---

  if (agreed) {
    // Консенсус: смотрим точность консенсуса для этого профиля
    const acc = stat.consensusHits / stat.total;
    const consensusReactor = mlReactor; // оба одинаковы

    if (acc >= 0.55) {
      return {
        reactor: consensusReactor,
        confidence: acc,
        mode: "profile_trusted",
        profileKey: curLabel,
        profileAccuracy: acc,
        profileSamples: stat.total,
        sampleCount,
        explanation: `Профиль «${curLabel}»: консенсус верен в ${Math.round(acc * 100)}% (${stat.consensusHits}/${stat.total}) — доверяем`,
      };
    } else {
      const inverted: Reactor = consensusReactor === "alpha" ? "omega" : "alpha";
      return {
        reactor: inverted,
        confidence: 1 - acc,
        mode: "profile_inverted",
        profileKey: curLabel,
        profileAccuracy: acc,
        profileSamples: stat.total,
        sampleCount,
        explanation: `Профиль «${curLabel}»: консенсус верен лишь в ${Math.round(acc * 100)}% — инвертируем`,
      };
    }
  } else {
    // Расхождение: кто точнее при этом профиле?
    const mlAcc = stat.mlHits / stat.total;
    const aiAcc = stat.aiHits / stat.total;

    if (mlAcc >= aiAcc) {
      return {
        reactor: mlReactor,
        confidence: mlAcc,
        mode: "profile_pick_ml",
        profileKey: curLabel,
        profileAccuracy: mlAcc,
        profileSamples: stat.total,
        sampleCount,
        explanation: `Профиль «${curLabel}»: ML точнее при расхождении (${Math.round(mlAcc * 100)}% vs ИИ ${Math.round(aiAcc * 100)}%)`,
      };
    } else {
      return {
        reactor: aiReactor,
        confidence: aiAcc,
        mode: "profile_pick_ai",
        profileKey: curLabel,
        profileAccuracy: aiAcc,
        profileSamples: stat.total,
        sampleCount,
        explanation: `Профиль «${curLabel}»: ИИ точнее при расхождении (${Math.round(aiAcc * 100)}% vs ML ${Math.round(mlAcc * 100)}%)`,
      };
    }
  }
}

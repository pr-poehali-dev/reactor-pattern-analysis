/**
 * EnsemblePredictor v1
 *
 * 5 методов предсказания + взвешенный ансамбль:
 *   1. Марковская цепь порядка K (автоподбор K=1..6 по кросс-валидации)
 *   2. LZ-суффиксный метод (самое длинное совпадение с концом)
 *   3. Случайный лес (Random Forest, in-browser, на признаках)
 *   4. Градиентный бустинг (simplified GBM, деревья решений с additive scoring)
 *   5. kNN с расстоянием Хэмминга (динамическое окно 2..10)
 *
 * Ансамбль:
 *   - Взвешенное голосование, веса = точность на последних 20% истории
 *   - Rolling retraining: модель обновляется после каждого раунда
 *
 * Диагностика:
 *   - Confusion matrix, точность по классам, точность по времени
 *   - Важность признаков (Random Forest)
 */

import type { Reactor, RoundResult } from "./screenAnalyzer";

// ────────────────────────── Типы ──────────────────────────

export interface MethodResult {
  name: string;
  reactor: Reactor;
  confidence: number; // 0..1 (вероятность предсказанного класса)
  available: boolean;
}

export interface EnsemblePrediction {
  reactor: Reactor;
  confidence: number;
  reason: string;
  methods: MethodResult[];
  bestMethod: string;
  ensembleWeights: Record<string, number>;
  confusionMatrix: ConfusionMatrix;
  featureImportance: FeatureImportance[];
  accuracyByClass: { alpha: number | null; omega: number | null };
  recentAccuracyCurve: number[]; // точность по блокам по 10 раундов
  accuracy: number | null; // общая точность ансамбля
}

export interface ConfusionMatrix {
  // predicted → actual
  alphaAlpha: number; // predicted alpha, actual alpha ✓
  alphaOmega: number; // predicted alpha, actual omega ✗
  omegaAlpha: number; // predicted omega, actual alpha ✗
  omegaOmega: number; // predicted omega, actual omega ✓
}

export interface FeatureImportance {
  name: string;
  label: string;
  importance: number; // 0..1
}

// ────────────────────── Признаки ──────────────────────────

interface Features {
  last1: number;       // 1=alpha, 0=omega
  last2: number;
  last3: number;
  last4: number;
  streakLen: number;   // длина текущей серии
  streakAlpha: number; // 1 если серия alpha
  alphaRatio10: number; // доля alpha за последние 10
  altRate: number;     // частота чередования за 8 раундов
  roundNum: number;    // номер раунда (нормировано /100)
  timeSinceSwitch: number; // сколько раундов с последней смены (нормировано /10)
  flickerRate: number; // темп мерцания текущего раунда
  flickerBias: number; // смещение мерцания
  flickerRateDelta: number; // разница с предыдущим раундом
  flickerRateMA3: number;  // скользящее среднее темпа за 3 раунда
  flickerBin: number;  // бинаризованный темп: 0=низкий(<0.5), 0.5=средний, 1=высокий(>2)
  pattern0110: number; // встречается ли паттерн 0110 в хвосте
  pattern1001: number; // встречается ли паттерн 1001 в хвосте
}

function encodeReactor(r: Reactor): number {
  return r === "alpha" ? 1 : 0;
}

function extractFeatures(
  history: RoundResult[],
  flickerBias: number,
  flickerRate: number
): Features {
  const n = history.length;
  const get = (i: number): number =>
    n > i ? encodeReactor(history[n - 1 - i].winner) : 0.5;

  const last1 = get(0);
  const last2 = get(1);
  const last3 = get(2);
  const last4 = get(3);

  // Длина серии
  let streakLen = 0;
  const streakSide: Reactor = n > 0 ? history[n - 1].winner : null;
  for (let i = n - 1; i >= 0; i--) {
    if (history[i].winner === streakSide) streakLen++;
    else break;
  }
  const streakAlpha = streakSide === "alpha" ? 1 : 0;

  // Доля alpha за 10
  const last10 = history.slice(-10);
  const alphaRatio10 = last10.length > 0
    ? last10.filter(r => r.winner === "alpha").length / last10.length
    : 0.5;

  // Чередование за 8
  let alternations = 0;
  for (let i = 1; i < Math.min(n, 8); i++) {
    if (history[n - i].winner !== history[n - i - 1].winner) alternations++;
  }
  const altRate = Math.min(n - 1, 7) > 0 ? alternations / Math.min(n - 1, 7) : 0;

  // Время с последней смены
  let timeSinceSwitch = 0;
  if (n > 1) {
    for (let i = n - 1; i > 0; i--) {
      if (history[i].winner === history[i - 1].winner) timeSinceSwitch++;
      else break;
    }
  }

  // Мерцание: дельта и скользящее среднее
  const prevRate = n >= 2 ? history[n - 1].flickerRate : flickerRate;
  const flickerRateDelta = flickerRate - prevRate;
  const maRates = history.slice(-3).map(r => r.flickerRate);
  const flickerRateMA3 = maRates.length > 0
    ? maRates.reduce((a, b) => a + b, 0) / maRates.length
    : flickerRate;
  const flickerBin = flickerRate < 0.5 ? 0 : flickerRate < 2 ? 0.5 : 1;

  // Паттерны 0110 / 1001
  const tail = history.slice(-4).map(r => encodeReactor(r.winner));
  const str = tail.join("");
  const pattern0110 = str.includes("0110") ? 1 : 0;
  const pattern1001 = str.includes("1001") ? 1 : 0;

  return {
    last1, last2, last3, last4,
    streakLen: Math.min(streakLen, 10) / 10,
    streakAlpha,
    alphaRatio10,
    altRate,
    roundNum: Math.min(n, 100) / 100,
    timeSinceSwitch: Math.min(timeSinceSwitch, 10) / 10,
    flickerRate: Math.min(flickerRate, 5) / 5,
    flickerBias,
    flickerRateDelta: Math.max(-2, Math.min(2, flickerRateDelta)) / 2,
    flickerRateMA3: Math.min(flickerRateMA3, 5) / 5,
    flickerBin,
    pattern0110,
    pattern1001,
  };
}

function featuresToArray(f: Features): number[] {
  return [
    f.last1, f.last2, f.last3, f.last4,
    f.streakLen, f.streakAlpha, f.alphaRatio10, f.altRate,
    f.roundNum, f.timeSinceSwitch,
    f.flickerRate, f.flickerBias, f.flickerRateDelta, f.flickerRateMA3, f.flickerBin,
    f.pattern0110, f.pattern1001,
  ];
}

const FEATURE_LABELS: { name: keyof Features; label: string }[] = [
  { name: "last1", label: "Предыдущий результат" },
  { name: "last2", label: "2й с конца" },
  { name: "last3", label: "3й с конца" },
  { name: "last4", label: "4й с конца" },
  { name: "streakLen", label: "Длина серии" },
  { name: "streakAlpha", label: "Серия на Alpha" },
  { name: "alphaRatio10", label: "Доля Alpha / 10 раундов" },
  { name: "altRate", label: "Частота чередования" },
  { name: "roundNum", label: "Номер раунда" },
  { name: "timeSinceSwitch", label: "Раундов без смены" },
  { name: "flickerRate", label: "Темп мерцания" },
  { name: "flickerBias", label: "Смещение мерцания" },
  { name: "flickerRateDelta", label: "Дельта темпа" },
  { name: "flickerRateMA3", label: "Скользящее среднее темпа" },
  { name: "flickerBin", label: "Темп: низкий/сред/высокий" },
  { name: "pattern0110", label: "Паттерн 0110" },
  { name: "pattern1001", label: "Паттерн 1001" },
];

// ──────────────── 1. Марковская цепь ──────────────────────

function markovPredict(history: RoundResult[], K: number): { prob: number } {
  // prob — вероятность alpha
  const winners = history.map(r => r.winner).filter((r): r is "alpha" | "omega" => r !== null);
  const n = winners.length;
  if (n <= K) return { prob: 0.5 };

  const tail = winners.slice(-K).join(",");
  let alphaCount = 0;
  let omegaCount = 0;

  for (let i = 0; i <= n - K - 1; i++) {
    const seg = winners.slice(i, i + K).join(",");
    if (seg === tail) {
      if (winners[i + K] === "alpha") alphaCount++;
      else omegaCount++;
    }
  }

  const total = alphaCount + omegaCount;
  if (total === 0) return { prob: 0.5 };
  return { prob: alphaCount / total };
}

// Подбираем K кросс-валидацией на истории (leave-one-out по скользящему окну)
function selectBestK(history: RoundResult[]): number {
  const winners = history.map(r => r.winner).filter((r): r is "alpha" | "omega" => r !== null);
  const n = winners.length;
  if (n < 10) return 1;

  let bestK = 1;
  let bestAcc = 0;

  for (let K = 1; K <= Math.min(6, n - 3); K++) {
    let hits = 0;
    let total = 0;
    // Скользящая валидация: train=[0..i-1], predict i
    for (let i = K + 1; i < n; i++) {
      const sub = winners.slice(0, i);
      const tail = sub.slice(-K).join(",");
      let ac = 0; let oc = 0;
      for (let j = 0; j <= i - K - 1; j++) {
        const seg = sub.slice(j, j + K).join(",");
        if (seg === tail) {
          if (sub[j + K] === "alpha") ac++;
          else oc++;
        }
      }
      if (ac + oc === 0) continue;
      const pred = ac >= oc ? "alpha" : "omega";
      if (pred === winners[i]) hits++;
      total++;
    }
    const acc = total > 0 ? hits / total : 0;
    if (acc > bestAcc) { bestAcc = acc; bestK = K; }
  }

  return bestK;
}

// ──────────────── 2. LZ-суффиксный ────────────────────────

function lzSuffixPredict(history: RoundResult[]): { prob: number } {
  const winners = history.map(r => r.winner).filter((r): r is "alpha" | "omega" => r !== null);
  const n = winners.length;
  if (n < 3) return { prob: 0.5 };

  // Ищем самое длинное совпадение суффикса истории с любым ранним отрезком
  for (let matchLen = Math.min(n - 1, 20); matchLen >= 1; matchLen--) {
    const suffix = winners.slice(-matchLen);
    let alphaCount = 0;
    let omegaCount = 0;

    for (let i = 0; i <= n - matchLen - 1; i++) {
      const seg = winners.slice(i, i + matchLen);
      if (seg.every((v, j) => v === suffix[j])) {
        if (winners[i + matchLen] === "alpha") alphaCount++;
        else omegaCount++;
      }
    }

    if (alphaCount + omegaCount >= 2) {
      const total = alphaCount + omegaCount;
      // Взвешиваем уверенность по длине совпадения
      const rawProb = alphaCount / total;
      // При коротком совпадении — тянем к 0.5
      const weight = Math.min(matchLen / 5, 1);
      return { prob: 0.5 + (rawProb - 0.5) * weight };
    }
  }

  return { prob: 0.5 };
}

// ──────────────── Дерево решений (базис для RF и GBM) ─────

interface TreeNode {
  featureIdx: number;
  threshold: number;
  left: TreeNode | null;
  right: TreeNode | null;
  value: number | null; // вероятность alpha в листе
}

function buildDecisionTree(
  X: number[][],
  y: number[], // 1=alpha, 0=omega
  maxDepth: number,
  minSamples: number,
  featureSubset?: number[] // для Random Forest
): TreeNode {
  const n = X.length;
  if (n === 0) return { featureIdx: 0, threshold: 0, left: null, right: null, value: 0.5 };

  const alphaSum = y.reduce((s, v) => s + v, 0);
  const leafValue = alphaSum / n;

  if (maxDepth === 0 || n <= minSamples) {
    return { featureIdx: 0, threshold: 0, left: null, right: null, value: leafValue };
  }

  const nFeatures = X[0].length;
  const candidates = featureSubset ?? Array.from({ length: nFeatures }, (_, i) => i);

  let bestGain = -Infinity;
  let bestFeat = 0;
  let bestThresh = 0;

  const gini = (subset: number[]): number => {
    if (subset.length === 0) return 0;
    const p = subset.reduce((s, v) => s + v, 0) / subset.length;
    return 2 * p * (1 - p);
  };

  for (const fi of candidates) {
    const values = X.map(x => x[fi]);
    const sorted = [...new Set(values)].sort((a, b) => a - b);
    for (let ti = 0; ti < sorted.length - 1; ti++) {
      const thresh = (sorted[ti] + sorted[ti + 1]) / 2;
      const leftY: number[] = [];
      const rightY: number[] = [];
      for (let i = 0; i < n; i++) {
        if (X[i][fi] <= thresh) leftY.push(y[i]);
        else rightY.push(y[i]);
      }
      if (leftY.length === 0 || rightY.length === 0) continue;
      const gain = gini(y) - (leftY.length / n) * gini(leftY) - (rightY.length / n) * gini(rightY);
      if (gain > bestGain) { bestGain = gain; bestFeat = fi; bestThresh = thresh; }
    }
  }

  if (bestGain <= 0) {
    return { featureIdx: 0, threshold: 0, left: null, right: null, value: leafValue };
  }

  const leftX: number[][] = []; const leftY: number[] = [];
  const rightX: number[][] = []; const rightY: number[] = [];
  for (let i = 0; i < n; i++) {
    if (X[i][bestFeat] <= bestThresh) { leftX.push(X[i]); leftY.push(y[i]); }
    else { rightX.push(X[i]); rightY.push(y[i]); }
  }

  // Вычисляем subset для следующего уровня (RF: sqrt признаков)
  const nextSubset = featureSubset
    ? sampleFeatures(nFeatures, Math.max(2, Math.floor(Math.sqrt(nFeatures))))
    : undefined;

  return {
    featureIdx: bestFeat,
    threshold: bestThresh,
    left: buildDecisionTree(leftX, leftY, maxDepth - 1, minSamples, nextSubset),
    right: buildDecisionTree(rightX, rightY, maxDepth - 1, minSamples, nextSubset),
    value: null,
  };
}

function predictTree(tree: TreeNode, x: number[]): number {
  if (tree.value !== null) return tree.value;
  if (x[tree.featureIdx] <= tree.threshold) return predictTree(tree.left!, x);
  else return predictTree(tree.right!, x);
}

function sampleFeatures(total: number, k: number): number[] {
  const arr = Array.from({ length: total }, (_, i) => i);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(pseudoRand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr.slice(0, k);
}

// Детерминированный псевдорандом (фиксированный seed для воспроизводимости)
let _seed = 42;
function pseudoRand(): number {
  _seed = (_seed * 1664525 + 1013904223) & 0xffffffff;
  return ((_seed >>> 0) / 0xffffffff);
}

function bootstrapSample(n: number): number[] {
  return Array.from({ length: n }, () => Math.floor(pseudoRand() * n));
}

// ──────────────── 3. Random Forest ────────────────────────

interface RandomForest {
  trees: TreeNode[];
  featureImportances: number[];
  nFeatures: number;
}

const rfCache: { forest: RandomForest | null; trainedOnN: number } = {
  forest: null,
  trainedOnN: 0,
};

function buildRandomForest(X: number[][], y: number[], nTrees = 20): RandomForest {
  _seed = 42; // фиксированный seed для стабильности
  const nFeatures = X[0]?.length ?? 0;
  const trees: TreeNode[] = [];
  const importances: number[] = new Array(nFeatures).fill(0);

  for (let t = 0; t < nTrees; t++) {
    const indices = bootstrapSample(X.length);
    const bX = indices.map(i => X[i]);
    const bY = indices.map(i => y[i]);
    const subset = sampleFeatures(nFeatures, Math.max(2, Math.floor(Math.sqrt(nFeatures))));
    const tree = buildDecisionTree(bX, bY, 5, 2, subset);
    trees.push(tree);

    // Важность признаков: считаем сколько раз каждый признак использовался в сплите
    const countFeature = (node: TreeNode): void => {
      if (node.value !== null) return;
      importances[node.featureIdx]++;
      if (node.left) countFeature(node.left);
      if (node.right) countFeature(node.right);
    };
    countFeature(tree);
  }

  const totalImp = importances.reduce((a, b) => a + b, 0) || 1;
  const normImportances = importances.map(v => v / totalImp);

  return { trees, featureImportances: normImportances, nFeatures };
}

function rfPredict(forest: RandomForest, x: number[]): number {
  const probs = forest.trees.map(t => predictTree(t, x));
  return probs.reduce((a, b) => a + b, 0) / probs.length;
}

// ──────────────── 4. Градиентный бустинг (GBM) ───────────

interface GBM {
  trees: TreeNode[];
  learningRate: number;
  initProb: number;
}

const gbmCache: { model: GBM | null; trainedOnN: number } = {
  model: null,
  trainedOnN: 0,
};

function buildGBM(X: number[][], y: number[], nTrees = 15, lr = 0.15): GBM {
  _seed = 137;
  const n = X.length;
  const initProb = y.reduce((a, b) => a + b, 0) / n;

  // Инициализируем остатки
  let residuals = y.map(yi => yi - initProb);
  const trees: TreeNode[] = [];

  for (let t = 0; t < nTrees; t++) {
    // Обучаем дерево на остатках
    const tree = buildDecisionTree(X, residuals, 3, 2);
    trees.push(tree);

    // Обновляем остатки
    residuals = residuals.map((r, i) => {
      const pred = predictTree(tree, X[i]);
      return r - lr * pred;
    });
  }

  return { trees, learningRate: lr, initProb };
}

function gbmPredict(model: GBM, x: number[]): number {
  let score = model.initProb;
  for (const tree of model.trees) {
    score += model.learningRate * predictTree(tree, x);
  }
  // Clamp к [0,1]
  return Math.max(0, Math.min(1, score));
}

// ──────────────── 5. kNN с расстоянием Хэмминга ──────────

function knnPredict(history: RoundResult[], windowSize: number, k: number): { prob: number } {
  const n = history.length;
  if (n <= windowSize) return { prob: 0.5 };

  const winners = history.map(r => r.winner).filter((r): r is "alpha" | "omega" => r !== null);
  if (winners.length <= windowSize) return { prob: 0.5 };

  const query = winners.slice(-windowSize);

  // Все возможные окна в истории (кроме последнего)
  const candidates: { dist: number; next: "alpha" | "omega" }[] = [];
  for (let i = 0; i <= winners.length - windowSize - 1; i++) {
    const window = winners.slice(i, i + windowSize);
    let dist = 0;
    for (let j = 0; j < windowSize; j++) {
      if (window[j] !== query[j]) dist++;
    }
    candidates.push({ dist, next: winners[i + windowSize] });
  }

  // Берём k ближайших
  candidates.sort((a, b) => a.dist - b.dist);
  const topK = candidates.slice(0, k);
  if (topK.length === 0) return { prob: 0.5 };

  // Взвешенное голосование: ближайшие имеют больший вес
  let alphaW = 0;
  let totalW = 0;
  for (const c of topK) {
    const w = 1 / (c.dist + 1);
    if (c.next === "alpha") alphaW += w;
    totalW += w;
  }

  return { prob: totalW > 0 ? alphaW / totalW : 0.5 };
}

// Оптимальный windowSize для kNN по кросс-валидации
function selectBestKnnWindow(history: RoundResult[]): number {
  const winners = history.map(r => r.winner).filter((r): r is "alpha" | "omega" => r !== null);
  const n = winners.length;
  if (n < 15) return 3;

  let bestW = 3;
  let bestAcc = 0;

  for (let w = 2; w <= Math.min(10, n - 3); w++) {
    let hits = 0; let total = 0;
    for (let i = w + 1; i < n; i++) {
      const sub = winners.slice(0, i);
      const query = sub.slice(-w);
      const cands: { dist: number; next: "alpha" | "omega" }[] = [];
      for (let j = 0; j <= i - w - 1; j++) {
        const seg = sub.slice(j, j + w);
        let d = 0;
        for (let k = 0; k < w; k++) if (seg[k] !== query[k]) d++;
        cands.push({ dist: d, next: sub[j + w] });
      }
      cands.sort((a, b) => a.dist - b.dist);
      const top5 = cands.slice(0, 5);
      if (top5.length === 0) continue;
      let aw = 0; let tw = 0;
      for (const c of top5) {
        const wt = 1 / (c.dist + 1);
        if (c.next === "alpha") aw += wt;
        tw += wt;
      }
      const pred = aw / tw >= 0.5 ? "alpha" : "omega";
      if (pred === winners[i]) hits++;
      total++;
    }
    const acc = total > 0 ? hits / total : 0;
    if (acc > bestAcc) { bestAcc = acc; bestW = w; }
  }

  return bestW;
}

// ──────────────── Ансамблевые веса ────────────────────────

const METHOD_KEYS = ["markov", "lz", "rf", "gbm", "knn"] as const;
type MethodKey = typeof METHOD_KEYS[number];

const methodHistory: Record<MethodKey, boolean[]> = {
  markov: [], lz: [], rf: [], gbm: [], knn: [],
};
const MAX_METHOD_HISTORY = 80;

function recordMethodResult(key: MethodKey, hit: boolean) {
  methodHistory[key].push(hit);
  if (methodHistory[key].length > MAX_METHOD_HISTORY) methodHistory[key].shift();
}

// Валидационные веса: точность на последних 20% истории (но минимум 8 раундов)
function getEnsembleWeights(history: RoundResult[]): Record<MethodKey, number> {
  const n = history.length;
  const valSize = Math.max(8, Math.floor(n * 0.2));

  const weights: Record<MethodKey, number> = {
    markov: 1, lz: 1, rf: 1, gbm: 1, knn: 1,
  };

  if (n < 12) return weights;

  for (const key of METHOD_KEYS) {
    const hist = methodHistory[key];
    if (hist.length < 5) continue;
    const recent = hist.slice(-valSize);
    const acc = recent.filter(Boolean).length / recent.length;
    // acc=50% → weight=0.5, acc=70% → weight=1.5, acc<40% → weight≈0.1
    weights[key] = Math.max(0.05, acc < 0.40 ? (acc - 0.30) : 0.5 + (acc - 0.5) * 2);
  }

  return weights;
}

// ──────────────── Диагностика ─────────────────────────────

function buildConfusionMatrix(history: RoundResult[]): ConfusionMatrix {
  let alphaAlpha = 0; let alphaOmega = 0;
  let omegaAlpha = 0; let omegaOmega = 0;

  for (const r of history) {
    if (r.winner === null || r.predictedBefore === null) continue;
    if (r.predictedBefore === "alpha" && r.winner === "alpha") alphaAlpha++;
    else if (r.predictedBefore === "alpha" && r.winner === "omega") alphaOmega++;
    else if (r.predictedBefore === "omega" && r.winner === "alpha") omegaAlpha++;
    else if (r.predictedBefore === "omega" && r.winner === "omega") omegaOmega++;
  }

  return { alphaAlpha, alphaOmega, omegaAlpha, omegaOmega };
}

function buildAccuracyCurve(history: RoundResult[], blockSize = 10): number[] {
  const blocks: number[] = [];
  for (let i = 0; i < history.length; i += blockSize) {
    const block = history.slice(i, i + blockSize);
    const valid = block.filter(r => r.predictionHit !== null);
    if (valid.length === 0) continue;
    blocks.push(valid.filter(r => r.predictionHit === true).length / valid.length);
  }
  return blocks;
}

// ──────────────── Основная функция ────────────────────────

let lastEnsemblePrediction: EnsemblePrediction | null = null;

// Записываем точность методов по итогам раунда
export function recordEnsembleResult(
  actual: Reactor,
  prevMethods: MethodResult[]
) {
  if (!actual) return;
  for (const m of prevMethods) {
    if (!m.available || m.reactor === null) continue;
    const key = m.name as MethodKey;
    if (METHOD_KEYS.includes(key)) {
      recordMethodResult(key, m.reactor === actual);
    }
  }
}

// Получить последнее предсказание (для записи точности)
export function getLastEnsemblePrediction(): EnsemblePrediction | null {
  return lastEnsemblePrediction;
}

export function ensemblePredict(
  history: RoundResult[],
  flickerBias: number,
  flickerRate: number
): EnsemblePrediction {
  const n = history.length;
  const feats = extractFeatures(history, flickerBias, flickerRate);
  const xVec = featuresToArray(feats);

  // ── Подготовка обучающей выборки ──────────────────────
  const MIN_TRAIN = 8;
  const X: number[][] = [];
  const y: number[] = [];

  // Rolling window: последние 100 раундов для обучения
  const trainHistory = history.slice(-100);
  for (let i = 1; i < trainHistory.length; i++) {
    const sub = trainHistory.slice(0, i);
    const prevR = trainHistory[i - 1];
    const f = extractFeatures(sub, prevR.flickerBias, prevR.flickerRate);
    X.push(featuresToArray(f));
    y.push(encodeReactor(trainHistory[i].winner));
  }

  // ── 1. Марков ─────────────────────────────────────────
  const bestK = n >= 10 ? selectBestK(history) : 1;
  const markovProb = markovPredict(history, bestK).prob;

  // ── 2. LZ-суффикс ─────────────────────────────────────
  const lzProb = lzSuffixPredict(history).prob;

  // ── 3. Random Forest ──────────────────────────────────
  let rfProb = 0.5;
  if (X.length >= MIN_TRAIN) {
    // Пересобираем только при появлении новых данных (rolling retraining каждый раунд)
    if (n !== rfCache.trainedOnN || !rfCache.forest) {
      rfCache.forest = buildRandomForest(X, y, 25);
      rfCache.trainedOnN = n;
    }
    rfProb = rfPredict(rfCache.forest, xVec);
  }

  // ── 4. GBM ────────────────────────────────────────────
  let gbmProb = 0.5;
  if (X.length >= MIN_TRAIN) {
    if (n !== gbmCache.trainedOnN || !gbmCache.model) {
      gbmCache.model = buildGBM(X, y, 20, 0.15);
      gbmCache.trainedOnN = n;
    }
    gbmProb = gbmPredict(gbmCache.model!, xVec);
  }

  // ── 5. kNN ────────────────────────────────────────────
  const bestWindow = n >= 15 ? selectBestKnnWindow(history) : 3;
  const knnProb = knnPredict(history, bestWindow, 5).prob;

  // ── Сборка результатов методов ────────────────────────
  const methods: MethodResult[] = [
    {
      name: "markov",
      reactor: markovProb >= 0.5 ? "alpha" : "omega",
      confidence: Math.abs(markovProb - 0.5) * 2 * 0.42 + 0.5,
      available: n >= 3,
    },
    {
      name: "lz",
      reactor: lzProb >= 0.5 ? "alpha" : "omega",
      confidence: Math.abs(lzProb - 0.5) * 2 * 0.42 + 0.5,
      available: n >= 5,
    },
    {
      name: "rf",
      reactor: rfProb >= 0.5 ? "alpha" : "omega",
      confidence: Math.abs(rfProb - 0.5) * 2 * 0.42 + 0.5,
      available: X.length >= MIN_TRAIN,
    },
    {
      name: "gbm",
      reactor: gbmProb >= 0.5 ? "alpha" : "omega",
      confidence: Math.abs(gbmProb - 0.5) * 2 * 0.42 + 0.5,
      available: X.length >= MIN_TRAIN,
    },
    {
      name: "knn",
      reactor: knnProb >= 0.5 ? "alpha" : "omega",
      confidence: Math.abs(knnProb - 0.5) * 2 * 0.42 + 0.5,
      available: n >= bestWindow + 2,
    },
  ];

  // ── Ансамблевое голосование ───────────────────────────
  const weights = getEnsembleWeights(history);
  let alphaScore = 0;
  let totalWeight = 0;

  const probs: Record<MethodKey, number> = {
    markov: markovProb,
    lz: lzProb,
    rf: rfProb,
    gbm: gbmProb,
    knn: knnProb,
  };

  for (const key of METHOD_KEYS) {
    const m = methods.find(m => m.name === key)!;
    if (!m.available) continue;
    const w = weights[key];
    alphaScore += probs[key] * w;
    totalWeight += w;
  }

  const ensembleProb = totalWeight > 0 ? alphaScore / totalWeight : 0.5;
  const reactor: Reactor = ensembleProb >= 0.5 ? "alpha" : "omega";
  const rawConf = Math.max(ensembleProb, 1 - ensembleProb);
  const confidence = Math.min(0.93, 0.5 + (rawConf - 0.5) * 1.6);

  // Лучший метод по весу
  let bestMethod = "markov";
  let bestW = 0;
  for (const key of METHOD_KEYS) {
    if (weights[key] > bestW && methods.find(m => m.name === key)?.available) {
      bestW = weights[key];
      bestMethod = key;
    }
  }

  // ── Диагностика ───────────────────────────────────────
  const cm = buildConfusionMatrix(history);
  const accuracyCurve = buildAccuracyCurve(history);

  const totalPredicted = history.filter(r => r.predictionHit !== null).length;
  const totalHits = history.filter(r => r.predictionHit === true).length;
  const accuracy = totalPredicted >= 5 ? totalHits / totalPredicted : null;

  const alphaPred = cm.alphaAlpha + cm.alphaOmega;
  const omegaPred = cm.omegaAlpha + cm.omegaOmega;
  const accuracyByClass = {
    alpha: alphaPred > 0 ? cm.alphaAlpha / alphaPred : null,
    omega: omegaPred > 0 ? cm.omegaOmega / omegaPred : null,
  };

  // Важность признаков из RF
  const featureImportance: FeatureImportance[] = rfCache.forest
    ? FEATURE_LABELS.map((fl, i) => ({
        name: fl.name as string,
        label: fl.label,
        importance: rfCache.forest!.featureImportances[i] ?? 0,
      })).sort((a, b) => b.importance - a.importance)
    : [];

  // Описание методов для reason
  const methodNames: Record<string, string> = {
    markov: `Марков K=${bestK}`,
    lz: "LZ-суффикс",
    rf: "Random Forest",
    gbm: "GBM",
    knn: `kNN w=${bestWindow}`,
  };

  const voteSummary = methods
    .filter(m => m.available)
    .map(m => `${methodNames[m.name]}:${m.reactor === "alpha" ? "α" : "ω"}`)
    .join(" ");

  const reason = `${voteSummary} → ансамбль:${reactor === "alpha" ? "α" : "ω"} (${Math.round(confidence * 100)}%)`;

  const ensembleWeightsDisplay: Record<string, number> = {};
  for (const key of METHOD_KEYS) ensembleWeightsDisplay[key] = Math.round(weights[key] * 100) / 100;

  const result: EnsemblePrediction = {
    reactor,
    confidence,
    reason,
    methods,
    bestMethod,
    ensembleWeights: ensembleWeightsDisplay,
    confusionMatrix: cm,
    featureImportance,
    accuracyByClass,
    recentAccuracyCurve: accuracyCurve,
    accuracy,
  };

  lastEnsemblePrediction = result;
  return result;
}
